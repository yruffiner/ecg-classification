'''***********************************************
*
*       project: physioNet
*       created: 22.03.2017
*       purpose: abstract network interface-class
*
***********************************************'''

'''***********************************************
*           Imports
***********************************************'''

import tensorflow as tf
import numpy as np
import datetime as dt
import os
import json
import csv
import time

from definitions import *
import utils.dataset_helper as dsh
import utils.transformations as trans
import utils.nn_layers as nn
import utils.metrics as met
import utils.tf_helper as tfh


'''***********************************************
*           Classes
***********************************************'''

class Network:

    '''***********************************************
    *           Initialisation
    ***********************************************'''

    def __init__(self):

        #***************************************#
        #  from training job:                   #
        #***************************************#   

        self.job_name               = None
        self.job_description        = None
        self.worker                 = None
        self.cvid                   = None
        self.model                  = None
        self.split                  = None
        self.log_en                 = None
        self.log_test_score         = None

        #***************************************#
        #  from model:                          #
        #***************************************#

        self.model_name             = None

        # preprocessing
        self.spectrogram            = None
        self.nperseg                = None
        self.noverlap               = None

        # loss_function_parameters
        self.l2_penalty             = None
        self.class_penalty          = None

        # training_parameters
        self.learning_rate          = None
        self.batch_size             = None
        self.drop_rate              = None
        self.exponential_decay      = None
        self.dataset_compensation   = None
        self.validation_step        = None
        self.early_stop_wait        = None

        # data_augmentation
        self.resampling             = None
        self.resample_method        = None
        self.zero_filter            = None
        self.reload_step            = None
        self.awgn                   = None

        #***************************************#
        #  internal constants:                  #
        #***************************************#

        # max number of sequences trainable in 
        # one shot (VRAM limitation of GPU)
        self.max_s = 20

        #***************************************#
        #  internal variables:                  #
        #***************************************#

        # training phase: determines how the network
        # is routed (also for classification)
        self.phase = 0

        # for training
        self.termination_epoch = 0
        self.termination_cost = 0
        self.tmp_dir = None
        self.log_dir = None

        # input size parameters
        self.ext_len = None
        self.max_shape = None

        # dataset parameters
        self.n_classes = dsh.n_classes
        self.class_distribution = dsh.class_distribution
        self.class_tags = dsh.class_tags

    '''***********************************************
    *  Build network
    ***********************************************'''

    def build(self, longest_seq=dsh.longest_seq):

        # we work with the global default graph,
        # if .build() is called multiple times inside the same process,
        # multiple instances of the complete graph would exits inside the global default graph
        # as soon as a session runs, it claims its inputs have not been fed properly.
        # Because the default graph then has twice as many inputs as expected
        # therefore reset the default graph before each build,
        # such that this error cannot possibly occur again
        tf.reset_default_graph()

        # all signals are extended to the length of the longest sequence in the dataset
        data = dsh.load_data(longest_seq)
        self.ext_len = data.shape[1]
        print('Extension length: {:}'.format(self.ext_len))

        data = self.load_input(longest_seq)
        self.max_shape = [data.shape[1], data.shape[2]]

        with tf.device(default_dev):
            with tf.name_scope('inputs'):
                self.create_inputs()
            with tf.name_scope('model'):
                self.pred = self.create_model(self.data_subset)
            with tf.name_scope('cost_function'):
                self.cost = self.cost_function(self.pred, self.label_subset)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                with tf.name_scope('optim'):
                    if(self.exponential_decay):
                        step = tf.Variable(0, trainable=False)
                        rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                          global_step=step,
                                                          decay_steps=1,
                                                          decay_rate=0.9999)
                        optimizer = tf.train.AdamOptimizer(rate)
                        self.optimizerizer = optimizer.minimize(self.cost, aggregation_method=2, global_step=step)
                    else:
                        optimizer = tf.train.AdamOptimizer(self.learning_rate)
                        self.optimizerizer = optimizer.minimize(self.cost, aggregation_method=2)


    def create_inputs(self):
        [self.data_init, self.data] = tfh.create_input(dtype='int16', shape=[None, self.max_shape[0], self.max_shape[1]], name='data')
        [self.label_init, self.label] = tfh.create_input(dtype='float64', shape=[None, self.n_classes], name='labels')
        self.subset = tf.placeholder(dtype=tf.int32, shape=[None], name='batch_selector')
        self.dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')
        self.training_phase = tf.placeholder(dtype=tf.int32, shape=[], name='training_phase')
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
        data = tf.gather(self.data, self.subset)
        if self.awgn:
            data = tf.cond(
                self.is_training,
                lambda: nn.awgn_channel(data, snr=3),
                lambda: data
            )
        self.data_subset = data
        self.label_subset = tf.gather(self.label,  self.subset)


    def create_model(self, data, feat_s):
        raise NotImplementedError("Must be overridden with proper definition of forward path")


    def cost_function(self, pred, label):

        eq_w = [1 for _ in self.class_distribution]
        occ_w = [100/r for r in self.class_distribution]
        c = self.class_penalty
        weights = [[e * (1-c) + o * c for e,o in zip(eq_w, occ_w)]]
        class_weights = tf.constant(weights, dtype=tf.float32)
        # select cost multiplier for each signal
        weight_per_sig = tf.matmul(class_weights,
                                   tf.transpose(label))

        # generate l2 loss over all trainable weights (not biases)
        penal = tf.constant(self.l2_penalty, dtype=tf.float32)
        vars = tf.trainable_variables()
        vars = [v for v in vars if
                'bias' not in v.name and
                'batch_normalization' not in v.name and
                'cond' not in v.name]
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])

        softmax = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
        # scale with importance of respecting rare classes
        softmax = tf.matmul(weight_per_sig, tf.expand_dims(softmax, 1))
        # overall cost is classification loss + l2 penalty of learned parameters
        cost = tf.reduce_mean(softmax + penal * lossL2)

        return cost


    '''***********************************************
    *  Load and preprocess inputs
    ***********************************************'''

    def load_input(self, load_list, is_training=False):
        data = dsh.load_data(load_list, ext_len=self.ext_len, data2d=True)

        if is_training:
            if self.zero_filter:
                data = trans.zero_filter(data, threshold=2, depth=10)
            if self.resampling:
                data = trans.random_resample(data, upscale_factor=1)

        if self.spectrogram:
            data = trans.spectrogram(data, nperseg=self.nperseg, noverlap=self.noverlap)
        else:
            data = np.expand_dims(data, axis=2)

        return data

    def load_label(self, load_list, output_type='onehot'):
        label = dsh.load_label(load_list, output_type)
        return label

    '''***********************************************
    *  Main functions: train, classify, predict_score
    ***********************************************'''

    def train(self, epochs, phase=0):

        try:

            print('Start training phase', phase)

            self.phase = phase
            if self.log_en:
                self.setup_logger()
            self.new_session()
            if self.phase > 0:
                load_dir = os.path.join(self.log_dir, 'trained_phase{:0>1}'.format(self.phase-1))
                self.load_session(load_dir)
            max_valid_score = 0
            cost = 0

            for epoch in range(epochs):

                t = time.time()

                if epoch % self.validation_step == 0:
                    [vacc,vscore,_] = self.predict_score(run_set=self.valid_set)
                    print('Validation accuracy = {0:.2f}%'.format(vacc), '/ score = {0:.2f} %'.format(vscore))
                    self.do_log(tag='validation_acc_phase'+str(self.phase), value=vacc, epoch=epoch)
                    self.do_log(tag='validation_score_phase'+str(self.phase), value=vscore, epoch=epoch)
                    hot_log_fields = [epoch, self.termination_epoch, cost, vacc, vscore]
                    if self.log_test_score:
                        [tacc,tscore,_] = self.predict_score(run_set=self.test_set)
                        self.do_log(tag='test_acc_phase'+str(self.phase), value=tacc, epoch=epoch)
                        self.do_log(tag='test_score_phase'+str(self.phase), value=tscore, epoch=epoch)
                        hot_log_fields.extend([tacc, tscore])
                    self.hot_log(fields=hot_log_fields)
                    # if validation score improves: save network and update termination epoch/cost
                    if vscore >= max_valid_score:
                        print('Saving network...')
                        self.save_session(self.tmp_dir)
                        max_valid_score = vscore
                        self.termination_epoch = epoch
                        self.termination_cost = cost
                        print('Best score is {0:.2f}%'.format(max_valid_score),
                              '(achieved in Epoch {:0>4})'.format(self.termination_epoch))
                    # early stop if no improvement for 100 epochs
                    if epoch > self.termination_epoch + self.early_stop_wait and epoch > 300:
                        print('Early stopping triggered, as no improvement')
                        break

                cost = 0
                if epoch == 0 or (epoch & self.reload_step == 0 and (self.zero_filter or self.resampling)):
                    train_data = self.load_input(self.train_set, is_training=True)
                    train_label = self.load_label(self.train_set)
                self.sess.run(self.data.initializer, feed_dict={self.data_init: train_data})
                self.sess.run(self.label.initializer, feed_dict={self.label_init: train_label})
                batches = dsh.batch_splitter(train_data.shape[0], batch_s=self.batch_size, shuffle=True,
                                             labels=train_label, compensation_factor=self.dataset_compensation)
                for batch in batches:
                    [_, c] = self.sess.run([self.optimizerizer, self.cost],
                                      feed_dict={
                                         self.subset: batch,
                                         self.dropout_rate: self.drop_rate,  
                                         self.training_phase: self.phase,
                                         self.is_training: True
                                      })
                    cost += c

                cost = cost/len(batches)
                self.do_log(tag='cost_phase'+str(self.phase), value=cost, epoch=epoch)
                dur = time.time() - t
                print('Epoch {:0>4}:'.format(epoch+1), 'cost= {:.6f}'.format(cost), '({:.1f}s)'.format(dur))

            # after training store best parameters and the achieved accuracy/score
            self.load_session(self.tmp_dir)
            save_dir = os.path.join(self.log_dir, 'trained_phase{:0>1}'.format(self.phase))
            self.save_session(save_dir)
            scoreDict = self.genScoreDict()
            score_file = os.path.join(save_dir, 'scores.json')
            with open(score_file, 'w+') as fh:
                json.dump(scoreDict, fh, indent=4, sort_keys=True)

        except KeyboardInterrupt:
            print('KeyboardInterrupt: running training phase canceled')


    def classify(self, id_list, phase=0):
        self.phase = phase
        self.new_session()
        load_dir = os.path.join(self.log_dir, 'trained_phase{:0>1}'.format(self.phase))
        print(load_dir)
        self.load_session(load_dir)
        data = self.load_input(id_list)
        pred_prob = self.predict(data)
        pred_int = self.prob2label(pred_prob)
        pred_label = [self.class_tags[pred] for pred in pred_int]
        return [pred_label, pred_int, pred_prob]


    def predict_score(self, run_set):
        data = self.load_input(run_set)
        act_label = self.load_label(run_set, output_type='int')
        pred_prob = self.predict(data)
        pred_label = self.prob2label(pred_prob)
        acc = met.compute_accuracy(pred_label, act_label)
        [score, scdict] = met.compute_score(pred_label, act_label, class_tags=self.class_tags, verbose=True)
        return [acc, score, scdict]


    def predict(self, data):
        self.sess.run(self.data.initializer, feed_dict={self.data_init: data})
        self.sess.run(self.label.initializer, feed_dict={self.label_init: [range(self.n_classes)]})
        batches = dsh.batch_splitter(len(data), batch_s=self.max_s, shuffle=False)
        predict_list = [self.sess.run(self.pred, feed_dict={
                                self.subset: batch,
                                self.dropout_rate: 0,
                                self.training_phase: self.phase,
                                self.is_training: False 
                            })
                       for batch in batches]
        predictions = np.concatenate(predict_list, axis=0)
        return predictions

    def prob2label(self, prob_vec):
        prediction =  np.argmax(prob_vec, axis=1)
        return prediction

    '''***********************************************
    *  Logging
    ***********************************************'''

    def setup_logger(self):
        tf_log_dir = os.path.join(self.log_dir, 'tboard')
        if not os.path.exists(tf_log_dir):
            os.makedirs(tf_log_dir)
        logname = 'phase{:0>1}_'.format(self.phase) + dt.datetime.now().strftime('%Y_%m_%d_%H%M%S')
        logfile = os.path.join(tf_log_dir, logname)
        self.logger = tf.summary.FileWriter(logfile, graph=tf.get_default_graph())

    def do_log(self, tag, value, epoch=None):
        if self.log_en:
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=value)
            self.logger.add_summary(summary,epoch)
            self.logger.flush()

    def hot_log(self, fields):
        logfile = os.path.join(self.log_dir, 'hotlog.csv')
        with open(logfile, 'a+') as fh:
            writer = csv.writer(fh)
            writer.writerow(fields)

    '''***********************************************
    *  Create/save/load a session
    ***********************************************'''

    def new_session(self):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(init, feed_dict={self.training_phase:self.phase})

    def save_session(self, save_dir):
        saver = tf.train.Saver()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver.save(self.sess, os.path.join(save_dir, 'session'))

    def load_session(self, load_dir):
        loader = tf.train.Saver()
        loader.restore(self.sess, os.path.join(load_dir, 'session'))

    '''***********************************************
    *  Load job from json-file
    ***********************************************'''

    def load_job(self, job):

        # load job and corresponding model+split
        self.jobFromDict(job_dict=job)
        self.load_model(model=self.model)
        self.load_split(split=self.split, cvid=self.cvid)

        # create directories
        self.tmp_dir = os.path.join(root, tmp_dir, self.worker)
        model_dir = os.path.join(root, log_dir, self.model_name)
        job_dir = os.path.join(model_dir, self.job_name)
        self.log_dir = os.path.join(job_dir, 'fold' + str(self.cvid))
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # save job and model into log folder
        job_save_file = os.path.join(self.log_dir, 'job.json')
        with open(job_save_file, 'w+') as fh:
            json.dump(self.jobToDict(), fh, indent=4, sort_keys=True)
        model_save_file = os.path.join(self.log_dir, 'model.json')
        with open(model_save_file, 'w+') as fh:
            json.dump(self.modelToDict(), fh, indent=4, sort_keys=True)

    def load_model(self, model):
        model_path = model_fmt.format(model)
        print('Loading model from:', model_path)
        with open(model_path) as fh:
            model_str = fh.read()
        model_dict = json.loads(model_str)
        self.modelFromDict(model_dict)

    def load_split(self, split, cvid):
        split_path = dsh.split_fmt.format(split)
        print('Loading split from:', split_path)
        [self.train_set, self.valid_set, self.test_set, self.holdout_set] = dsh.load_split(split_path, self.cvid)
        self.train_set = self.set_filter(self.train_set)
        self.valid_set = self.set_filter(self.valid_set)
        self.test_set = self.set_filter(self.test_set)
        self.holdout_set = self.set_filter(self.holdout_set)
        print('Train set:  ', self.train_set.shape[0], self.train_set[0:20])
        print('Valid set:  ', self.valid_set.shape[0], self.valid_set[0:10])
        print('Test set:   ', self.test_set.shape[0], self.test_set[0:10])
        print('Holdout set:', self.holdout_set.shape[0], self.holdout_set[0:10])

    def set_filter(self, ids):
        return ids

    '''***********************************************
    *  Dict conversion functions
    ***********************************************'''

    def genScoreDict(self):
        [tr_acc, tr_sc, tr_dict] = self.predict_score(run_set=self.train_set)
        [va_acc, va_sc, va_dict] = self.predict_score(run_set=self.valid_set)
        [te_acc, te_sc, te_dict] = self.predict_score(run_set=self.test_set)
        score_dict = {
            'scoring':
            {
                'acc_train' : tr_acc,
                'sc_train'  : tr_sc,
                'acc_valid' : va_acc,
                'sc_valid'  : va_sc,
                'acc_test'  : te_acc,
                'sc_test'   : te_sc
            },
            'split_scoring':
            {
                'train'     : tr_dict,
                'valid'     : va_dict,
                'test'      : te_dict
            },
            'termination':
            {
                'termination_epoch': self.termination_epoch,
                'termination_cost': self.termination_cost
            }
        }
        return score_dict

    def jobFromDict(self, job_dict):
        self.job_name           = job_dict['name']
        self.job_description    = job_dict['description']
        self.model              = job_dict['model']
        self.split              = job_dict['split']
        self.log_en             = job_dict['log_en']
        self.log_test_score     = job_dict['log_test_score']
        self.cvid               = job_dict['cvid']
        self.worker             = job_dict['worker']

    def jobToDict(self):
        job_dict = {
            'name'              : self.job_name,
            'description'       : self.job_description,
            'model'             : self.model,
            'split'             : self.split,
            'log_en'            : self.log_en,
            'log_test_score'    : self.log_test_score,
            'cvid'              : self.cvid,
            'worker'            : self.worker
        }
        return job_dict

    def modelFromDict(self, model_dict):
        self.model_name = model_dict['model_name']
        mp = model_dict['model_parameters']
        pp = model_dict['preprocessing']
        lp = model_dict['loss_function_parameters']
        tp = model_dict['training_parameters']
        da = model_dict['data_augmentation']
        self.set_modelParameters(mp)
        self.spectrogram            = pp['spectrogram']
        self.nperseg                = pp['nperseg']
        self.noverlap               = pp['noverlap']
        self.l2_penalty             = lp['l2_penalty']
        self.class_penalty          = lp['class_penalty']
        self.learning_rate          = tp['learning_rate']
        self.batch_size             = tp['batch_size']
        self.drop_rate              = tp['drop_rate']
        self.exponential_decay      = tp['exponential_decay']
        self.dataset_compensation   = tp['dataset_compensation']
        self.validation_step        = tp['validation_step']
        self.early_stop_wait        = tp['early_stop_wait']
        self.resampling             = da['resampling']
        self.resample_method        = da['resample_method']
        self.zero_filter            = da['zero_filter']
        self.reload_step            = da['reload_step']
        self.awgn                   = da['awgn']

    def set_modelParameters(self, param_dict):
        raise NotImplementedError("Must be overridden by specific model, parametrize model from dictionary")

    def modelToDict(self):
        model_dict = {
            'model_name': self.model_name,
            'model_parameters': self.get_modelParameters(),
            'preprocessing':
            {
                'spectrogram'           : self.spectrogram,
                'nperseg'               : self.nperseg,
                'noverlap'              : self.noverlap
            },
            'loss_function_parameters':
            {
                'l2_penalty'            : self.l2_penalty,
                'class_penalty'         : self.class_penalty
            },
            'training_parameters':
            {
                'learning_rate'         : self.learning_rate,
                'batch_size'            : self.batch_size,
                'drop_rate'             : self.drop_rate,
                'exponential_decay'     : self.exponential_decay,
                'dataset_compensation'  : self.dataset_compensation,
                'validation_step'       : self.validation_step,
                'early_stop_wait'       : self.early_stop_wait
            },
            'data_augmentation': {
                'resampling'            : self.resampling,
                'resample_method'       : self.resample_method,
                'zero_filter'           : self.zero_filter,
                'reload_step'           : self.reload_step,
                'awgn'                  : self.awgn
            }
        }
        return model_dict

    def get_modelParameters(self):
        raise NotImplementedError("Must be overridden by specific model, dump all architectural parameters")