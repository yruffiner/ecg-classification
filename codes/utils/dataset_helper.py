'''***********************************************
*
*       project: physioNet
*       created: 21.11.2017
*       purpose: interface to the physioNet database
*
***********************************************'''

'''***********************************************
*           Imports
***********************************************'''

import os
import sys
import numpy as np
import scipy.io as sio
import json
import math
import parse

from definitions import *
import utils.split_helper as sh

'''***********************************************
*           Variables
***********************************************'''

version = 3

# directories
# db_dir = '.' if entry else os.path.join(data_dir, 'training2017')
# mat_dir = '.' if entry else 'mat_files'
db_dir = os.path.join(data_dir, 'training2017')
mat_dir = 'mat_files'
hea_dir = 'hea_files'
hr_dir = 'heartrates'
split_dir = 'splits'

# different formats
data_fmt = 'A{:0>5}'
hea_fmt = os.path.join(root, db_dir, hea_dir, data_fmt + '.hea')
mat_fmt = os.path.join(root, db_dir, mat_dir, data_fmt + '.mat')
hr_fmt = os.path.join(root, db_dir, hr_dir, data_fmt + '_{}' + '.json')
ref_fmt = os.path.join(root, db_dir, '{}.csv')
rec_fmt = os.path.join(root, db_dir, '{}')
split_fmt= os.path.join(root, db_dir, split_dir, '{}')

# version specific variables
ref_files = ['REFERENCE-original', 'REFERENCE', 'REFERENCE-v3']
class_distributions = [[60.44, 9.04, 29.98, 0.54],
                       [59.22, 8.65, 28.80, 3.33],
                       [59.52, 8.89, 28.32, 3.27]]

# records and reference file
rec_file = rec_fmt.format('RECORDS')
ref_file = ref_fmt.format(ref_files[version-1])

# class information
n_classes = 4
class_distribution = class_distributions[version-1]
class_tags = ['N', 'A', 'O', '~']
class_descriptions = ['normal', 'AF pathology', 'other pathology', 'noise']

# special signals
n_sequences = 8528
shortest_seq = 5493
longest_seq = 5736
min_seq_length = 2714
max_seq_length = 18286

'''***********************************************
* Main load functions
***********************************************'''

def load_data(load_list, ext_len=None, data2d=True):

    if load_list == []:
        return []

    id_list = convert_to_id_list(load_list)
    file_list = [mat_fmt.format(id) for id in id_list]
    signals = np.array([load_signal(file) for file in file_list])

    if ext_len is not None:
        signals = [extend_signal(sig,ext_len) for sig in signals]
        signals = np.vstack(signals)

    if len(signals) == 1 and not data2d:
        signals = signals[0]

    return signals


def load_label(load_list, output_type='onehot'):

    if load_list == [] or np.array_equal(load_list, []):
        return []

    id_list = convert_to_id_list(load_list)

    with open(ref_file) as fh:
        ref_content = fh.readlines()

    line_fmt = '{},{}\n'
    labels = [parse.parse(line_fmt, ref_content[id-1])[1] for id in id_list]
        
    if output_type is 'int' or 'onehot':
        labels = np.array([label_str_to_int(label) for label in labels])
    
    if output_type is 'onehot':
        labels = [label_int_to_onehot(label) for label in labels]
        labels = np.vstack(labels)

    if len(labels) == 1:
        labels = labels[0]

    return labels

'''***********************************************
* Load helper functions
***********************************************'''

def load_signal(path):
    dict = sio.loadmat(path)
    signal = dict['val']
    return signal[0,:]

def extend_signal(signal, length):
    extended = np.zeros(length)
    siglength = np.min([length, signal.shape[0]])
    extended[:siglength] = signal[:siglength]
    return extended 

'''***********************************************
* Format conversion Functions
***********************************************'''

def convert_to_id_list(conv_list):
    # if single integer or string, convert to list
    if isinstance(conv_list, (str, int)):
        conv_list = [conv_list]
    # turn strings into integers
    if isinstance(conv_list[0], str):
        conv_list = [name2id(name) for name in conv_list]
    return conv_list

def name2id(name):
    [id_str] = parse.parse(data_fmt, name)
    id = int(id_str)
    return id

def label_str_to_int(str_label):    
    return class_tags.index(str_label)

def label_int_to_onehot(int_label):
    onehot = np.zeros(n_classes)
    onehot[int_label] = 1
    return onehot

'''***********************************************
* Batch split function
***********************************************'''

def batch_splitter(set_size, batch_s, shuffle=False, labels=None, compensation_factor=0, pretraining=False):
    if set_size==0:
        return []
    else:
        set = np.array(range(set_size))
        if labels is not None and not pretraining:
            masks = [labels[:,i].astype(bool) for i in range(labels.shape[1])]
            sets = [set[mask] for mask in masks]
            lst = []
            for idx, set in enumerate(sets):
                scale = int(100*compensation_factor/class_distribution[idx]) + 1
                set = np.matlib.repmat(set, scale, 1)
                set = set.reshape([-1,1])
                lst.append(set)
            set = np.vstack(lst)
            set = set.squeeze()
            np.random.shuffle(set)
            set = set[0:set_size]
            set = np.sort(set)
        set_size = set.shape[0]
        n_batches = math.ceil(set_size / batch_s)
        if shuffle:
            np.random.shuffle(set)
        batches = np.array_split(set, n_batches)
        return batches

'''***********************************************
* Dataset split generation & loading
***********************************************'''

def gen_split(holdout, n_folds, tv_frac, seed):

    properties = sh.get_properties(holdout, n_folds, tv_frac, seed)

    save_dir = split_fmt.format(properties['name'])
    holdout_file = os.path.join(save_dir, 'holdout.json')
    test_file_base = os.path.join(save_dir, 'test.json')
    valid_file_base = os.path.join(save_dir, 'valid.json')
    train_file_base = os.path.join(save_dir, 'train.json')

    # prevent overwriting an existing split
    if os.path.exists(save_dir):
        print('Error: Split with this name already exists!')
        sys.exit(1)
    os.makedirs(save_dir)

    prop_file = os.path.join(save_dir,'properties.json')
    with open(prop_file, 'w+') as fh:
        json.dump(properties, fh, indent=4, sort_keys=True)

    ids = range(1, n_sequences + 1)
    if holdout:
        sets = sh.stratified_split(
            id_list=ids,
            labels=load_label(ids),
            rel_size=[1]*(n_folds+1),
            shuffle=True,
            seed=seed
        )
        holdout_set = sets[0]
        sets = sets[1:]
        rest =  sh.adjoint_set(ids, holdout_set)
        fname = holdout_file
        with open(fname, 'w+') as fh:
            json.dump(holdout_set.tolist(), fh)
    else:
        sets = sh.stratified_split(
            id_list=ids,
            labels=load_label(ids),
            rel_size=[1]*n_folds,
            shuffle=True,
            seed=seed
        )
        rest = ids

    for idx, test_set in enumerate(sets):
        train_valid_set = sh.adjoint_set(rest, test_set)
        [train_set, valid_set] = sh.stratified_split(
            id_list=train_valid_set,
            labels=load_label(train_valid_set),
            rel_size=[(tv_frac-1), 1],
            shuffle=True,
            seed=seed
        )
        fname = sh.ins_id_into_fname(test_file_base, idx)
        with open(fname, 'w+') as fh:
            json.dump(test_set.tolist(), fh)
        fname = sh.ins_id_into_fname(train_file_base, idx)
        with open(fname, 'w+') as fh:
            json.dump(train_set.tolist(), fh)
        fname = sh.ins_id_into_fname(valid_file_base, idx)
        with open(fname, 'w+') as fh:
            json.dump(valid_set.tolist(), fh)


def load_split(name, cvid):

    load_dir = os.path.join(root, db_dir, split_dir, name)

    test_file_base = os.path.join(load_dir, 'test.json')
    test_file = sh.ins_id_into_fname(test_file_base, cvid)
    with open(test_file) as fh:
        test_set = np.array(json.load(fh))

    train_file_base = os.path.join(load_dir, 'train.json')
    train_file = sh.ins_id_into_fname(train_file_base, cvid)
    with open(train_file) as fh:
        train_set = np.array(json.load(fh))

    valid_file_base = os.path.join(load_dir, 'valid.json')
    valid_file = sh.ins_id_into_fname(valid_file_base, cvid)
    with open(valid_file) as fh:
        valid_set = np.array(json.load(fh))

    holdout_set = np.array([])
    if 'holdout' in name:
        holdout_file = os.path.join(load_dir, 'holdout.json')
        with open(holdout_file) as fh:
            holdout_set = np.array(json.load(fh))

    return [train_set, valid_set, test_set, holdout_set]

'''***********************************************
* Heartrate stuff
***********************************************'''

def load_heartrate(load_list):

    loadname = os.path.join(root, data_dir, 'training2017', 'heartrates', 'factors.json')
    with open(loadname) as fh:
        ref_content = fh.readlines()
    line_fmt = '{},{}\n'
    factor_sel = [int(parse.parse(line_fmt, ref_content[id-1])[1]) for id in range(1,n_sequences+1)]

    if load_list == []:
        return []

    id_list = convert_to_id_list(load_list)

    file_list = [hr_fmt.format(id, factor_sel[id-1]) for id in id_list]
    peaks_list = [np.genfromtxt(file, delimiter=',')[:-1] for file in file_list]
    dists_list = [peaks[1:]-peaks[:-1] for peaks in peaks_list]
    mean_list = [np.mean(dists) for dists in dists_list]
    hr_list = [300*60/mean for mean in mean_list]

    return hr_list

'''***********************************************
* Script
***********************************************'''

if __name__ == '__main__':
    pass