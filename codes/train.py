'''***********************************************
*
*       project: physioNet
*       file: train
*       created: 22.03.2017
*       purpose: main script to train networks
*
***********************************************'''

'''***********************************************
*           Imports
***********************************************'''

from definitions import *
import sys
import multiprocessing as mp
import os
import time
import json

from network.cnn import CNN
from network.crnn import CRNN

'''***********************************************
*           run training on queue
***********************************************'''

def train_queue():

    # generate/load jobs to be done
    jobs = load_jobs()
    print('jobs in queue:')
    [print('  ' + job['name'] + '(' + str(job['cvid']) + ')' ) for job in jobs]

    # store jobes in queue
    job_queue = mp.Queue(len(jobs))
    for job in jobs:
        job_queue.put(job)

    # Instantiate workers
    if default_dev == CPU:
        workers = [{'dev': '0', 'name': 'CPU0'}]
    else:
        workers = [ {'dev': dev, 'name': 'GPU' + dev} for dev in GPU_devices]
    print('workers used in this queue:')
    [print('  ' + worker['name']) for worker in workers]

    # generate and start a process for every available GPU
    processes = [mp.Process(name=worker['name']+'_process', target=run_worker, args=(worker, job_queue, ))
                 for worker in workers ]
    for process in processes:
        process.start()
        
    try:
        for process in processes:
            process.join()
    except:
        for process in processes:
            print('Terminanting', process.name)
            process.terminate()

    print('Main process terminated')


def load_jobs():
    with open(job_file) as fh:
        jobs_str = fh.read()
    jobs_dict = json.loads(jobs_str)
    # jobs over cv_ids, so the different jobs have a first run relatively soon
    # instead of running the same job len(cv_ids) times before touching an other config
    jobs = []
    for idx in range(5):
        for job_name, job_dict in jobs_dict.items():
            if idx < len(job_dict['cvids']):
                new_job = {
                    'name': job_name,
                    'description': job_dict['description'],
                    'model': job_dict['model'],
                    'split': job_dict['split'],
                    'log_en': job_dict['log_en'],
                    'log_test_score': job_dict['log_test_score'],
                    'cvid': job_dict['cvids'][idx]
                }
                jobs.append(new_job)
    return jobs


def run_worker(worker, queue):

    # setup environment
    outpath = os.path.normpath(os.path.join(root, tmp_dir, 'stdout'+worker['name']+'.out'))
    print('Worker ' + worker['name'] + ' starting. Output of this worker was redirected, to track it, open a new window and run ' +
            '\"tail -f ' + outpath + '\"')
    redirect_output(outpath)
    print('---------------------------------------------------------------------------')
    os.environ['CUDA_VISIBLE_DEVICES'] = worker['dev']
    print('GPU device', worker['dev'], 'used in process', os.getpid(), '( Worker', worker['name'], ')')
    print('Worker ' + worker['name'] + ' starting')

    # load and execute jobs until queue empty
    while not queue.empty():

        try:
            job = queue.get()
            job.update({'worker': worker['name']})
        except Exception:
            print('Exception while loading new job.')
            break;

        print('---------------------------------------------------------------------------')
        print(' Worker ' + worker['name'] + ' starting job ' + job['name'] + '(' + str(job['cvid']) + ')')
        print('---------------------------------------------------------------------------')

        try:
            train(job)
        except Exception as e:
            print(e)

        print(' Worker ' + worker['name'] + ' finished job ' + job['name'] + '(' + str(job['cvid']) + ')')

    print('Worker ' + worker['name'] + ' done.')


def redirect_output(filename):
    fd = os.open(filename, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    stdout = 1 # stdout
    errout = 2  # errout
    os.dup2(fd, stdout)
    os.dup2(fd, errout)

'''***********************************************
*           run single job
***********************************************'''

def train_single(model_file):
    job = {
            'name': model_file,
            'description': 'Running single model to debug',
            'model': model_file,
            'split': 'split_5_6_14',
            'log_en': False,
            'log_test_score': False,
            'cvid': 0,
            'worker': 'debugger'
        }
    train(job)

'''***********************************************
*           core training functions
***********************************************'''

def train(job):

    if 'CRNN' in job['model']:
        trainCRNN(job)
    elif 'CNN' in job['model']:
        trainCNN(job)
    elif 'HNNStage1' in job['model']:
        trainHNNStage1(job)
    elif 'HNNStage2R' in job['model']:
        trainHNNStage2R(job)
    elif 'HNNStage2' in job['model']:
        trainHNNStage2(job)
    else:
        print('[Error] No training function defined for this network')

def trainCRNN(job):
    network = CRNN()
    network.load_job(job)
    network.build()
    network.train(epochs=500, phase=0)
    network.train(epochs=100, phase=1)
    network.train(epochs=200, phase=2)
    network.learning_rate = network.learning_rate/10
    network.train(epochs=200, phase=3)
    network.learning_rate = network.learning_rate/10
    network.train(epochs=200, phase=4)

def trainCNN(job):
    network = CNN()
    network.load_job(job)
    network.build()
    network.train(epochs=500, phase=0)

def trainHNNStage1(job):
    network = HNNStage1()
    network.load_job(job)
    network.build()
    network.train(epochs=300, phase=0)

def trainHNNStage2(job):
    network = HNNStage2()
    network.load_job(job)
    network.build()
    network.train(epochs=500, phase=0)

def trainHNNStage2R(job):
    network = HNNStage2R()
    network.load_job(job)
    network.build()
    network.train(epochs=500, phase=0)
    network.train(epochs=100, phase=1)
    network.train(epochs=200, phase=2)
    network.learning_rate = network.learning_rate/10
    network.train(epochs=200, phase=3)
    network.learning_rate = network.learning_rate/10
    network.train(epochs=200, phase=4)

'''***********************************************
*           Script
***********************************************'''

if __name__ == '__main__':
    if len(sys.argv) < 2:
        train_queue()
    else:
        model_file = sys.argv[1]
        train_single(model_file)
