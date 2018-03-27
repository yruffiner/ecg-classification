'''***********************************************
*
*       project: physioNet
*       created: 01.03.2017
*       purpose: Global settings/definitions
*		 - global variables
*		 - physical environment setup
*
***********************************************'''

import os
import socket


'''***********************************************
* Physical Environment
***********************************************'''

GPU = '/gpu:0'
CPU = '/cpu:0'
default_dev = GPU

# For multi GPU training use (4 GPUs in this example)
#     GPU_devices = ['0','1','2','3']
GPU_devices = ['0']

'''***********************************************
* Directories & Files
***********************************************'''

filedir = os.path.dirname(__file__)
root = os.path.join(filedir, '..')

data_dir = 'data'
log_dir = 'log'
tmp_dir = 'tmp'
model_dir = 'models'
job_dir = 'jobs'

job_file_name = 'your_machine'

job_file = os.path.join(root, job_dir, job_file_name + '.json')
model_fmt = os.path.join(root, model_dir, '{}' + '.json')

if not os.path.exists(os.path.join(root, tmp_dir)):
	os.makedirs(os.path.join(root, tmp_dir))
if not os.path.exists(os.path.join(root, log_dir)):
	os.makedirs(os.path.join(root, log_dir))