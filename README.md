# Convolutional Recurrent Neural Networks for Electrocardiogram Classification

This is the implementation of the neural networks for electrocardiogram classification proposed in this [paper](https://arxiv.org/abs/1710.06122).

## Requirements

The code was tested on TensorFlow 1.0.0/Python 3.5.
To install the required python packages, run
```
pip install -r requirements.txt
```
For GPU support, use
```
pip install -r requirements_gpu.txt
```
Other versions of the packages might work, but were not tested.

## Get the data

The dataset we used was provided for the [2017 PhysioNet/CinC Challenge](https://physionet.org/challenge/2017/). To download the data and to have the right folder structure, switch to the `data` folder and run the script
```
./get_data.sh
```

## The framework

The framework is designed to experiment with different network architectures. Some code structures might seem complicated, but they help to have a clear organization of training jobs and a detailed log of all results.

The generic architecture of the CNN and the CRNN are defined in `codes/network`. The hyperparameters of the model and the training procedure are defined in a json-file in the folder `models`. The setups from the paper are predefined, the parameter names should be self-explanatory.

We always used the same split of the dataset, to ensure a fair comparison of different architectures. Other splits of the dataset can be generated with the python script `generateSplit.py`. The first two parameters determine the size of the test, validation, and training set. The last parameter is the seed used to generate the random split.

Before starting training, a job has to be defined in the folder `jobs`.

## Start training

Before starting, you have to adapt the file `codes/definitions.py` to your environment. The important parameters to set are `default_dev` and `GPU_devices` (see the file for more detail).

The file `jobs/your_machine.json` provided allows to reproduce the 5-fold CV experiments from the paper. All the folds `0,1,2,3,4` have to be activated in `cvids`. The proportions of the train/validation/test split is also chosen here (`split`). There is an option to turn off the tensorboard-log (`log_en`) since it uses a lot of disk space (a simple `.csv`-file with the learning curve is always available).

To start this training jobs, change to the `codes` folder and run
```
python train.py
```
This stores all the jobs (one per model per CV-fold) in a queue, and they get processed by the available GPUs.

If you change the code and want to debug it using the model `models/model_name`, you can simply run
```
python train.py model_name
```
This starts a single training-job with some fixed parameters.

## Visualize the results

After the first job is completed, the results can be found in the `log` folder. The folders are organized in the format `architecture/jobname/fold`. In each such folder the following files can be found:
* The tensorboard log, if it was activated for this job.
* The trained model that achieved the best validation score.
* The hotlog.csv is the simple csv-log.
* A copy of the model and the job that led to those results.

If you want to compare, how well different CNN configurations work, you can run from the code folder
```
python summarizeScores.py CNN
```
The same works for the CRNN.

## Citation

If you find this code useful for your research, please cite
```
@incollection{zihlmann2017convolutional,
	Author = {Zihlmann, Martin and Perekrestenko, Dmytro and Tschannen, Michael},
	Booktitle = {Computing in Cardiology (CinC)},
	Title = {Convolutional Recurrent Neural Networks for Electrocardiogram Classification},
	Year = {2017}}
```
and acknowledge this repository.
