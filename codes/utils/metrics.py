'''***********************************************
*
*       project: physioNet
*       created: 22.03.2017
*       purpose: calculate metrics to compare nets
*
***********************************************'''

'''***********************************************
*           Imports
***********************************************'''

import numpy as np

from definitions import *
import utils.dataset_helper as dsh

'''***********************************************
*           Functions
***********************************************'''

def maxAsIndexMatrix(a):
    out = (a == np.max(a, axis=1)[:,None]).astype(int)
    return out

def get_prediction(prob_vec):
    prediction =  np.argmax(prob_vec, axis=1)
    return prediction

def compute_accuracy(prediction, actual):
    if len(prediction)==0:
        accuracy = 0
    else:
        accuracy = 100 * sum(actual == prediction) / len(prediction)
    return accuracy

def compute_score(prediction, actual, class_tags, verbose=False):
    if len(prediction)==0:
        score = 0
        dictionary = {}
    else:
        f1 = []
        dictionary = {}
        n_classes = len(class_tags)
        for c in range(n_classes):
            [f, dict] = calculate_F1(prediction, actual, c, verbose=verbose, class_tags=class_tags)
            if class_tags[c] != '~':
                f1.append(f)
            dictionary.update(dict)
        f1 = np.array(f1)
        f1 = f1[~np.isnan(f1)]
        score = 100*sum(f1)/len(f1)
    return [score, dictionary]

def calculate_F1(index1, index2, val, verbose=False, class_tags=[]):
    b1 = np.where(index1 == val, 1, 0)
    b2 = np.where(index2 == val, 1, 0)
    nc = sum(b1*b2)
    n1 = sum(b1)
    n2 = sum(b2)
    f1 = 2*nc/(n1 + n2)
    dict = {class_tags[val]:{
            'score':100*f1,
            'pred':n1.item(),
            'actual':n2.item(),
            'correct':nc.item()}
            }
    if verbose:
        print(class_tags[val],
              '\tscore = {0:.0f} %'.format(100*f1),
              '\tpred:', n1,
              '\tactual:', n2,
              '\tcorrect:', nc)
    return [f1, dict]

'''***********************************************
*           Script
***********************************************'''
if __name__ == '__main__':
    pred = np.random.random([5,4])
    index = maxAsIndexMatrix(pred)
    print(pred)
    print('\n')
    print(index)