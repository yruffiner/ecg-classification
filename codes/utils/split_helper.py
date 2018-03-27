'''***********************************************
*
*       project: physioNet
*       created: 23.11.2017
*       purpose: helper functions to split dataset
*
***********************************************'''

'''***********************************************
*           Imports
***********************************************'''

import numpy as np
import math
import functools

from definitions import *

'''***********************************************
* External Functions
***********************************************'''

def get_properties(holdout, n_folds, tv_frac, seed):

    name = "split_%d_%d_%d" % (n_folds,tv_frac,seed)
    if holdout:
        name = name + "_holdout"

    size_relative = get_relative_size(holdout, n_folds, tv_frac)
    size_sum = np.sum(size_relative)
    size_fraction = [round(sr/size_sum,5) for sr in size_relative]

    properties = {
            'name': name,
            'inputs': {
                'holdout': holdout,
                'number of folds': n_folds,
                '(train+valid)/valid': tv_frac,
                'seed': seed
            },
            'relative size': {
                'train': size_relative[0],
                'valid': size_relative[1],
                'test': size_relative[2],
                'holdout': size_relative[3]                
            },
            'relative size fraction': {
                'train': size_fraction[0],
                'valid': size_fraction[1],
                'test': size_fraction[2],
                'holdout': size_fraction[3]               
            }
    }

    return properties


def ins_id_into_fname(fname, id):
    fname = fname[:-5] + str(id) + fname[-5:]
    return fname


def stratified_split(id_list, labels, rel_size, shuffle=False, seed=None):
    id_list = np.array(id_list)
    # split id_list into a set for each label
    class_sets = split_bylabels(id_list, labels)
    n_classes = len(class_sets)
    # for each label, split corresponding data into rel_size splits
    split_class_sets = [split_set(class_set, rel_size=rel_size, shuffle=shuffle, seed=seed) for class_set in class_sets]
    # concatenate the splits for the different labels again into a set
    # but keeping the rel_size splits apart
    #              ClassA    ClassB    ClassC
    #   Set0        1/3   +   1/3   +   1/3
    #   Set1        1/2   +   1/2   +   1/2
    #   Set2        1/6   +   1/6   +   1/6
    split_sets = [np.sort(np.hstack([split_class_sets[i][j] for i in range(n_classes)])) for j in range(len(rel_size))]
    return split_sets


def adjoint_set(parent_set, sibling_set):
    pset = np.array(parent_set)
    sset = np.array(sibling_set)
    set = np.setdiff1d(pset, sset)
    return set

'''***********************************************
* Internal Functions
***********************************************'''

def get_relative_size(holdout, n_folds, tv_frac):
    if holdout:
        denom = (n_folds+1)*n_folds*tv_frac
        [holdout_n, rest] = minisplit(denom,n_folds+1)
        [test_n, rest] = minisplit(rest,n_folds)
        [valid_n, train_n] = minisplit(rest,tv_frac)
    else:
        denom = n_folds*tv_frac
        holdout_n = 0
        [test_n, rest] = minisplit(denom,n_folds)
        [valid_n, train_n] = minisplit(rest,tv_frac)
    size_relative = div_by_gcd([train_n, valid_n, test_n, holdout_n])
    return size_relative

def minisplit(total,splitfactor):
    smallpart = int(total/splitfactor)
    bigpart = total-smallpart
    return [smallpart, bigpart]

def div_by_gcd(values):
    gcd = functools.reduce(math.gcd, values)
    values = [int(v/gcd) for v in values]
    return values

def split_bylabels(id_list, labels):
    n_classes = labels.shape[1]
    # generate a mask for each feature, selecting only those entries which are labelled accordingly
    masks = [labels[:, i].astype(bool) for i in range(n_classes)]
    # split dataset into subsets, one for each feature in labels
    class_sets = [id_list[mask] for mask in masks]
    return class_sets

def split_set(inset, rel_size, shuffle=False, seed=None):

    rel_size = div_by_gcd(rel_size)
    rel_size_sum = sum(rel_size)
    inset = np.array(inset).astype(int)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(inset)
        np.random.seed(seed=None)

    small_sets = np.array_split(inset, rel_size_sum)
    out_sets = []
    [out_sets.append([]) for idx,_ in enumerate(rel_size)]

    # guarantee at least one small_set per outset
    index = 0
    for idx,size in enumerate(rel_size):
        if size>0:
            out_sets[idx].append(small_sets[index])
            index+=1
        else:
            out_sets[idx].append([])
    rel_size = [r-1 for r in rel_size]
    for idx,size in enumerate(rel_size):
        if size>0:
            [out_sets[idx].append(small_sets[index+i]) for i in range(size)]
            index+=size
        else:
            out_sets[idx].append([])

    out_sets = [np.concatenate(s) for s in out_sets]
    out_sets = [np.sort(s).astype(int) for s in out_sets]

    return out_sets