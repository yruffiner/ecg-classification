'''***********************************************
*
*       project: physioNet
*       created: 23.11.2017
*       purpose: print a score overview
*
***********************************************'''

'''***********************************************
*           Imports
***********************************************'''

import numpy as np
import os
import sys
import json
import collections
from definitions import *
import utils.dataset_helper as dsh

'''***********************************************
*           Variables
***********************************************'''

keys = {
    'acc_test'  : ['scoring', 'acc_test'],
    'sco_test'  : ['scoring', 'sc_test'],
    'acc_valid' : ['scoring', 'acc_valid'],
    'sco_valid' : ['scoring', 'sc_valid'],
    'acc_train' : ['scoring', 'acc_train'],
    'sco_train' : ['scoring', 'sc_train'],
    'Nactual'   : ['split_scoring', 'test', '~', 'actual'],
    'Npredict'  : ['split_scoring', 'test', '~', 'pred'],
    'Ncorrect'  : ['split_scoring', 'test', '~', 'correct'],
    'sco_normal': ['split_scoring', 'test', 'N', 'score'],
    'sco_afib'  : ['split_scoring', 'test', 'A', 'score'],
    'sco_other' : ['split_scoring', 'test', 'O', 'score'],
    'sco_noise' : ['split_scoring', 'test', '~', 'score']
}

scorekeys = ['acc_test', 'sco_test', 'acc_valid', 'sco_valid', 'acc_train', 'sco_train']
noisekeys = ['Nactual', 'Npredict', 'Ncorrect']

pkdict_set1 = ['sco_test', 'sco_normal', 'sco_afib', 'sco_other', 'sco_noise']
pkdict_set2 = ['sco_test', 'sco_normal', 'sco_afib', 'sco_other']
pkdict_set3 = ['sco_test', 'Nprecis', 'Nrecall']

pkdict = {}
pkdict['CNN']           = pkdict_set1
pkdict['CRNN']          = pkdict_set1
pkdict['HNN']           = pkdict_set1
pkdict['HNNStage1']     = pkdict_set3
pkdict['HNNStage2']     = pkdict_set2
pkdict['HNNStage2R']    = pkdict_set2

'''***********************************************
* Help Functions
***********************************************'''

def checkDir(network, rootdir, subdir=''):
    outpaths = []
    dirs = os.listdir(os.path.join(rootdir, subdir))
    dirs = [d for d in dirs if network in d]
    for d in dirs:
        relpath = os.path.join(subdir, d)
        abspath = os.path.join(rootdir, relpath)
        if 'fold0' in os.listdir(abspath):
            outpaths.append(relpath)
        else:
            outpaths = outpaths+checkDir(network, rootdir, relpath)
    return outpaths

def insertDict(insDict, keylist, value):

    if len(keylist)==0:
        return value

    if not isinstance(insDict, dict):
        dictStack = [{} for k in range(len(keylist)+1)]
        dictStack[0] = value
        for i in range(1,len(keylist)+1):
            dictStack[i] = {keylist[-i]:dictStack[i-1]}
        return dictStack[-1]  
    else:
        key = keylist[0]
        if key in insDict:
            insDict[key] = insertDict(insDict[key], keylist[1:], value)
        else:
            dictStack = [{} for k in keylist]
            dictStack[0] = value
            for i in range(1,len(keylist)):
                dictStack[i] = {keylist[-i]:dictStack[i-1]}
                # print(dictStack)
            insDict[key] = dictStack[-1]
        return insDict

def readFromDict(readDict, keylist):
    tmpDict = readDict
    for key in keylist:
        tmpDict = tmpDict[key]
    return tmpDict

def print_line(symb, length):
    for i in range(length):
        print(symb, end='')
    print('')

def print_distance(length):
    for i in range(length):
        print('')

'''***********************************************
* Main Functions
***********************************************'''

def summarize(network):

    printkeys=pkdict[network]

    rootdir = os.path.join(root, 'log', network)
    relpaths = checkDir(network, rootdir)
    phases = ['phase'+str(p) for p in range(10)]
    folds = ['fold'+str(f) for f in range(5)]

    dictPFR, dictPRF, dictRPF, dictRFP, dictFPR, dictFRP = {}, {}, {}, {}, {}, {}

    # extract all required values from score-dict

    for phase in phases:
        for fold in folds:
            for relpath in relpaths:

                checkdir = os.path.join(rootdir, relpath, fold, 'trained_'+phase)
                if os.path.exists(checkdir):
                    with open(os.path.join(checkdir,'scores.json')) as fh:
                        scoreStr = fh.read()
                        scoreDict = json.loads(scoreStr)

                        subDict = {}
                        for key in scorekeys:
                            var = readFromDict(scoreDict, keys[key])
                            subDict[key] = round(var,3)

                        if 'Nprecis' in printkeys:
                            for key in noisekeys:
                                subDict[key] = readFromDict(scoreDict, keys[key])
                            Nprecis = 100*subDict['Ncorrect']/subDict['Npredict']
                            subDict['Nprecis'] = round(Nprecis,3)
                            Nrecall = 100*subDict['Ncorrect']/subDict['Nactual']
                            subDict['Nrecall'] = round(Nrecall,3)

                        if 'sco_normal' in printkeys:
                            subDict['sco_normal'] = readFromDict(scoreDict, keys['sco_normal'])

                        if 'sco_afib' in printkeys:
                            subDict['sco_afib'] = readFromDict(scoreDict, keys['sco_afib'])

                        if 'sco_other' in printkeys:
                            subDict['sco_other'] = readFromDict(scoreDict, keys['sco_other'])

                        if 'sco_noise' in printkeys:
                            subDict['sco_noise'] = readFromDict(scoreDict, keys['sco_noise'])

                        dictPRF = insertDict(dictPRF, [phase, relpath, fold], subDict)
                        dictRPF = insertDict(dictRPF, [relpath, phase, fold], subDict)

    # calculate all means and standard deviations

    for phase in phases:
        if phase in dictPRF:
            phaseDict = dictPRF[phase]

            for relpath in relpaths:
                if relpath in phaseDict:
                    relpathDict = phaseDict[relpath]

                    valueDict = {}
                    for pk in printkeys:
                        valueDict[pk] = np.array([])
                    for fold in folds:
                        if fold in relpathDict:
                            foldDict = relpathDict[fold]
                            for pk in printkeys:
                                newval = foldDict[pk]
                                valueDict[pk] = np.append(valueDict[pk], newval)
                    meanDict, stdevDict = {}, {}
                    for pk in printkeys:
                        meanDict[pk] = round(np.mean(valueDict[pk]),3)
                        stdevDict[pk] = round(np.std(valueDict[pk]),3)
                    dictPRF[phase][relpath]['mean'] = meanDict
                    dictPRF[phase][relpath]['stdev'] = stdevDict
                    dictRPF[relpath][phase]['mean'] = meanDict
                    dictRPF[relpath][phase]['stdev'] = stdevDict

    folds.append('mean')
    folds.append('stdev')

    # print complete overview

    print_distance(10)
    linelength = 16*len(printkeys)+8
    for phase in phases:
        if phase in dictPRF:
            phaseDict = dictPRF[phase]

            print_line('*', linelength)
            print('* ', phase)
            print_line('*', linelength)
            print('* \t', end="")
            for pk in printkeys:
                print(pk, '\t', end="")
            print('')
            print_line('*', linelength)

            for relpath in relpaths:
                if relpath in phaseDict:
                    relpathDict = phaseDict[relpath]

                    print_line('-', linelength)
                    print(relpath)
                    print_line('-', linelength)

                    for fold in folds:
                        if fold in relpathDict:
                            foldDict = relpathDict[fold]

                            if fold == 'mean':
                                print_line('- ', int(linelength/2))

                            print(fold, '\t', end="")
                            for pk in printkeys:
                                print(foldDict[pk], '\t\t', end="")
                            print('')


    # print short overview
    
    print_line('-', linelength)
    print_distance(3)

    shortDict = {}
    for relpath in relpaths:
        shortDict[relpath] = {}
        shortDict[relpath]['mean'] = {}
        shortDict[relpath]['stdev'] = {}
    for phase in phases:
        if phase in dictPRF:
            phaseDict = dictPRF[phase]
            for relpath in relpaths:
                if relpath in phaseDict:
                    relpathDict = phaseDict[relpath]
                    # print(relpathDict['stdev'])
                    shortDict[relpath]['mean'] = relpathDict['mean']
                    shortDict[relpath]['stdev'] = relpathDict['stdev']

    print_line('~', linelength+40)
    print('{0: <40}'.format(''), end="")
    for pk in printkeys:
        print(pk, '\t', end="")
    print('')
    print_line('~', linelength+40)
    for relpath in relpaths:
        mean = shortDict[relpath]['mean']
        stdev = shortDict[relpath]['stdev']
        print('{0: <40}'.format(relpath), end="")
        for pk in printkeys:
            print('%2.3f ±%1.3f\t' % (mean[pk], stdev[pk]), end="")
        print('')
    print_line('~', linelength+40)


    save_file = os.path.join(root, 'log', network, 'summary.json')
    with open(save_file, 'w+') as fh:
        json.dump(dictPRF, fh, indent=4, sort_keys=True)


'''***********************************************
*           Script
***********************************************'''

def main():
    network = sys.argv[1]
    summarize(network)

if __name__ == '__main__':
    main()
