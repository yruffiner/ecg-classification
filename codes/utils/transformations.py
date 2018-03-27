'''***********************************************
*
*       project: physioNet
*       created: 22.03.2017
*       purpose: data transformations
*
***********************************************'''

'''***********************************************
* Imports
***********************************************'''

import numpy as np
import scipy as sc
from scipy import signal
import math

from definitions import *

'''***********************************************
* External functions
***********************************************'''

def spectrogram(data, nperseg=32, noverlap=16):
    log_spectrogram = True
    fs = 300
    _, _, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.transpose(Sxx,[0,2,1])
    if log_spectrogram:
        Sxx = abs(Sxx)
        mask = Sxx > 0
        Sxx[mask] = np.log(Sxx[mask])
    return Sxx

def upscale(signals, upscale_factor=1):
    signals = np.repeat(signals, upscale_factor, axis=0)
    return signals

def random_resample(signals, upscale_factor=1):
    [n_signals,length] = signals.shape
    # pulse variation from 60 bpm to 120 bpm, expected 80 bpm
    new_length = np.random.randint(
        low=int(length*80/120),
        high=int(length*80/60),
        size=[n_signals, upscale_factor]
    )
    signals = [np.array(s) for s in signals.tolist()]
    new_length = [np.array(nl) for nl in new_length.tolist()]
    sigs = [stretch_squeeze(s,l) for s,nl in zip(signals,new_length) for l in nl]
    sigs = [fit_tolength(s, length) for s in sigs]
    sigs = np.array(sigs)
    return sigs

def random_resample_with_mean(signals, meanHRs):
    [n_signals,length] = signals.shape
    new_lengths = [np.random.randint(low=int(length*hr/120), high=int(length*hr/60)) for hr in meanHRs]
    signals = [np.array(s) for s in signals.tolist()]
    new_lengths = [np.array(nl) for nl in new_lengths]
    sigs = [stretch_squeeze(s,nl) for s,nl in zip(signals,new_lengths)]
    sigs = [fit_tolength(s, length) for s in sigs]
    sigs = np.array(sigs)
    return sigs

def resample_with_mean(signals, meanHRs):
    [n_signals,length] = signals.shape
    new_lengths = [int(length*hr/80) for hr in meanHRs]
    signals = [np.array(s) for s in signals.tolist()]
    new_lengths = [np.array(nl) for nl in new_lengths]
    sigs = [stretch_squeeze(s,nl) for s,nl in zip(signals,new_lengths)]
    sigs = [fit_tolength(s, length) for s in sigs]
    sigs = np.array(sigs)
    return sigs

def zero_filter(input, threshold=2, depth=8):
    shape = input.shape
    # compensate for lost length due to mask processing
    noise_shape = [shape[0], shape[1] + depth]
    noise = np.random.normal(0,1,noise_shape)
    mask = np.greater(noise, threshold)
    # grow a neighbourhood of True values with at least length depth+1
    for d in range(depth):
        mask = np.logical_or(mask[:, :-1], mask[:, 1:])
    output = np.where(mask, np.zeros(shape), input)
    return output

'''***********************************************
* Internal functions
***********************************************'''

def stretch_squeeze(source, length):
    target = np.zeros([1, length])
    interpol_obj = sc.interpolate.interp1d(np.arange(source.size), source)
    grid = np.linspace(0, source.size - 1, target.size)
    result = interpol_obj(grid)
    return result

def fit_tolength(source, length):
    target = np.zeros([length])
    w_l = min(source.size, target.size)
    target[0:w_l] = source[0:w_l]
    return target

'''***********************************************
*           Script
***********************************************'''

if __name__ == '__main__':
    pass


