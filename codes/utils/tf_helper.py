'''***********************************************
*
*       project: physioNet
*       created: 10.03.2017
*       purpose: abstract tf peculiarities away
*
***********************************************'''


'''***********************************************
*           Imports
***********************************************'''

from definitions import *
import os
import tensorflow as tf
import datetime as dt
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

'''***********************************************
*           Variables
***********************************************'''

gpu_prec = tf.float32

'''***********************************************
*           Yannick functions
# ***********************************************'''

def create_constant(value, type):
    const = tf.constant(value, dtype=type)
    cast = tf.cast(const, gpu_prec)
    return cast

def create_variable(name, shape):
    var = tf.Variable(tf.random_normal(shape), name=name, trainable=True)
    tf.add_to_collection('vars', var)
    return var

def create_input(dtype, shape, name):
    init_name = '_'.join([name,'init'])
    init = tf.placeholder(dtype, shape, name=init_name)
    cast = tf.cast(init, gpu_prec)
    var = tf.Variable(cast, name=name, trainable=False, collections=[], validate_shape=False)
    var.set_shape(shape)
    tf.add_to_collection('inputs', var)
    tf.add_to_collection('inputs', init)
    return [init, var]

def get_dynamic_shape(tensor):
    return tf.shape(tensor)

def get_static_shape(tensor):
    return tensor.get_shape().as_list()

def get_dynamiclast(output, seq_l):
    rng = tf.range(0, tf.shape(seq_l)[0])
    indexes = tf.stack([rng, seq_l - 1], 1)
    relevant = tf.gather_nd(output, indexes)
    return relevant

def get_staticlast(output, seq_max_l):
    output = tf.transpose(output, [1, 0, 2])
    last = output[seq_max_l - 1]
    return last

def set_dynamiczero(input, length):
    shape = get_static_shape(input)
    shape[0] = get_dynamic_shape(input)[0]
    dim = len(shape)
    # generate 2d Matrix of same shape with col_n in each entry
    r = tf.range(0, shape[1], 1)
    r = tf.expand_dims(r, 0)
    if dim==2:
        r = tf.tile(r, [shape[0],1])
    else:
        r = tf.expand_dims(r, 2)
        r = tf.tile(r, [shape[0], 1, shape[2]])
    # generate 2d Matrix with length of each dataset in batch in row
    l = tf.expand_dims(length, 1)
    if dim==2:
        l = tf.tile(l, [1, shape[1]])
    else:
        l = tf.expand_dims(l, 2)
        l = tf.tile(l, [1,shape[1], shape[2]])
    # when col_n smaller than length mask entry is true
    mask = tf.less(r, l)
    # when col_n larger than length, set input to zero
    output = tf.where(mask, input, tf.zeros(shape, dtype=gpu_prec))
    return output

def get_length(sequence):
    shape = get_static_shape(sequence)
    dim = len(shape)
    if dim==2:
        used = tf.sign(tf.abs(sequence))
    elif dim==3:
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    else:
        raise ValueError('only 2D or 3D sequences supported')
    used = tf.cast(used, tf.int32)
    rng = tf.range(0, shape[1])
    ranged =  rng * used
    length = tf.reduce_max(ranged, axis=1) + 1
    return length

def where_greater(input, threshold, replacement):
    shape = get_static_shape(input)
    shape[0] = get_dynamic_shape(input)[0]
    condition = tf.greater(input, threshold)
    rep = tf.ones(shape, dtype=tf.int32) * replacement
    output = tf.where(condition, input, rep)
    return output

def get_prediction(net_output):
    prediction =  tf.argmax(net_output, axis=1)
    return prediction

def truncate_static(data, size):
    trunc = data[:,0:size]
    return trunc

def truncate_dynamic(data):
    length = get_length(data)
    print(length)
    size = tf.reduce_max(length)
    trunc = data[:,0:size]
    return trunc

def compute_score(prediction, label):
    pred = get_prediction(prediction)
    lab = get_prediction(label)
    f1 = [calculate_F1(pred, lab, i) for i in range(4)]
    not_nan = tf.less(f1, 1)
    f1 = tf.where(not_nan, f1, [0,0,0,0])
    f = tf.reduce_sum(f1)/tf.reduce_sum(tf.cast(not_nan, gpu_prec))
    return f

def calculate_F1(indexA, indexB, val):
    b1 = tf.equal(indexA, val)
    b2 = tf.equal(indexB, val)
    both_true = tf.logical_and(b1, b2)
    n1 = tf.reduce_sum(tf.cast(b1, gpu_prec))
    n2 = tf.reduce_sum(tf.cast(b2, gpu_prec))
    ncorrect = tf.reduce_sum(tf.cast(both_true, gpu_prec))
    f1 = 2*ncorrect/(n1 + n2)
    return f1


'''***********************************************
*           Script
***********************************************'''
