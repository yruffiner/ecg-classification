'''***********************************************
*
*       project: physioNet
*       created: 22.03.2017
*       purpose: nn building blocks
*
***********************************************'''

'''***********************************************
*           Imports
***********************************************'''

import utils.tf_helper as tfh
import tensorflow as tf
import scipy.signal as sig
import tensorflow.contrib.rnn as rnn


'''***********************************************
*           Variables
***********************************************'''

rnn_all_outputs = True

'''***********************************************
*           Functions
***********************************************'''

def conv2d_block(inputs, length, kernel_size=[3,3], n_channels=None, growth=0, dilation_rates=[1,1,1],
                 strides_end=[2,2], max_pooling=False, is_training=False, drop_rate=0):

    # inherit layer-width from input
    if n_channels is None:
        n_channels = tfh.get_static_shape(inputs)[3]

    conv = inputs
    max_pool_en = False
    strides = [1,1]
    depth = len(dilation_rates)

    for d in range(depth):

        if d == depth-1:
            n_channels = n_channels+growth
            if max_pooling:
                max_pool_en = True
            else:
                strides = strides_end

        conv = tf.layers.conv2d(
                    inputs=conv,
                    filters=n_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding='same',
                    dilation_rate=(1, dilation_rates[d]))

        conv = tf.layers.batch_normalization(
                    inputs=conv,
                    center=True,
                    scale=True,
                    training=is_training)

        conv = tf.nn.relu(conv)
        if max_pool_en:
            conv = tf.layers.max_pooling2d(
                    inputs=conv, 
                    pool_size=strides_end, 
                    strides=strides_end, 
                    padding='same')
        output = tf.layers.dropout(inputs=conv, rate=drop_rate)

    length = tf.floordiv((length+1),2)

    return [output, length]


def average_features(input, length):
    # as we use affine functions our zero padded datasets
    # are now padded with the bias of the previous layers
    # in order to get the mean of only meaningful data out
    # set the zero-padding part back to zero again
    data = tfh.set_dynamiczero(input, length)
    # as we have zero padded data,
    # reduce_mean would result into too small values for most sequences
    # therefore use reduce_sum and divide by actual length instead
    data = tf.reduce_sum(data, axis=1)
    divisor = tf.cast(length, tf.float32)
    divisor = tf.expand_dims(divisor, dim=1)
    output = tf.div(data, divisor)
    return output


def mean_branch(input, length, out_s):
    output = average_features(input, length)
    output = tf.layers.dense(inputs=output, units=out_s)
    output = tf.nn.relu(output)
    return output


def lstm_layer(data, length, n_neurons, n_layers, bidirectional=False, drop_rate=None):
    data = tfh.set_dynamiczero(data, length)
    seq_l = tfh.get_length(data)
    if bidirectional:
        # we concatenate forward and backward outputs to one output,
        # each must be only half the final size
        n_neurons = int(n_neurons/2)
        cell_fw = rnn.LSTMCell( n_neurons, state_is_tuple=True)
        cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=1-drop_rate)
        cell_fw = rnn.MultiRNNCell([cell_fw] * n_layers)
        cell_bw = rnn.LSTMCell( n_neurons, state_is_tuple=True)
        cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=1-drop_rate)
        cell_bw = rnn.MultiRNNCell([cell_bw] * n_layers)
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            data,
            sequence_length=seq_l,
            dtype=tf.float32
        )
        (output_fw, output_bw) = outputs
        if rnn_all_outputs:
            output_bw = average_features(output_bw, length)
            output_fw = average_features(output_fw, length)
        else:
            output_bw = output_bw[:,0,:]
            output_fw = tfh.get_dynamiclast(output_fw, seq_l)
        output = tf.concat([output_bw, output_fw], axis=1)
    else:
        cell = rnn.LSTMCell( n_neurons, state_is_tuple=True)
        cell = rnn.DropoutWrapper(cell, output_keep_prob=1-drop_rate)
        cell = rnn.MultiRNNCell([cell] * n_layers)
        output, _ = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=seq_l,
            inputs=data)
        output = tfh.get_dynamiclast(output, seq_l)
    return output


def awgn_channel(input, snr):
    # adds white gaussian noise to input, wherever input is not zero padded
    shape = tfh.get_static_shape(input)
    shape[0] = tfh.get_dynamic_shape(input)[0]

    dim = len(shape)
    l = tfh.get_length(input)
    # total energy of input signal
    e = tf.multiply(input, input)
    if dim==2:
        e = tf.reduce_sum(e, axis=1)
    else:
        e = tf.reduce_mean(e, axis=2)
        e = tf.reduce_sum(e, axis=1)
    # average power of input signal (neglecting zero padding in division)
    p = tf.div(e, tf.cast(l, tf.float32))
    snr = tf.constant(snr, tf.float32)
    stddev = tf.sqrt(tf.div(p, snr))
    # make 3d
    stddev = tf.expand_dims(stddev, 1)
    stddev = tf.expand_dims(stddev, 1)
    # generate noise of same shape
    noise = tf.random_normal( shape, mean=0.0, stddev=1, dtype=tf.float32)
    # each row of noise has its own stddev -> broadcast multiplication with stddev
    noise = tf.multiply(stddev, noise)
    # drop noise where zero padding in data
    noise = tfh.set_dynamiczero(noise, l)
    output = input + noise
    return output

'''***********************************************
*           Script
***********************************************'''

if __name__ == '__main__':
    pass
