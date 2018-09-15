# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

def leaky_relu(x, leaky=0.1):
    pos = 0.5 * (1 + leaky)
    neg = 0.5 * (1 - leaky)
    return pos * x + neg * tf.abs(x)

def prelu(x):
    alpha = tf.get_variable('relu_alpha', shape=x.get_shape()[-1],
                            initializer=tf.zeros_initializer())
    pos = tf.nn.relu(x)
    neg = alpha * (x - tf.abs(x)) * 0.5
    return pos + neg


from tensorflow.python.training.moving_averages import assign_moving_average

def batch_norm_layer(x, is_training, eps=1e-05, decay=0.90, scale=True, scope='batch_norm'):
    with tf.variable_scope(scope), tf.device('/cpu:0'):
        params_shape = x.get_shape().as_list()[-1:]
        moving_mean = tf.get_variable('mean', shape=params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', shape=params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)
        mean, variance = tf.nn.moments(x, list(np.arange(len(x.shape) - 1)), name='moments')
        update_moving_mean = assign_moving_average(moving_mean, mean, decay)
        update_moving_variance = assign_moving_average(moving_variance, variance,decay)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)


        mean, variance = tf.cond(is_training, lambda :(mean, variance), lambda :(moving_mean, moving_variance))

        beta = tf.get_variable('beta', params_shape,
                               initializer=tf.zeros_initializer)
        gamma = tf.get_variable('gamma', params_shape,
                                initializer=tf.ones_initializer) if scale else None
        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)

        return x