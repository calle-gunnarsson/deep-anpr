# Copyright (c) 2016 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Definition of the neural networks.

"""


__all__ = (
    'get_training_model',
    'get_detect_model',
)


import tensorflow as tf

import common

HEIGHT = common.IMAGE_SHAPE[0]
WIDTH = common.IMAGE_SHAPE[1]

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    #tf.summary.histogram('histogram', var)

def dense_layer(input, units, name="dense"):
  with tf.name_scope(name):
    act = tf.layers.dense(inputs=input, 
                          units=units, 
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          bias_initializer=tf.constant_initializer(0.1),
                          name=name,
                          activation=tf.nn.relu)
    return act

def conv2d_layer(input, filters, padding="SAME", name="conv2d", pool_size=(2, 2), stride=(2, 2)):
  with tf.name_scope(name):
    act = tf.layers.conv2d(input, 
                          filters=filters,
                          kernel_size=[5, 5],
                          bias_initializer=tf.constant_initializer(0.1),
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                          name=name,
                          padding=padding,
                          activation=tf.nn.relu
                          )

    return tf.layers.max_pooling2d(act,pool_size=pool_size, strides=stride, padding=padding)

def convolutional_layers():
    """
    Get the convolutional layers of the model.

    """
    OUT = 48

    x = tf.placeholder(tf.float32, [None, None, None], name="x")
    x_image = tf.reshape(x, [-1, HEIGHT, WIDTH, 1])

    tf.summary.image('input', x_image, 25)

    # First layer
    pool1  = conv2d_layer(x_image, OUT, name="layer1")

    # Second layer
    pool2 = conv2d_layer(pool1, HEIGHT, pool_size=(2, 1), stride=(2, 1), name="layer2")

    # Third layer
    pool3 = conv2d_layer(pool2, WIDTH, name="layer3")

    return x, pool3


def get_training_model():
    """
    The training model acts on a batch of WIDTHxHEIGHT windows, and outputs a (1 +
    common.PLATE_LEN * len(common.CHARS) vector, `v`. `v[0]` is the probability that a plate is
    fully within the image and is at the correct scale.

    `v[1 + i * len(common.CHARS) + c]` is the probability that the `i`'th
    character is `c`.

    """
    x, conv_layer = convolutional_layers()

    conv_layer_flat = tf.reshape(conv_layer, [-1, 32 * 8 * WIDTH])

    dense = dense_layer(conv_layer_flat, 2048, name="densely_connected_layer")
    #tf.summary.histogram('dense', dense)

    training = tf.placeholder(tf.bool, name="training")
    dropout = tf.layers.dropout(d1, rate=0.4, training=training, name="dropout")
    #tf.summary.histogram('dropout', dropout)

    y = dense_layer(dropout, 1 + common.PLATE_LEN * len(common.CHARS), name="output_layer")

    return (x, y, training)


def get_detect_model():
    """
    The same as the training model, except it acts on an arbitrarily sized
    input, and slides the WIDTHxHEIGHT window across the image in 8x8 strides.

    The output is of the form `v`, where `v[i, j]` is equivalent to the output
    of the training model, for the window at coordinates `(8 * i, 4 * j)`.

    """
    x, conv_layer = convolutional_layers()

    # Fourth layer
    W_fc1 = weight_variable([8 * 32 * WIDTH, 2048])
    W_conv1 = tf.reshape(W_fc1, [8,  32, WIDTH, 2048])
    b_fc1 = bias_variable([2048])
    h_conv1 = tf.nn.relu(conv2d(conv_layer, W_conv1,padding="VALID") + b_fc1)
    # Fifth layer
    W_fc2 = weight_variable([2048, 1 + common.PLATE_LEN * len(common.CHARS)])
    W_conv2 = tf.reshape(W_fc2, [1, 1, 2048, 1 + common.PLATE_LEN * len(common.CHARS)])
    b_fc2 = bias_variable([1 + common.PLATE_LEN * len(common.CHARS)])
    h_conv2 = conv2d(h_conv1, W_conv2) + b_fc2

    return (x, h_conv2)