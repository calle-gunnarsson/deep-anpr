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
    'variable_summaries',
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

# Utility functions
def weight_variable(shape):
  with tf.name_scope('weights'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial, name="W")
    variable_summaries(var)
  return var

def bias_variable(shape):
  with tf.name_scope('biases'):
    initial = tf.constant(0.1, shape=shape)
    var = tf.Variable(initial, name="B")
    variable_summaries(var)
  return var

def conv2d(x, W, stride=(1, 1), padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                      padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def conv_layer(input, size_in, size_out, ksize=(2, 2), stride=(2, 2), padding="SAME", name="conv"):
  with tf.name_scope(name):
    w = weight_variable([5, 5, size_in, size_out])
    b = bias_variable([size_out])
    act = tf.nn.relu(conv2d(input, w) + b)

    return max_pool(act, ksize=ksize, stride=stride)

def fc_layer(input, w_shape, b_shape, name="fc"):
  with tf.name_scope(name):
    w = weight_variable(w_shape)
    b = bias_variable(b_shape)
    act = tf.matmul(input, w) + b
    #tf.summary.histogram("activations", act)
    return act

def convolutional_layers():
    """
    Get the convolutional layers of the model.

    """
    IN = 1
    OUT = 48

    x = tf.placeholder(tf.float32, [None, None, None], name="x")
    x_image = tf.expand_dims(x, 3)

    tf.summary.image('input', x_image, 25)

    # Input Layer
    input = conv_layer(x_image, IN, OUT, name="input_layer")

    # Second layer
    conv1 = conv_layer(input, OUT, HEIGHT, ksize=(2, 1), stride=(2, 1), name="conv_layer1")

    # Third layer
    conv2 = conv_layer(conv1, HEIGHT, WIDTH, name="conv_layer2")

    return (x, conv2)


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

    dense1 = fc_layer(conv_layer_flat,
                                 [32 * 8 * WIDTH, 2048],
                                 [2048],
                                 name="dense_layer1")
    h_dense1 = tf.nn.relu(dense1)
    #tf.summary.histogram('h_fc1', h_fc1)

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_drop = tf.nn.dropout(h_dense1, keep_prob, name="dropout")
    #tf.summary.histogram('h_fc1_drop', h_fc1)

    y = fc_layer(h_drop,
                 [2048, 1 + common.PLATE_LEN * len(common.CHARS)],
                 [1 + common.PLATE_LEN * len(common.CHARS)],
                 name="output_layer")

    return (x, y, keep_prob)


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
