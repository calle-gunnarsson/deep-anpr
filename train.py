#!/usr/bin/env python3
#
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
Routines for training the network.

"""


__all__ = (
    'train',
)


import glob
import itertools
import datetime
import sys
import time
import os

import cv2
import numpy
import tensorflow as tf

import common
import gen2
import gen
import model

MODELS_PATH = "models"

def code_to_vec(p, code):
    def char_to_vec(c):
        y = numpy.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y

    c = numpy.vstack([char_to_vec(c) for c in code])

    return numpy.concatenate([[1. if p else 0], c.flatten()])


def vec_to_code(v):
    return "".join(common.CHARS[i] for i in v)


def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        code = fname.split(os.sep)[1][9:15]
        p = fname.split(os.sep)[1][16] == '1'
        yield im, code_to_vec(p, code)


def unzip(b):
    xs, ys = list(zip(*b))
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def read_batches(batch_size):
    g = gen.generate_ims()
    print("Read batches")
    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            yield im, code_to_vec(p, c)

    while True:
        yield unzip(gen_vecs())


def read_batches2(batch_size):
    g = gen2.generate_ims()
    print("Read batches2")
    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            yield im, code_to_vec(p, c)

    while True:
        yield unzip(gen_vecs())

def get_loss(y, y_):
    # Calculate the loss from digits being incorrect.  Don't count loss from
    # digits that are in non-present plates.
    with tf.name_scope('xent'):
        d_logits = tf.reshape(y[:, 1:], [-1, len(common.CHARS)])
        d_labels = tf.reshape(y_[:, 1:], [-1, len(common.CHARS)])
        p_logits = y[:, :1]
        p_labels = y_[:, :1]

        with tf.name_scope('digits'):
            digits_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                logits=d_logits,
                                                labels=d_labels)
            digits_loss = tf.reshape(digits_loss, [-1, common.PLATE_LEN])
            digits_loss = tf.reduce_sum(digits_loss, 1)
            digits_loss *= (y_[:, 0] != 0)
            digits_loss = tf.reduce_sum(digits_loss)

        with tf.name_scope('presence'):
            # Calculate the loss from presence indicator being wrong.
            presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=p_labels)
            presence_loss = common.PLATE_LEN * tf.reduce_sum(presence_loss)

    #tf.summary.histogram('d_logits', d_logits) 
    #tf.summary.histogram('p_logits', p_logits) 
    return digits_loss, presence_loss, digits_loss + presence_loss

last_batch_idx = 0
last_batch_time = time.time()
def print_report(batch_size, batch_idx, r):
    global last_batch_idx
    global last_batch_time
    num_correct = numpy.sum(
                    numpy.logical_or(
                        numpy.all(r[0] == r[1], axis=1),
                        numpy.logical_and(r[2] < 0.5,
                                            r[3] < 0.5)))

    r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
    for b, c, pb, pc in zip(*r_short):
        print("{} {} <-> {} {}".format(vec_to_code(c), pc,
                                        vec_to_code(b), float(pb)))
    num_p_correct = numpy.sum(r[2] == r[3])

    print ("B{:3d} {:2.02f}% {:02.02f}% loss: {} (digits: {}, presence: {}) |{}|".format(
        batch_idx,
        100. * num_correct / (len(r[0])),
        100. * num_p_correct / len(r[2]),
        r[6],
        r[4],
        r[5],
        "".join("X "[numpy.array_equal(b, c) or (not pb and not pc)]
                                        for b, c, pb, pc in zip(*r_short))))

    batch_time = time.time()
    if last_batch_idx != batch_idx:
        print("time for {} batches {}".format(batch_size,
            60 * (last_batch_time - batch_time) /
                            (last_batch_idx - batch_idx)))
        last_batch_idx = batch_idx
        last_batch_time = batch_time

def train(learn_rate, report_steps, batch_size, initial_model=None, max_steps=0):
    tf.reset_default_graph()

    x, y, keep_prob = model.get_training_model()

    y_ = tf.placeholder(tf.float32, [None, common.PLATE_LEN * len(common.CHARS) + 1], name="labels")

    digits_loss, presence_loss, loss = get_loss(y, y_)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    saver = tf.train.Saver()

    accuracy = None
    correct = None

    with tf.name_scope('accuracy'):
        best = tf.argmax(tf.reshape(y[:, 1:], [-1, common.PLATE_LEN, len(common.CHARS)]), 2)
        correct = tf.argmax(tf.reshape(y_[:, 1:], [-1, common.PLATE_LEN, len(common.CHARS)]), 2)
        is_correct = tf.equal(best, correct)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss', loss)

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    test_xs, test_ys = unzip(list(read_data(os.path.join("test","*.png")))[:batch_size])

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_writer = tf.summary.FileWriter('./summaries/train', sess.graph)
        test_writer = tf.summary.FileWriter('./summaries/test')

        sess.run(init)
        if initial_model is not None:
            if not os.path.exists(initial_model):
                print(initial_model, "does not exist")
                saver.restore(sess, initial_model)

        try:
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                if batch_idx % report_steps == 0:
                    r = sess.run([merged, best, correct,
                                tf.greater(y[:, 0], 0),
                                y_[:, 0],
                                digits_loss,
                                presence_loss,
                                loss],
                                feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0})

                    summary = r.pop(0)
                    test_writer.add_summary(summary, batch_idx)
                    print_report(batch_size, batch_idx, r)

                summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.40})
                train_writer.add_summary(summary, batch_idx)

                if max_steps != 0 and batch_idx >= max_steps:
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            try:
                print("Interrupted, saving...")
                fpath = os.path.join(MODELS_PATH, 'model_{0:%Y%m%dT%H%M%S}.ckpt'.format(datetime.datetime.now()))
                save_path = saver.save(sess, fpath)
                print("Models saved to", save_path)
            except KeyboardInterrupt:
                print("Saving interruped")
                exit()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        initial_model = sys.argv[1]
    else:
        initial_model = None

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    train(learn_rate=0.0001,
        report_steps=50,
        batch_size=100,
        initial_model=initial_model,
        max_steps=500000)