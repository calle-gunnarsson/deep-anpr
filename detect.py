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
Routines to detect number plates.

Use `detect` to detect all bounding boxes, and use `post_process` on the output
of `detect` to filter using non-maximum suppression.

"""


__all__ = (
    'detect',
    'post_process',
)


import collections
import math
import sys
import os

import cv2
import numpy
import tensorflow as tf

import common
import model

import time

OUTPUT_DIR="results"

def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0] / ratio, im.shape[1] / ratio)

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(im, (shape[1], shape[0]))


def detect(sess, im, model_checkpoint):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.

    """

    # Convert the image to various scales.
    scaled_ims = list(make_scaled_ims(im, common.IMAGE_SHAPE))
    y_vals = []
    for scaled_im in scaled_ims:
        feed_dict = {x: numpy.stack([scaled_im])}
        y_vals.append(sess.run(y, feed_dict=feed_dict))

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                                       -math.log(1./0.99 - 1)):
            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    common.PLATE_LEN, len(common.CHARS)))
            letter_probs = common.softmax(letter_probs)

            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = numpy.array(common.IMAGE_SHAPE) * img_scale

            present_prob = common.sigmoid(
                               y_val[0, window_coords[0], window_coords[1], 0])

            yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs


def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])


def _group_overlapping_rectangles(matches):
    matches = list(matches)
    num_groups = 0
    match_to_group = {}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_group[idx1] = match_to_group[idx2]
                break
        else:
            match_to_group[idx1] = num_groups
            num_groups += 1

    groups = collections.defaultdict(list)
    for idx, group in match_to_group.items():
        groups[group].append(matches[idx])

    return groups


def post_process(matches):
    """
    Take an iterable of matches as returned by `detect` and merge duplicates.

    Merging consists of two steps:
      - Finding sets of overlapping rectangles.
      - Finding the intersection of those sets, along with the code
        corresponding with the rectangle with the highest presence parameter.

    """
    groups = _group_overlapping_rectangles(matches)

    for group_matches in groups.values():
        mins = numpy.stack(numpy.array(m[0]) for m in group_matches)
        maxs = numpy.stack(numpy.array(m[1]) for m in group_matches)
        present_probs = numpy.array([m[2] for m in group_matches])
        letter_probs = numpy.stack(m[3] for m in group_matches)

        yield (numpy.max(mins, axis=0).flatten(),
               numpy.min(maxs, axis=0).flatten(),
               numpy.max(present_probs),
               letter_probs[numpy.argmax(present_probs)])


def letter_probs_to_code(letter_probs):
    return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))


def scale_pt(factor, pt):
    return (int(pt[0] * factor), int(pt[1] * factor))

if __name__ == "__main__":
    model_checkpoint = sys.argv[1]

    startTime = time.time()

    imgs = sys.argv[2:]

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Load the model which detects number plates over a sliding window.
    x, y = model.get_detect_model()

    saver = tf.train.Saver()
    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        saver.restore(sess, model_checkpoint)
        print("Searching for plates in {} images.".format(len(imgs)))

        for img in imgs:
            fname = os.path.basename(img).split('.')[0]
            image = cv2.imread(img)

            im = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
            width=im.shape[1]
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

            i_height, i_width, _ = im.shape
            b_height, b_width = common.IMAGE_SHAPE

            bpt1 = (int(i_width/2 - b_width/2), int(i_height/2 - b_height/2))
            bpt2 = (int(i_width/2 + b_width/2), int(i_height/2 + b_height/2))

            factor = image.shape[1] / width

            bpt1 = scale_pt(factor, bpt1)
            bpt2 = scale_pt(factor, bpt2)

            color = (255.0, 0.0, 0.0)
            cv2.rectangle(image, bpt1, bpt2, color)

            print(img, end="", flush=True)
            plates = 0
            for pt1, pt2, present_prob, letter_probs in post_process(
                                                        detect(sess, im_gray, model_checkpoint)):
                plates += 1
                pt1 = tuple(reversed(list(map(int, pt1))))
                pt2 = tuple(reversed(list(map(int, pt2))))

                code = letter_probs_to_code(letter_probs)

                pt1 = scale_pt(factor, pt1)
                pt2 = scale_pt(factor, pt2)

                color = (0.0, 255.0, 0.0)
                cv2.rectangle(image, pt1, pt2, color)

                cv2.putText(image,
                            code,
                            pt1,
                            cv2.FONT_HERSHEY_PLAIN,
                            1.5,
                            (0, 0, 0),
                            thickness=5)

                cv2.putText(image,
                            code,
                            pt1,
                            cv2.FONT_HERSHEY_PLAIN,
                            1.5,
                            (255, 255, 255),
                            thickness=2)

            if plates > 0:
                print(" - found {} plates".format(plates), end="", flush=True)
                cv2.imwrite(os.path.join(OUTPUT_DIR, "{}.png".format(fname)), image)
            print("")
    resTime = time.time() - startTime
    print("Processed {} images in {} seconds (avg: {} sec/img)".format(len(imgs), resTime, int(resTime/len(imgs))))
