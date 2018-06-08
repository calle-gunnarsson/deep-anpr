#!/usr/bin/env python3
"""
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys
import json

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

PLATES_DIR = "./plates"

OUTPUT_DIR = "./test2"

FONT_HEIGHT = 64  # Pixel size to which the chars are resized

OUTPUT_SHAPE = common.IMAGE_SHAPE

debug_mode = False

def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def make_affine_transform(from_shape, to_shape,
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False


    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)

    #scale = random.uniform(min_scale, max_scale)
    if scale > max_scale or scale < min_scale:
        #print("Out of scale")
        out_of_bounds = True

    roll = random.uniform(-0.5, 0.5) * rotation_variation
    pitch = random.uniform(-0.4, 0.4) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.8) or numpy.any(trans > 0.8):
        #print("Out of trans", scale)
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def generate_code(form, country=None):
    return form.format(
        random.choice(common.LETTERS),
        random.choice(common.LETTERS),
        random.choice(common.LETTERS),
        random.choice(common.DIGITS),
        random.choice(common.DIGITS),
        random.choice(common.DIGITS))

def write_image(name, im):
    fname = os.path.join(OUTPUT_DIR, "{}.png".format(name))
    print(fname, im.shape)
    cv2.imwrite(fname, im)


def generate_plate(plate_data):
    form = random.choice(plate_data["formats"])
    code = generate_code(form["format"])

    plate = plate_data['frame'].copy()


    font = plate_data['font']

    img_pil = Image.fromarray(plate)
    draw = ImageDraw.Draw(img_pil)

    (x, y) = form['offset']
    draw.text((x, y),  code, 0, font=font)

    plate = numpy.array(img_pil)

    code = code.replace(" ", "")

    if debug_mode:
        write_image(code+"_orig", plate)

    r = (OUTPUT_SHAPE[1] / plate.shape[1])

    plate = cv2.resize(plate, (int(OUTPUT_SHAPE[1]), int(round(plate.shape[0]*r))))

    if debug_mode:
        write_image(code+"_small", plate)

    return cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), code


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        try:
            bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if (bg.shape[1] >= OUTPUT_SHAPE[1] and
                bg.shape[0] >= OUTPUT_SHAPE[0]):
                found = True
        except:
            continue

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])

    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(plate_data, num_bg_images):
    bg = generate_bg(num_bg_images)

    plate, code = generate_plate(plate_data)
    plate_mask = numpy.ones(plate.shape)

    M, out_of_bounds = make_affine_transform(
                            from_shape=(plate.shape[0], plate.shape[1]),
                            to_shape=bg.shape,
                            min_scale=0.5,
                            max_scale=1.0,
                            rotation_variation=1.0,
                            scale_variation=1.2,
                            translation_variation=1.2)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    if debug_mode:
        write_image("{}_bg".format(code), bg)
        write_image("{}_warped".format(code), plate)
        write_image("{}_mask".format(code), plate_mask)

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = (cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))) / 255.

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code, not out_of_bounds


def load_plates(folder_path):
    plates = {}
    for p in os.listdir(folder_path):
        if not os.path.isdir(os.path.join(folder_path, p)):
            continue

        all_exists = True
        files = [
            os.path.join(folder_path, p, "data.json"),
            os.path.join(folder_path, p, "font.ttf"),
            os.path.join(folder_path, p, "frame.png")
        ]
        for f in files:
            if not os.path.exists(f):
                print("{} is missing".format(f))
                all_exists = False

        if not all_exists:
            continue

        plate = {}
        with open(files[0]) as f:
            plate  = json.load(f)
            assert plate['formats']

        if "enabled" in plate and plate["enabled"] == 0:
            continue

        plate['font'] = ImageFont.truetype(files[1], plate['size'])
        plate['frame'] = cv2.imread(files[2])

        plates[p] = plate

    return plates


def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    plates = load_plates(PLATES_DIR)
    num_bg_images = len(os.listdir("bgs"))
    while True:
        yield generate_im(plates[random.choice(list(plates))], num_bg_images)


if __name__ == "__main__":
    debug_mode = os.getenv("DEBUG", "0") == "1"
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    im_gen = itertools.islice(generate_ims(), int(sys.argv[1]))
    for img_idx, (im, c, p) in enumerate(im_gen):
        name = "{:08d}_{}_{}".format(img_idx, c, "1" if p else "0")
        write_image(name, im*255)
