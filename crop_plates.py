import os
import sys
import math
from argparse import ArgumentParser

import cv2
import numpy as np
import yaml

parser = ArgumentParser(description='OpenALPR License Plate Cropper')

parser.add_argument( "--input_dir", dest="input_dir", action="store", type=str, required=True,
                  help="Directory containing plate images and yaml metadata" )

parser.add_argument( "--out_dir", dest="out_dir", action="store", type=str, required=True,
                  help="Directory to output cropped plates" )

options = parser.parse_args()


if not os.path.isdir(options.input_dir):
    print("input_dir (%s) doesn't exist")
    sys.exit(1)


if not os.path.isdir(options.out_dir):
    os.makedirs(options.out_dir)

CROP_WIDTH=128.0
CROP_HEIGHT=64.0

def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i+n]
        if len(val) == n:
            yield tuple(val)


def get_box(points):
    (x1, y1, x2, y2, x3, y3, x4, y4) = points

    width = math.hypot(x2 - x1, y2 - y1)
    height = math.hypot(x3 - x2, y3 - y2)

    if height > width:
        width *= 2.5
        height = int(round((CROP_WIDTH / CROP_HEIGHT) * width))
    else:
        height *= 2.5
        width = int(round((CROP_WIDTH / CROP_HEIGHT) * height))


    points = list(group(points, 2))
    moment = cv2.moments(np.asarray(points))
    centerx = int(round(moment['m10']/moment['m00']))
    centery = int(round(moment['m01']/moment['m00']))
    top_left_x = int(round(centerx - (width / 2)))
    top_left_y = int(round(centery - (height / 2)))

    if top_left_x < 0:
        top_left_x = 0
    if top_left_y < 0:
        top_left_y = 0

    return (top_left_x, top_left_y, int(round(width)), int(round(height)))


def crop_rect(img, x, y, width, height):
    crop_img =  img[y:y+height-1, x:x+width-1]
    crop_img = cv2.resize(crop_img, dsize=(int(CROP_WIDTH), int(CROP_HEIGHT)), interpolation=cv2.INTER_CUBIC)
    return crop_img


count = 1
yaml_files = []
for in_file in os.listdir(options.input_dir):
    if in_file.endswith('.yaml') or in_file.endswith('.yml'):
        yaml_files.append(in_file)


yaml_files.sort()

for yaml_file in yaml_files:
    print("Processing: " + yaml_file + " (" + str(count) + "/" + str(len(yaml_files)) + ")")
    count += 1

    yaml_path = os.path.join(options.input_dir, yaml_file)
    yaml_without_ext = os.path.splitext(yaml_path)[0]
    with open(yaml_path, 'r') as yf:
        yaml_obj = yaml.load(yf)

    image = yaml_obj['image_file']

    # Skip missing images
    full_image_path = os.path.join(options.input_dir, image)
    if not os.path.isfile(full_image_path):
        print("Could not find image file %s, skipping" % (full_image_path))
        continue


    plate_corners = yaml_obj['plate_corners_gt']

    cc = plate_corners.strip().split()
    #points = [(230, 230), (250, 220), (250, 245), (235, 241)]
    for i in range(0, len(cc)):
        cc[i] = int(cc[i])

    x, y, width, height = get_box(cc)

    img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE) / 255.

    crop = crop_rect(img, x, y , width, height)

    visible = "0" if yaml_obj['plate_number_gt'] == 0 else "1"
    if visible == "0":
        plate = "AAA000"
    else:
        plate = yaml_obj['plate_number_gt']

    out_crop_path = os.path.join(options.out_dir, "00000000_{}_{}.png".format(plate, visible))
    cv2.imwrite(out_crop_path, crop * 255.)
