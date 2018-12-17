import numpy as np
import math
import re
import os
from paths_to_data import *
from image_process import *
from data_process import *


def train_generator(patch_dim):

    images_name = os.listdir(TRAIN_IMAGES_PATH)
    pattern = re.compile('(.*)\.png')

    while True:

        idx = np.random.randint(len(images_name))
        file = images_name[idx]

        if not pattern.match(file):
            continue

        img = Image.open(TRAIN_IMAGES_PATH + "/" + file)
        mask = Image.open(GROUNDTRUTH_PATH + "/" + file)
        mask = relabel(mask)

        img, mask = generate_rand_image(img, mask, noise=True, flip=True)

        np_img = pillow2numpy(img)
        np_mask = pillow2numpy(mask)/255

        batch_img = get_patches(np_img, patch_dim)
        batch_mask = get_patches(np_mask, patch_dim)

        yield batch_img, batch_mask


def validation_generator(patch_dim):

    images_name = os.listdir(VALIDATION_IMAGES_PATH)
    pattern = re.compile('(.*)\.png')
    while True:
        for file in images_name:
            if not pattern.match(file):
                continue

            img = Image.open(VALIDATION_IMAGES_PATH + "/" + file)
            mask = Image.open(GROUNDTRUTH_PATH + "/" + file)
            mask = relabel(mask)

            np_img = pillow2numpy(img)
            np_mask = pillow2numpy(mask)/255

            batch_img = get_patches(np_img, patch_dim)
            batch_mask = get_patches(np_mask, patch_dim)

            yield batch_img, batch_mask


def test_generator(patch_dim):

    images_name = os.listdir(TEST_FOLDER_PATH)
    pattern = re.compile('test_[0-9]+')
    for file in images_name:
        if not pattern.match(file):
            continue

        img = Image.open(TEST_FOLDER_PATH + "/" + file + '/' + file + '.png')

        np_img = pillow2numpy(img)

        np_img = resize_test_image(np_img, patch_dim)

        batch_img = get_patches(np_img, patch_dim)

        yield batch_img
