import numpy as np
import math
import re
import os
import threading
from paths_to_data import *
from image_process import *
from data_process import *


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def train_generator(patch_dim, relabel_mask=False):

    images_name = os.listdir(TRAIN_IMAGES_PATH)
    pattern = re.compile('(.*)\.png')

    while True:

        idx = np.random.randint(len(images_name))
        file = images_name[idx]

        if not pattern.match(file):
            continue

        img = Image.open(TRAIN_IMAGES_PATH + "/" + file)
        mask = Image.open(GROUNDTRUTH_PATH + "/" + file)
        if relabel_mask:
            mask = relabel(mask)

        img, mask = generate_rand_image(img, mask, noise=True, flip=True)

        np_img = pillow2numpy(img)
        np_mask = pillow2numpy(mask)/255

        np_img = resize_image(np_img, patch_dim)
        np_mask = resize_image(np_mask, patch_dim)

        batch_img = get_patches(np_img, patch_dim)
        batch_mask = get_patches(np_mask, patch_dim)

        yield batch_img, batch_mask


@threadsafe_generator
def validation_generator(patch_dim, relabel_mask=False):

    images_name = os.listdir(VALIDATION_IMAGES_PATH)
    pattern = re.compile('(.*)\.png')
    while True:
        for file in images_name:
            if not pattern.match(file):
                continue

            img = Image.open(VALIDATION_IMAGES_PATH + "/" + file)
            mask = Image.open(GROUNDTRUTH_PATH + "/" + file)
            if relabel_mask:
                mask = relabel(mask)

            np_img = pillow2numpy(img)
            np_mask = pillow2numpy(mask)/255

            np_img = resize_image(np_img, patch_dim)
            np_mask = resize_image(np_mask, patch_dim)

            batch_img = get_patches(np_img, patch_dim)
            batch_mask = get_patches(np_mask, patch_dim)

            yield batch_img, batch_mask


@threadsafe_generator
def test_generator(patch_dim, num_test):

    file_name = 'test_'
    for i in range(1, num_test+1):

        img = Image.open(TEST_FOLDER_PATH + "/" + file_name + str(i) + '/' + file_name + str(i) + '.png')

        np_img = pillow2numpy(img)

        np_img = resize_image(np_img, patch_dim)

        batch_img = get_patches(np_img, patch_dim)

        yield batch_img
