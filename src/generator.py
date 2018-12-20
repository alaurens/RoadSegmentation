import numpy as np
import math
import re
import os
import threading
import random
from paths_to_data import *
from image_process import *
from data_process import *

"""
File that provides the image generators for traing, validation and testing
"""


class threadsafe_iter:

    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    This function and the following were taken from:
        https://anandology.com/blog/using-iterators-and-generators/
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        # locks the thread so the generator will not crash
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def train_generator(patch_dim, relabel_mask=True):
    """
    The following generator loads as many images as needed for the training
    of the network. Every image is modified at random for data augmentation
    """

    # Get all the files in the in the train data folder
    files = os.listdir(TRAIN_IMAGES_PATH)
    # Get pattern to check for valid image
    pattern = re.compile('satImage_\d+\.png')

    # Generate images as long as needed
    while True:

        # Choose one file
        file = random.choice(files)

        # Check that it's an image otherwise go to the next file
        if not pattern.match(file):
            continue

        # load the image and the mask
        img = Image.open(TRAIN_IMAGES_PATH + "/" + file)
        mask = Image.open(GROUNDTRUTH_PATH + "/" + file)
        # If desired relabel mask to have only 0 and 255
        if relabel_mask:
            mask = relabel(mask)

        # Data augmentation applied to the image
        img, mask = generate_rand_image(img, mask, noise=True, flip=True)

        # Generate the numpy arrays from the images and scale the mask to 0 and 1
        # so the logistic regression function can work
        np_img = pillow2numpy(img)
        np_mask = pillow2numpy(mask)/255

        # If the patch size is not a multiple of the image extend it with reflection padding
        # and do the same for the mask
        np_img = resize_image(np_img, patch_dim)
        np_mask = resize_image(np_mask, patch_dim)

        # Separate the images into batches according to the patch size
        # every batch is composed of the patches of one image
        batch_img = get_patches(np_img, patch_dim)
        batch_mask = get_patches(np_mask, patch_dim)

        # Yield the batches
        yield batch_img, batch_mask


@threadsafe_generator
def validation_generator(patch_dim, relabel_mask=True):
    """
        This loads all the files from the validation data folder and feed them
        for the validation step of the network
    """

    # Get all the files in the in the train data folder
    files = os.listdir(VALIDATION_IMAGES_PATH)
    # Get pattern to check for valid image
    pattern = re.compile('satImage_\d+\.png')

    # Infinite loop through going each time through all the files
    while True:

        # Get file and check it is valid
        for file in files:
            if not pattern.match(file):
                continue

            # Load image and mask
            img = Image.open(VALIDATION_IMAGES_PATH + "/" + file)
            mask = Image.open(GROUNDTRUTH_PATH + "/" + file)

            # Relabel if desired the mask to 0 or 255
            if relabel_mask:
                mask = relabel(mask)

            # Generate the numpy arrays from the images and scale the mask to 0 and 1
            # so the logistic regression function can work
            np_img = pillow2numpy(img)
            np_mask = pillow2numpy(mask)/255

            # If the patch size is not a multiple of the image extend it with reflection padding
            # and do the same for the mask
            np_img = resize_image(np_img, patch_dim)
            np_mask = resize_image(np_mask, patch_dim)

            # Separate the images into batches according to the patch size
            # every batch is composed of the patches of one image
            batch_img = get_patches(np_img, patch_dim)
            batch_mask = get_patches(np_mask, patch_dim)

            yield batch_img, batch_mask


@threadsafe_generator
def test_generator(patch_dim, num_test):

    # Begining of name of images folder
    file_name = 'test_'

    # We want to be sure we load all images in order for reconstruction
    for i in range(1, num_test+1):

        # Load test image
        img = Image.open(TEST_FOLDER_PATH + "/" + file_name + str(i) + '/' + file_name + str(i) + '.png')

        # Generate the numpy array
        np_img = pillow2numpy(img)

        # If the patch size is not a multiple of the image extend it with reflection padding
        # and do the same for the mask
        np_img = resize_image(np_img, patch_dim)

        # Separate the image into batches according to the patch size
        # every batch is composed of the patches of one image
        batch_img = get_patches(np_img, patch_dim)

        yield batch_img
