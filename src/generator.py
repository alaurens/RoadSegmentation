import numpy as np
import math
import re
from paths_to_data import *
from imageProcess import *


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


def load_image_test(number_img):

    test_folder = 'test_set_images/test_'
    for i in range(1, number_img+1):
        direc = TEST_FOLDER_PATH + '/test_' + str(i) + "/test_" + str(i) + ".png"
        test_imgs = np.asarray([load_image(direc)])
        test_imgs = np.squeeze(test_imgs, axis=0)
        yield patch_generator(test_imgs, 608)


def get_patch(img, patch_dim):
    num_channels = img.shape[2]
    size = np.size(img, 0)
    dim = (0, patch_dim, patch_dim, num_channels)
    patches = []
    for i in range(0, size, patch_dim):
        for j in range(0, size, patch_dim):
            patch = img[i:i+patch_dim, j:j+patch_dim, :]

            patches.append(patch)

    patches = np.asarray(patches)
    return patches


def patch_generator(test_image, patch_dim):

    if np.size(test_image, 0) % patch_dim == 0:
        test_image = test_image
    else:
        test_image = numpy2pillow(test_image)
        add_pixel = patch_dim*(np.floor(np.size(test_image, 0) /
                                        patch_dim)+1) - np.size(test_image, 0)
        test_image = mirror_extend(add_pixel/2, test_image)
        test_image = pillow2numpy(test_image)

    vec = get_patch(test_image, patch_dim)

    return vec


def prediction_generator(prediction_patch):

    number_patch = (prediction_patch.shape[0])
    len_patch = int(math.sqrt(number_patch))
    img = []

    for i in range(0, prediction_patch.shape[0], len_patch):
        for j in range(0, prediction_patch.shape[0], len_patch):
            img[i, j] = prediction_patch[j+i*len_patch]

    img = numpy2pillow(img)

    return img
