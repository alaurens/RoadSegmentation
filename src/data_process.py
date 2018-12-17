import os
import numpy as np
import sys
import math
import matplotlib.image as mpimg
from paths_to_data import *
from image_process import *


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def load_image_train():

    files = os.listdir(TRAIN_IMAGES_PATH)
    n = len(files)
    #print("Loading " + str(n) + " images")
    imgs = np.asarray([load_image(TRAIN_IMAGES_PATH + files[i])
                       for i in range(n) if 'png' in files[i]])
    # print(np.shape(imgs))
    #print("Loading " + str(n) + " images")
    groundtruth_imgs = np.asarray([load_image(GROUNDTRUTH_PATH + files[i])
                                   for i in range(n) if 'png' in files[i]])
    a3d = np.expand_dims(groundtruth_imgs, axis=3)
    # print(np.shape(a3d))

    return imgs, a3d


def save_results(TEST_PREDICTED_IMAGES_PATH, files_to_save):

    if not os.path.exists(TEST_PREDICTED_IMAGES_PATH):
        os.mkdir(TEST_PREDICTED_IMAGES_PATH)
    for i, item in enumerate(files_to_save):

        print(item.shape)

        img = ip.numpy2pillow(item.squeeze())
        file_name = "{}.png".format(i)
        img.save(TEST_PREDICTED_IMAGES_PATH + "/" + file_name, "PNG")


def resize_test_image(test_image, patch_dim):

    if len(test_image.shape) == 2:
        test_image = np.expand_dims(test_image, axis=3)

    if np.size(test_image, 0) % patch_dim == 0:
        test_image = test_image
    else:
        test_image = numpy2pillow(test_image)
        add_pixel = patch_dim*(np.floor(np.size(test_image, 0) /
                                        patch_dim)+1) - np.size(test_image, 0)
        test_image = mirror_extend(add_pixel/2, test_image)
        test_image = pillow2numpy(test_image)

    return test_image


def reconstruct_images(patches, num_images):

    num_patches = patches.shape[0]
    patches_per_img = int(num_patches / num_images)
    patches_per_side = int(math.sqrt(patches_per_img))
    images = []

    width = patches.shape[2] * patches_per_side
    num_channels = patches.shape[3]

    for _ in range(num_images):
        b = np.empty((0, width, num_channels))
        for i in range(0, patches_per_img, patches_per_side):
            tmp = np.concatenate((patches[i:i+4]), axis=1)
            b = np.concatenate((b, tmp), axis=0)

        images.append(b)
    return images


def relabel(img):

    np_img = pillow2numpy(img)
    max = np.max(np_img)

    np_img[np_img <= (max*0.9)] = 0
    np_img[np_img > (max*0.9)] = 1

    return numpy2pillow(np_img)


def relabel_all_images():

    label_imgs = os.listdir(GROUNDTRUTH_PATH)

    if not os.path.exists(RELABELED_PATH):
        os.mkdir(RELABELED_PATH)

    for img_name in label_imgs:
        img = Image.open(GROUNDTRUTH_PATH + '/' + img_name)
        relabel(img).save(RELABELED_PATH + '/' + img_name, "PNG")


def get_patches(np_img, patch_dim):

    if len(np_img.shape) == 2:
        np_img = np.expand_dims(np_img, axis=3)

    size, _, num_channels = np_img.shape

    dim = (0, patch_dim, patch_dim, num_channels)
    patches = []
    for i in range(0, size, patch_dim):
        for j in range(0, size, patch_dim):
            patch = np_img[i:i+patch_dim, j:j+patch_dim, :]
            patches.append(patch)

    patches = np.asarray(patches)

    return patches
