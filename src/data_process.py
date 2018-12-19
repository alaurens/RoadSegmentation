import os
import numpy as np
import sys
import math
import matplotlib.image as mpimg
from paths_to_data import *
from image_process import *
from paths_to_data import *
from mask_to_submission import *


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


def resize_image(np_image, patch_dim):

    if len(np_image.shape) == 2:
        np_image = np.expand_dims(np_image, axis=3)

    if np.size(np_image, 0) % patch_dim == 0:
        return np_image

    else:
        np_image = numpy2pillow(np_image)
        add_pixel = patch_dim*(np.floor(np.size(np_image, 0) /
                                        patch_dim)+1) - np.size(np_image, 0)
        np_image = mirror_extend(add_pixel/2, np_image)
        np_image = pillow2numpy(np_image)

        return np_image


def reconstruct_images(patches, num_images):

    num_patches = patches.shape[0]
    patches_per_img = int(num_patches / num_images)
    patches_per_side = int(math.sqrt(patches_per_img))
    images = []

    width = patches.shape[2] * patches_per_side
    num_channels = patches.shape[3]

    for i in range(num_images):
        b = np.empty((0, width, num_channels))
        for j in range(i*patches_per_img, (i+1)*patches_per_img, patches_per_side):
            tmp = np.concatenate((patches[j:j+patches_per_side]), axis=1)
            b = np.concatenate((b, tmp), axis=0)
        images.append(b.copy())
    return images


def crop_prediction(image, original_img_size):

    shape = image.size[0]
    border_i = (shape - original_img_size)/2
    border_f = border_i + original_img_size
    cropped_image = image.crop((border_i, border_i, border_f, border_f))

    return cropped_image


def relabel(img):

    np_img = pillow2numpy(img)
    max = np.max(np_img)
    threshold = 0.6
    np_img[np_img <= (max*threshold)] = 0
    np_img[np_img > (max*threshold)] = 1

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


def save_results(patches, num_images, original_img_size):

    if not os.path.exists(PREDICTED_IMAGES_PATH):
        os.mkdir(PREDICTED_IMAGES_PATH)

    image = reconstruct_images(patches, num_images)

    for i, item in enumerate(image):

        img = numpy2pillow(item.squeeze())
        pred = crop_prediction(img, original_img_size)

        file_name = "prediction{}.png".format(i+1)
        pred.save(PREDICTED_IMAGES_PATH + "/" + file_name, "PNG")


def create_submission(submission_filename):

    if not os.path.exists(SUBMISSION_PATH):
        os.mkdir(SUBMISSION_PATH)
    submission_filename = SUBMISSION_PATH + '/' + submission_filename
    files = os.listdir(PREDICTED_IMAGES_PATH)

    images_name = list(filter(lambda x: x.endswith('.png'), files))

    masks_to_submission(submission_filename, images_name)
