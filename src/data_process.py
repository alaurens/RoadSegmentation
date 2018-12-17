import os
import numpy as np
import generator as gr
import sys
import math
import matplotlib.image as mpimg
from paths_to_data import *


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
