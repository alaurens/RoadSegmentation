import os
import numpy as np
import generator as gr
import sys
import math
import matplotlib.image as mpimg


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def load_image_train():

    image_folder = "images/"
    files = os.listdir(image_folder)
    n = len(files)
    print("Loading " + str(n) + " images")
    imgs = np.asarray([load_image(image_folder + files[i]) for i in range(n) if 'png' in files[i]])
    print(np.shape(imgs))
    groundtruth_folder = 'groundtruth/'
    print("Loading " + str(n) + " images")
    groundtruth_imgs = np.asarray([load_image(groundtruth_folder + files[i])
                                   for i in range(n) if 'png' in files[i]])
    a3d = np.expand_dims(groundtruth_imgs, axis=3)
    print(np.shape(a3d))

    return imgs, a3d


def load_image_test(number_img):

    test_folder = 'test_set_images/test_'
    for i in range(1, number_img+1):
        direc = test_folder + str(i) + "/test_" + str(i) + ".png"
        test_imgs = np.asarray([load_image(direc)])
        test_imgs = np.squeeze(test_imgs, axis=0)
        yield gr.patch_generator(test_imgs, 608)


def save_results(TEST_PREDICTED_IMAGES_PATH, files_to_save):

    if not os.path.exists(TEST_PREDICTED_IMAGES_PATH):
        os.mkdir(TEST_PREDICTED_IMAGES_PATH)
    for i, item in enumerate(files_to_save):

        print(item.shape)

        img = ip.numpy2pillow(item.squeeze())
        file_name = "{}.png".format(i)
        img.save(TEST_PREDICTED_IMAGES_PATH + "/" + file_name, "PNG")
