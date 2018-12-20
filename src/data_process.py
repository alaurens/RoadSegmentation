import os
import numpy as np
import sys
import math
import matplotlib.image as mpimg
from paths_to_data import *
from image_process import *
from paths_to_data import *
from mask_to_submission import *

"""
This file contains all function relative to the data processing
"""


def resize_image(np_image, patch_dim):
    """
    Resizes an image such that the final image size is a multiple of the patch size
    """

    # Adds another dimension for images with one channel
    if len(np_image.shape) == 2:
        np_image = np.expand_dims(np_image, axis=3)

    # If the image size is already a multiple of the patch size the image size is not changed
    if np.size(np_image, 0) % patch_dim == 0:
        return np_image

    else:
        np_image = numpy2pillow(np_image)

        # Computes the number of patches in the extended image
        nb_patch = int(np.size(np_image, 0) / patch_dim) + 1

        # Number of pixels to add to get the correct size
        add_pixel = patch_dim * nb_patch - np.size(np_image, 0)

        # Adds the the pixels using a mirror on the borders
        np_image = mirror_extend(add_pixel/2, np_image)

        # COnvert to numpy
        np_image = pillow2numpy(np_image)

        return np_image


def reconstruct_images(patches, num_images):
    """
    Reconstructs images from patches
    """

    # Computes the dimension of the original images
    num_patches = patches.shape[0]
    # Number of patches per image and patches per image side
    patches_per_img = int(num_patches / num_images)
    patches_per_side = int(math.sqrt(patches_per_img))
    images = []

    # Get the width of one image and the number of channels
    width = patches.shape[2] * patches_per_side
    num_channels = patches.shape[3]

    # Patches are concatenated line by line and then all lines are concatenated
    for i in range(num_images):
        # Initialize a container for the images
        b = np.empty((0, width, num_channels))
        # Loop through images
        for j in range(i*patches_per_img, (i+1)*patches_per_img, patches_per_side):
            # Concatenate line by line
            tmp = np.concatenate((patches[j:j+patches_per_side]), axis=1)
            b = np.concatenate((b, tmp), axis=0)
        # Add the reconstructed image to the list of all images
        images.append(b.copy())
    return images


def crop_prediction(image, original_img_size):
    """
    Crops a prediction to obtain the correct size prediction
    """

    # Computes the intial and final position of the box to crop
    shape = image.size[0]
    border_i = (shape - original_img_size)/2
    border_f = border_i + original_img_size

    # Crops the image according to the previsous index
    cropped_image = image.crop((border_i, border_i, border_f, border_f))

    return cropped_image


def relabel(img):
    """
    Transforms a mask into black-white images by attributing to each pixel label
    0 and 1 according to a specific threshold
    """

    # Converts the image in a numpy array and take the maximum value of this array
    np_img = pillow2numpy(img)
    max = np.max(np_img)

    # Relabels pixels to 1 or 0 according to a threshold value
    threshold = 0.3
    np_img[np_img <= (max*threshold)] = 0
    np_img[np_img > (max*threshold)] = 1

    return numpy2pillow(np_img)


def get_patches(np_img, patch_dim):
    """
    Cuts the image in patches according the the patch size
    """

    # Adds an other dimension for images with one channel
    if len(np_img.shape) == 2:
        np_img = np.expand_dims(np_img, axis=3)

    # Get the width/height of on array and the number of channels
    size, _, num_channels = np_img.shape
    # Dimension of the patches
    dim = (0, patch_dim, patch_dim, num_channels)
    patches = []

    # Cuts the image into patches of size patch_dim
    for i in range(0, size, patch_dim):
        for j in range(0, size, patch_dim):
            patch = np_img[i:i+patch_dim, j:j+patch_dim, :]
            patches.append(patch)

    # Stores all patches in an array
    patches = np.asarray(patches)

    return patches


def save_results(patches, num_images, original_img_size):
    """
    Takes the predicted patches, reconstructs, the full predictions and saves them
    """

    # Creates a folder to store predicted images if it dosen't exist
    if not os.path.exists(PREDICTED_IMAGES_PATH):
        os.mkdir(PREDICTED_IMAGES_PATH)

    # Reconstructs prediction images from patches
    image = reconstruct_images(patches, num_images)

    for i, item in enumerate(image):

        # Crops the enlarged predicitons to the correct size
        img = numpy2pillow(item.squeeze())
        pred = crop_prediction(img, original_img_size)

        # Apply dilation and then erosion to fill holes in prediction
        for _ in range(4):
            pred = pred.filter(ImageFilter.MaxFilter(5))
        for _ in range(4):
            pred = pred.filter(ImageFilter.MinFilter(5))

        # Save prediction images in the folder in PNG format
        file_name = "prediction{}.png".format(i+1)
        pred.save(PREDICTED_IMAGES_PATH + "/" + file_name, "PNG")


def create_submission(submission_filename):
    """
    Converts the prediction images into csv files containing only 0 and 1 values for submission
    """

    # Creates a folder to stock csv submission in case the folder already doesn't exist
    if not os.path.exists(SUBMISSION_PATH):
        os.mkdir(SUBMISSION_PATH)

    # Creates a list of all files constituting the folder of prediction images
    submission_filename = SUBMISSION_PATH + '/' + submission_filename
    files = os.listdir(PREDICTED_IMAGES_PATH)

    # Applies a filter to only select .png files
    images_name = list(filter(lambda x: x.endswith('.png'), files))

    # Converts the prediction images into csv files
    masks_to_submission(submission_filename, images_name)


def load_image(infilename):
    """
    Loads an image from a specific file to the numpy format
    """
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    """
        Transforms a given image from floating point numbers (from 0 to 1)
        to ints (from 0 to 255)
    """
    # Remove the minimum of the image
    rimg = img - np.min(img)
    # Divide by the max to garanty a spread from 0 to 1 and map to ints from 0 to 255
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def pillow2numpy(img):
    """Function to convert an image from pillow to numpy"""
    return np.array(img)


def numpy2pillow(np_img):
    """Function to convert an image from numpy to pillow"""
    # Convert the numpy matrix to a 0 to 255 matrix
    tmp = img_float_to_uint8(np_img)
    # Remove the channels index if it is one (black and white images)
    tmp = tmp.squeeze()
    # Transform to pillow
    return Image.fromarray(tmp)
