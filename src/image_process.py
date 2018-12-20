from PIL import Image, ImageFilter, ImageOps
import numpy as np
import re
import os
from paths_to_data import *

"""
File containing all functions relative to image processing
"""


def add_noise(image, type="s&p"):
    """
    Function to add noise (gaussian or salt and pepper) to image
    """
    # Get the width and height of the image
    w, h = image.size

    # Add salt and pepper noise
    if type == "s&p":
        # Choose a random amount of noise (lower number = more noise)
        salt = np.random.randint(100, 400)
        # Generate an array to determine location of noise
        noise = np.random.randint(salt+1, size=(h, w))

        # Find the index of the salt and pepper (respectively location with max/min random value)
        idx_salt = noise == salt
        idx_pepper = noise == 0

        # Create a numpy array from the initial image and add the salt and pepper
        np_img = np.array(image)
        np_img[idx_salt, :] = 255
        np_img[idx_pepper, :] = 0

        return Image.fromarray(np.uint8(np_img))

    # Add gaussian noise to image
    if type == "gauss":
        # Get the number of channels
        c = len(image.getbands())

        # Get a random value for the mean and the standard deviation of the noise
        mean = np.random.randint(-4, 5)
        std = np.random.randint(5)

        # Generate the noise array
        noise = np.random.normal(mean, std, (h, w, c))

        # Add noise to the image
        return Image.fromarray(np.uint8(np.array(image) + noise))

    else:
        # If the name of the given noise is not correct
        return image


def get_border(border, length, image):
    """
    Function to obtain a border of an image. Captures the border up to a certain length
    """
    # Size of the image
    size = image.size

    # Depending on the border to capture determine the indices of box to crop and crop it
    # crop(left,upper,right,lower)
    if border == "left":
        # Make sure we don't go more then the size of the image
        length = min(length, size[1])
        return image.crop((0, 0, length, size[1]))
    elif border == "right":
        length = min(length, size[1])
        return image.crop((size[0]-length, 0, size[0], size[1]))
    elif border == "top":
        length = min(length, size[0])
        return image.crop((0, 0, size[0], length))
    elif border == "bottom":
        length = min(length, size[0])
        return image.crop((0, size[1]-length, size[0], size[1]))
    raise NameError(border + ' is not a valid border name must be top,bottom,left or right')


def concat_images(images, axis=0):
    """
    Function to concatenate a list of images along a certain axis
    """
    # Get the width and the heights
    widths, heights = zip(*(i.size for i in images))

    # Initalize an offset to append the next image to the end of the previous
    offset = 0

    # Concatenate along the lines
    if axis == 1:
        # Get the width of the final image and the height
        max_width = max(widths)
        total_height = sum(heights)
        # Initalize the final image with the first subimage
        new_im = Image.new(images[0].mode, (max_width, total_height))

        # Append all consecutive images
        for im in images:
            new_im.paste(im, (0, offset))
            offset += im.size[1]

    # Concatenate along the columns
    else:
        # Get the width and the height of the final image
        total_width = sum(widths)
        max_height = max(heights)
        # Initalize the final image with the first subimage
        new_im = Image.new(images[0].mode, (total_width, max_height))

        # Append all consecutive images
        for im in images:
            new_im.paste(im, (offset, 0))
            offset += im.size[0]

    return new_im


def mirror_extend(num_added_pixels, image):
    """
    Function to extend the size of an image using a mirroring pattern for padding
    the added pixels
    """

    # Get the top and the bottom of the initial image
    top = get_border("top", num_added_pixels, image)
    bottom = get_border("bottom", num_added_pixels, image)

    # Concatenate the image with the flip top and bottom parts
    tmp = concat_images([ImageOps.flip(top), image, ImageOps.flip(bottom)], axis=1)

    # Get the left and right part of the previously extended image
    left = get_border("left", num_added_pixels, tmp)
    right = get_border("right", num_added_pixels, tmp)

    # Concatenante the extended image with the mirrored borders and returns the final images
    return concat_images([ImageOps.mirror(left), tmp, ImageOps.mirror(right)], axis=0)


def rotate_with_extension(image, alpha):
    """
        Rotates an image while extending the border through padding mirroring pattern
        to avoid padding with black pixels
    """
    # Determine if alpha is larger than 90 degrees and rotate accordingly
    # Number of 90 degree turns
    quarter = int(alpha / 90)
    # turn the image 90 deg quarter times
    image = image.rotate(quarter * 90)
    # Get the angle of the rest of the rotation
    alpha = alpha % 90

    # Size of image
    size = image.size[0]
    # Get radians
    alpha_rad = alpha/180 * np.pi

    # Compute the size of the extended image needed to keep an image of the
    # right size after cropping
    cos = np.cos(alpha_rad)
    sin = np.sin(alpha_rad)
    L = int(size * (sin + cos))

    # Extend the image then rotate it
    extend = mirror_extend(int((L-size)/2), image)
    rotate = extend.rotate(alpha, expand=1)
    # Get the current side length of the rotated image and compute the
    # difference with the original image to obtain the number of pixels to remove
    # from each border
    side = rotate.size[0]
    pixel_to_remove = (side - size)/2
    # Crop the image
    return rotate.crop((pixel_to_remove, pixel_to_remove, pixel_to_remove + size, pixel_to_remove+size))


def shift_with_extension(image, shift):
    """ Shifts the image with a mirroring/reflective padding"""

    # Get the size of the image
    image_size = image.size[0]

    # Get the x and y shifts
    x, y = shift

    # Get the maximum value to shift by
    max_shift = max(abs(x), abs(y))

    # Perform the mirror extension of the original image by the correct amount
    extend = mirror_extend(max_shift, image)

    # Determine the left right upper and lower indices of the box to crop
    left = max_shift + y
    right = left + image_size
    upper = max_shift + x
    lower = upper + image_size

    # Crop the image
    return extend.crop((left, upper, right, lower))


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


def generate_rand_image(image, groundtruth, noise=True, flip=True):
    """
    Given an image and the groudtruth mask, generates a augmented version of
    the image and changes the mask accordingly
    """
    # Get the size of the image
    x_size, y_size = image.size

    def rotate_augmentation():
        """Generate a function to perform a random rotation of an image
        using mirroring for padding"""
        rand_rotate = np.random.randint(180)
        return lambda image: rotate_with_extension(image, rand_rotate)

    def shift_augmentation():
        """Generates a function to perform a random shift of the image using mirroring
        for padding"""
        shift = np.random.randint(-200, 201, size=2)
        return lambda image: shift_with_extension(image, shift)

    def zoom_augmentation():
        """Generates a function that performs a random zoom on the image"""
        # Get the width and the height of the zoomed version
        x_len, y_len = np.random.randint(250, 350, size=2)
        # Get left upper ,right and lower bound of the pixels in the original image
        left = np.random.randint(x_size-x_len)
        upper = np.random.randint(y_size-y_len)
        right, lower = left + x_len, upper+y_len
        # Crops the box and resizes it to the original image size
        box = (left, upper, right, lower)
        return lambda image: image.transform(image.size, Image.EXTENT, box)

    def flip_augmentation():
        """Generates a function to flip the image"""
        return lambda image: ImageOps.flip(image)

    def mirror_augmentation():
        """Generates a function to mirror an image"""
        return lambda image: ImageOps.mirror(image)

    # All possible augmentations
    augmentations = [rotate_augmentation(), shift_augmentation(), zoom_augmentation(),
                     flip_augmentation(), mirror_augmentation()]

    # Loop through all augmentations and apply each one with a probability of 0.5
    for augmentation in augmentations:
        if np.random.randint(2) == 1:
            image = augmentation(image)
            groundtruth = augmentation(groundtruth)

    # Add salt or pepper noise each one with a probability of 0.33
    if noise:
        noises = ["s&p", "gauss"]
        num_noises = len(noises)
        # Choose noise to apply
        noise_rand = np.random.randint(num_noises + 1)
        # apply the noise only to the image and not the groundtruth
        if noise_rand < num_noises:
            image = add_noise(image, type=noises[noise_rand])

    return (image, groundtruth)


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
