from PIL import Image, ImageFilter, ImageOps
import numpy as np
import re
import os

FILE_PATH = os.path.dirname(__file__)
GROUNDTRUTH_PATH = FILE_PATH + '/../data/groundtruth'
RELABELED_PATH = FILE_PATH + '/../data/relabeled'
TRAIN_IMAGES_PATH = FILE_PATH + "/../data/images"
TRAIN_FILTERED_IMAGES_PATH = FILE_PATH + "/../data/filteredImages/train"

all_filters = {"blur": ImageFilter.BLUR,
               "contour": ImageFilter.CONTOUR,
               "detail": ImageFilter.DETAIL,
               "edge_enhance": ImageFilter.EDGE_ENHANCE,
               "edge_enhance_more": ImageFilter.EDGE_ENHANCE_MORE,
               "emboss": ImageFilter.EMBOSS,
               "find_edges": ImageFilter.FIND_EDGES,
               "sharpen": ImageFilter.SHARPEN,
               "smooth": ImageFilter.SMOOTH,
               "smooth_more": ImageFilter.SMOOTH_MORE,
               "maxfilter": ImageFilter.MaxFilter(size=5),
               "medianfilter": ImageFilter.MedianFilter(size=5),
               "minfilter": ImageFilter.MinFilter(size=5)
               }

FILTERS = {"find_edges": ImageFilter.FIND_EDGES,
           "sharpen": ImageFilter.SHARPEN,
           "smooth": ImageFilter.SMOOTH,
           "medianfilter": ImageFilter.MedianFilter(size=3)
           }


def add_noise(image, type="s&p"):
    """Function to add noise to image"""
    w, h = image.size

    if type == "s&p":
        salt = 200
        noise = np.random.randint(salt+1, size=(h, w))

        idx_salt = noise == salt
        idx_pepper = noise == 0

        np_img = np.array(image)
        np_img[idx_salt, :] = 255
        np_img[idx_pepper, :] = 0

        return Image.fromarray(np.uint8(np_img))

    if type == "gauss":
        c = len(image.getbands())
        mean = 0
        std = 4
        noise = np.random.normal(mean, std, (h, w, c))
        return Image.fromarray(np.uint8(np.array(image) + noise))

    else:
        return images


def get_border(border, length, image):
    size = image.size

    if border == "left":
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
    widths, heights = zip(*(i.size for i in images))
    offset = 0
    rgb = images
    if axis == 1:
        max_width = max(widths)
        total_height = sum(heights)
        new_im = Image.new(images[0].mode, (max_width, total_height))

        for im in images:
            new_im.paste(im, (0, offset))
            offset += im.size[1]
    else:
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new(images[0].mode, (total_width, max_height))
        for im in images:
            new_im.paste(im, (offset, 0))
            offset += im.size[0]

    return new_im


def mirror_extend(num_added_pixels, image):
    top = get_border("top", num_added_pixels, image)
    bottom = get_border("bottom", num_added_pixels, image)

    tmp = concat_images([ImageOps.flip(top), image, ImageOps.flip(bottom)], axis=1)

    left = get_border("left", num_added_pixels, tmp)
    right = get_border("right", num_added_pixels, tmp)

    return concat_images([ImageOps.mirror(left), tmp, ImageOps.mirror(right)], axis=0)


def apply_filter(filter, image):
    size = image.size
    if isinstance(filter, ImageFilter.RankFilter):
        kernel_size = filter.size
    else:
        kernel_size = filter.filterargs[0][0]
    offset = int(kernel_size/2)
    extended_img = mirror_extend(offset, image)

    filter_extended_img = extended_img.filter(filter)
    filter_img = filter_extended_img.crop((offset, offset, offset+size[0], offset+size[1]))

    return filter_img


def apply_set_of_filters(filters, image):
    return [apply_filter(filt, image) for filt_name, filt in filters.items()]


def rotate_with_extension(image, alpha):

    # determine if alpha is larger than 90 degrees and rotate accordingly
    quarter = int(alpha / 90)
    image = image.rotate(quarter * 90)
    alpha = alpha - (quarter * 90)

    # size of image
    size = image.size[0]
    # get radians
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
    # desired height
    side = rotate.size[0]
    h = (side - size)/2
    # Crop the image
    return rotate.crop((h, h, h + size, h+size))


def generate_filtered_images():

    images = os.listdir(TRAIN_IMAGES_PATH)
    for file in images:
        pattern = re.compile('(.*)\.png')
        if pattern.match(file):
            img = Image.open(TRAIN_IMAGES_PATH + "/" + file)
            name = re.search(pattern, file).group(1)
            filtered_imgs = apply_set_of_filters(FILTERS, img)
            for filtered_img, filt in zip(filtered_imgs, FILTERS):
                file_name = name + "_" + filt + ".png"
                filtered_img.save(TRAIN_FILTERED_IMAGES_PATH + "/" + file_name, "PNG")


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


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


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


def generate_rand_image(image, groundtruth, noise=True, flip=True):

    size = image.size[0]

    rand_rotate = np.random.randint(360)

    image = rotate_with_extension(image, rand_rotate)
    groundtruth = rotate_with_extension(groundtruth, rand_rotate)

    if noise:
        noises = ["s&p", "gauss"]
        num_noises = len(noises)
        noise_rand = np.random.randint(num_noises + 1)
        if noise_rand < num_noises:
            image = add_noise(image, type=noises[noise_rand])

    if flip:
        rand_flip = np.random.randint(3)
        if rand_flip == 1:
            image = ImageOps.flip(image)
            groundtruth = ImageOps.flip(groundtruth)
        if rand_flip == 2:
            image = ImageOps.mirror(image)
            groundtruth = ImageOps.mirror(groundtruth)

    return (image, groundtruth)


def pillow2numpy(img):
    return np.array(img)


def numpy2pillow(np_img):
    tmp = img_float_to_uint8(np_img)
    return Image.fromarray(tmp)


def load_image_numpy(infilename):
    data = Image.open(infilename)
    return pillow2numpy(data)
