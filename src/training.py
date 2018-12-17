from imageProcess import *
from paths_to_data import *
import numpy as np


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

        img, mask = generate_rand_image(img, mask, noise=True, flip=True)

        np_img = pillow2numpy(img)
        np_mask = pillow2numpy(mask)

        batch_img = get_patches(np_img, patch_dim)
        batch_mask = get_patches(np_mask, patch_dim)

        yield batch_img, batch_mask
