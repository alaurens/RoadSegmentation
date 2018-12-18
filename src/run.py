from model import *
from paths_to_data import *
from generator import *
from data_process import *
import h5py

num_test_imgs = 50
original_image_size = 608
weights = WEIGHTS_PATH + '/' + '2weights48.hdf5'
patch_dim = 400
channels = 3

layers = [64, 128, 256, 512]

test_data_gen = test_generator(patch_dim, num_test_imgs)

model = unet2(input_size=(patch_dim, patch_dim, channels), layers=layers, pretrained_weights=weights)

prediction = model.predict_generator(test_data_gen, num_test_imgs, verbose=1)

save_results(prediction, num_test_imgs, original_image_size)

create_submission('test.csv')
