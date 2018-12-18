from model import *
from paths_to_data import *
from generator import *
from data_process import *
import h5py

num_test_imgs = 3
original_image_size = 608

patch_dim=400
channels=3

layers=[16, 32, 64, 128]

test_data_gen = test_generator(patch_dim)

model = unet(input_size=(patch_dim, patch_dim, channels), layers=layers, pretrained_weights='weights.hdf5')

prediction = model.predict_generator(test_data_gen,num_test_imgs,verbose = 1)

save_results(prediction,num_test_imgs,original_image_size)

create_submission('test.csv')
