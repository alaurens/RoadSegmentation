from model import *
from paths_to_data import *
from generator import *
from data_process import *
import h5py

# Number of test images
num_test_imgs = 50
# Size of the test image
original_image_size = 608
# Name of the weights to load
weight_name = 'weights298'
# Get the full path to the weights
weights = WEIGHTS_PATH + '/' + weight_name + '.hdf5'
# Dimension of the patches used
patch_dim = 400
# Number of channels of the input images
channels = 3
# Number of filter per layers
layers = [128, 256, 512, 1024]

# Test image generator
test_data_gen = test_generator(patch_dim, num_test_imgs)

# Initialize the model with the images
model = unet1(input_size=(patch_dim, patch_dim, channels), layers=layers, pretrained_weights=weights)

# Obtain the prediction
prediction = model.predict_generator(test_data_gen, num_test_imgs, verbose=1)

# Save the predicted masks
save_results(prediction, num_test_imgs, original_image_size)

# Create a submission from the predicted masks
create_submission(weight_name + '.csv')
