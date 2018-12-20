from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from image_process import *
from data_process import *

"""
This file is used to visualize the model prediction. 
The plots can be seen in the jupyter notebook file "Plottin
"""

def make_img_overlay(img, predicted_img):
    """
    Function provided in the project description
    Overlays an image with its road prediction
    """
    # Creates a color mask 
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255
    
    # Converts the RGB background and the mask images into RGBA images. RGBA specifies the opacity of the color
    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    
    # Creates a new image by interpolating between two RGBA images using an interpolation alpha factor of 0.2
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def concatenate_images(img, gt_img):
    """
    Function provided in the project description
    Concatenates an image and its groundtruth
    """
    # Number of channels of the image, width and height of the image
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    
    # Concatenates RGB images
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# Concatenate two predictions
def concatenate_prediction_images(prediction1, prediction2):
    """
    Function inspired of the function "concatenate_images"
    Concatenates two RGBA images covered by the prediction of the road
    """
    # Converts pillow images into numpy arrays
    prediction1 = pillow2numpy(prediction1)
    prediction2 = pillow2numpy(prediction2)
    nChannels = len(prediction2.shape)
    w = prediction2.shape[0]
    h = prediction2.shape[1]
    #Creates a separation line between the two images 
    black_line = np.zeros((h,100,prediction2.shape[2]),dtype=int)
    # Concatenate images 
    if nChannels == 3:
        cimg = np.concatenate((prediction1,black_line, prediction2), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(prediction2)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    #return the plot of the two images with the separation
    return plt.imshow(cimg)

def prediction_2_image(nb_image1,nb_image2):
    """
    Returns the plot of two RGBA images covered by the prediction of the road with a 
    separation bewteen the two using as input the number of these images
    """
    # Accessing test images in test file  using the test folder path 
    test_image_1 = np.asarray([load_image(TEST_FOLDER_PATH+ "/test_"+str(nb_image1)+"/test_"+str(nb_image1)+".png")])
    test_image_1 = test_image_1.squeeze()
    test_image_2 = np.asarray([load_image(TEST_FOLDER_PATH+ "/test_"+str(nb_image2)+"/test_"+str(nb_image2)+".png")])
    test_image_2 = test_image_2.squeeze()
    
    #Accessing prediction images associated to the test images in prediction file using the predicted image path
    prediction_1 = np.asarray([load_image(PREDICTED_IMAGES_PATH+"/prediction"+str(nb_image1)+".png")])
    prediction_2 = np.asarray([load_image(PREDICTED_IMAGES_PATH+"/prediction"+str(nb_image2)+".png")])
    
    # Overlays the test images with their road predictions 
    new_img_1 = make_img_overlay(test_image_1, prediction_1)
    new_img_2 = make_img_overlay(test_image_2, prediction_2)
    
    # Concatenates and plots two RGBA images covered by the prediction of the road
    concatenate_prediction_images(new_img_1,new_img_2)
