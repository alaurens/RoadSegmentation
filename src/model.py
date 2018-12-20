from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import BatchNormalization
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from neural_net_blocks import *


"""
These two models were created beased on the work done at:
    https://github.com/zhixuhao/unet
"""


def unet_3_pool(input_size=(400, 400, 3), layers=[128, 256, 512, 1024],
                activation='relu', pretrained_weights=None):
    """
    Defines a unet model with 3 pooling layers
    """

    # Define the input
    inputs = Input(input_size)

    # Apply successively 3 down blocks, the descending part of the unet
    pool1, conv1 = down_block1(inputs, layers[0], activation_name=activation)

    pool2, conv2 = down_block1(pool1, layers[1], activation_name=activation)

    pool3, conv3 = down_block1(pool2, layers[2], activation_name=activation)

    # Apply 2 2D convolutions on the lowest part of the unet
    conv4 = straight_block1(pool3, layers[3], activation_name=activation)

    # Resample the image to the size of the original one using successive convolutions and upscalings
    conv5 = up_block1(conv4, layers[2], conv3, activation_name=activation)

    conv6 = up_block1(conv5, layers[1], conv2, activation_name=activation)

    conv7 = up_block1(conv6, layers[0], conv1, activation_name=activation)

    # Apply a sigmoid activation function to every element of the final set of
    # convoluted images to obtain the classification
    conv8 = Conv2D(1, 1, activation='sigmoid')(conv7)

    # Initialize the model
    model = Model(inputs=inputs, outputs=conv8)

    # Choose the optimizer, loss function and metrics for the model
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    # Load the pre trained weights if available
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def unet_4_pool(input_size=(400, 400, 3), layers=[64, 128, 256, 512, 1024],
                activation='relu', pretrained_weights=None):
    """
    Defines a unet model with 4 pooling layers structure similar to the one with
    3 pooling layers (see previous model for description of layers)
    """
    inputs = Input(input_size)

    pool1, conv1 = down_block1(inputs, layers[0], activation_name=activation)

    pool2, conv2 = down_block1(pool1, layers[1], activation_name=activation)

    pool3, conv3 = down_block1(pool2, layers[2], activation_name=activation)

    pool4, conv4 = down_block1(pool3, layers[3], activation_name=activation)

    conv5 = straight_block1(pool4, layers[4], activation_name=activation)

    conv6 = up_block1(conv5, layers[3], conv4, activation_name=activation)

    conv7 = up_block1(conv6, layers[2], conv3, activation_name=activation)

    conv8 = up_block1(conv7, layers[1], conv2, activation_name=activation)

    conv9 = up_block1(conv8, layers[0], conv1, activation_name=activation)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
