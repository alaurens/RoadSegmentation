from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import BatchNormalization
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from neural_net_blocks import *


def unet1(input_size=(400, 400, 3), layers=[16, 32, 64, 128],
          activation='relu', pretrained_weights=None):
    inputs = Input(input_size)

    pool1, conv1 = down_block1(inputs, layers[0], activation_name=activation)

    pool2, conv2 = down_block1(pool1, layers[1], activation_name=activation)

    pool3, conv3 = down_block1(pool2, layers[2], activation_name=activation)

    conv4 = straight_block1(pool3, layers[3], activation_name=activation)

    conv5 = up_block1(conv4, layers[2], conv3, activation_name=activation)

    conv6 = up_block1(conv5, layers[1], conv2, activation_name=activation)

    conv7 = up_block1(conv6, layers[0], conv1, activation_name=activation)

    conv8 = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=conv8)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def unet2(input_size=(320, 320, 3), layers=[64]*5, activation='relu', pretrained_weights=None):
    """ 
    The following unet is copied from the structure proposed by the following website:
    https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/
    """
    inputs = Input(input_size)

    pool, concat_conv1 = down_block2(inputs, layers[0], first=True)

    pool, concat_conv2 = down_block2(pool, layers[1], activation_name=activation)

    pool, concat_conv3 = down_block2(pool, layers[2], activation_name=activation)

    pool, concat_conv4 = down_block2(pool, layers[3], activation_name=activation)

    pool, concat_conv5 = down_block2(pool, layers[4], activation_name=activation)

    upconv = up_block2(pool, layers[4], activation_name=activation)

    upconv = up_block2(upconv, layers[4], activation_name=activation, concat_layer=concat_conv5)

    upconv = up_block2(upconv, layers[3], activation_name=activation, concat_layer=concat_conv4)

    upconv = up_block2(upconv, layers[2], activation_name=activation, concat_layer=concat_conv3)

    upconv = up_block2(upconv, layers[1], activation_name=activation, concat_layer=concat_conv2)

    outputs = up_block2(upconv, layers[0], activation_name=activation, concat_layer=concat_conv1, output=True)

    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
