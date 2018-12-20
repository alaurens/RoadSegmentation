from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import BatchNormalization
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras


def activation_func(name):
    """
    Chooses the activation function of the layer
    """
    if name == 'leaky':
        return LeakyReLU()
    if name == 'relu':
        return ReLU()
    if name == 'elu':
        return ELU(alpha=1.0)


def up_block1(input_layer, layer_size, concat_layer, batch_norm=True,
              activation_name='relu', padding='same', kernel_init='he_normal'):
    """
    Define a up block of a unet network composed of :
        - an upsampling of the input layer
        - a 2D convolution
        - a concatenation with a previous block of the down part of the unet
        - a 2D convolution
        - a batch normalization
        - a 2D convolution
        - a batch normalization
    """
    # Double the image size through up smapling
    up = UpSampling2D(size=(2, 2))(input_layer)

    # Apply 2D convolution
    conv = Conv2D(layer_size, 2, padding=padding, kernel_initializer=kernel_init)(up)
    conv = activation_func(activation_name)(conv)

    # Concatenate with the down scalling layer of same size
    merge = concatenate([concat_layer, conv], axis=3)

    # Apply another 2D convolution
    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(merge)
    conv = activation_func(activation_name)(conv)

    # Batch normalization for regularization
    if batch_norm:
        conv = BatchNormalization()(conv)

    # Next 2D convolution
    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(conv)
    conv = activation_func(activation_name)(conv)

    # Batch normalization for regularization
    if batch_norm:
        conv = BatchNormalization()(conv)

    return conv


def down_block1(input_layer, layer_size, dropout=0, batch_norm=True,
                activation_name='relu', padding='same', kernel_init='he_normal'):
    """
    Define a down block of a unet network composed of :
        - a 2D convolution
        - a batch normalization
        - a 2D convolution
        - a batch normalization
        - a max pooling
    """

    # Apply 2D convolution
    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(input_layer)
    conv = activation_func(activation_name)(conv)

    # Batch normalization for regularization
    if batch_norm:
        conv = BatchNormalization()(conv)

    # Apply 2D convolution
    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(conv)
    conv = activation_func(activation_name)(conv)

    # Batch normalization for regularization
    if batch_norm:
        conv = BatchNormalization()(conv)

    # If desired apply dropout
    if not dropout == 0:
        conv = Dropout(dropout)(conv)

    # Max pooling to downscal the image
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return pool, conv


def straight_block1(input_layer, layer_size, dropout=0, batch_norm=True,
                    activation_name='relu', padding='same', kernel_init='he_normal'):
    """
    Define the lowest block of a unet network composed of :
        - a 2D convolution
        - a batch normalization
        - a 2D convolution
        - a batch normalization
    """
    # Apply 2D convolution
    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(input_layer)
    conv = activation_func(activation_name)(conv)

    # Batch normalization for regularization
    if batch_norm:
        conv = BatchNormalization()(conv)

    # Apply 2D convolution
    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(conv)
    conv = activation_func(activation_name)(conv)

    # Batch normalization for regularization
    if batch_norm:
        conv = BatchNormalization()(conv)

    # If desired apply dropout
    if not dropout == 0:
        conv = Dropout(dropout)(conv)

    return conv
