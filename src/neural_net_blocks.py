from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import BatchNormalization
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras


def activation_func(name):
    if name == 'leaky':
        return LeakyReLU()
    if name == 'relu':
        return ReLU()
    if name == 'elu':
        return ELU(alpha=1.0)


def up_block1(input_layer, layer_size, concat_layer, batch_norm=True,
              activation_name='relu', padding='same', kernel_init='he_normal'):

    up = UpSampling2D(size=(2, 2))(input_layer)

    conv = Conv2D(layer_size, 2, padding=padding, kernel_initializer=kernel_init)(up)
    conv = activation_func(activation_name)(conv)

    merge = concatenate([concat_layer, conv], axis=3)

    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(merge)
    conv = activation_func(activation_name)(conv)

    if batch_norm:
        conv = BatchNormalization()(conv)

    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(conv)
    conv = activation_func(activation_name)(conv)

    if batch_norm:
        conv = BatchNormalization()(conv)

    return conv


def down_block1(input_layer, layer_size, dropout=0, batch_norm=True,
                activation_name='relu', padding='same', kernel_init='he_normal'):

    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(input_layer)
    conv = activation_func(activation_name)(conv)

    if batch_norm:
        conv = BatchNormalization()(conv)

    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(conv)
    conv = activation_func(activation_name)(conv)

    if batch_norm:
        conv = BatchNormalization()(conv)

    if not dropout == 0:
        conv = Dropout(dropout)(conv)

    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return pool, conv


def straight_block1(input_layer, layer_size, dropout=0, batch_norm=True,
                    activation_name='relu', padding='same', kernel_init='he_normal'):

    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(input_layer)
    conv = activation_func(activation_name)(conv)

    if batch_norm:
        conv = BatchNormalization()(conv)

    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(conv)
    conv = activation_func(activation_name)(conv)

    if batch_norm:
        conv = BatchNormalization()(conv)

    if not dropout == 0:
        conv = Dropout(dropout)(conv)

    return conv


def BN_CONV_RELU(input_layer, layer_size, activation_name='relu',
                 padding='same', kernel_init='he_normal'):

    batch_norm = BatchNormalization()(input_layer)

    conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(batch_norm)

    conv = activation_func(activation_name)(conv)
    return conv


def BN_UPCONV_RELU(input_layer, layer_size, activation_name='relu',
                   padding='same', kernel_init='he_normal'):

    batch_norm = BatchNormalization()(input_layer)

    conv = Conv2DTranspose(layer_size, 3, strides=(2, 2), padding=padding,
                           kernel_initializer=kernel_init)(batch_norm)

    conv = activation_func(activation_name)(conv)

    return conv


def down_block2(input_layer, layer_size, activation_name='relu', first=False,
                padding='same', kernel_init='he_normal'):

    if first:
        conv = Conv2D(layer_size, 3, padding=padding, kernel_initializer=kernel_init)(input_layer)
        conv = activation_func(activation_name)(conv)
    else:
        conv = BN_CONV_RELU(input_layer, layer_size, activation_name=activation_name,
                            padding='same', kernel_init='he_normal')

    concat_conv = BN_CONV_RELU(conv, layer_size, activation_name=activation_name,
                               padding='same', kernel_init='he_normal')

    conv = BN_CONV_RELU(concat_conv, layer_size, activation_name=activation_name,
                        padding='same', kernel_init='he_normal')

    pool = MaxPooling2D(pool_size=(2, 2))(conv)

    return pool, concat_conv


def up_block2(input_layer, layer_size, activation_name='relu',
              concat_layer=None, output=False):

    if not concat_layer == None:
        input_layer = concatenate([concat_layer, input_layer], axis=3)

    conv = BN_CONV_RELU(input_layer, layer_size, activation_name=activation_name,
                        padding='same', kernel_init='he_normal')

    conv = BN_CONV_RELU(conv, layer_size, activation_name=activation_name,
                        padding='same', kernel_init='he_normal')

    if output:
        return Conv2D(1, 1, activation='sigmoid')(conv)
    else:
        return BN_UPCONV_RELU(conv, layer_size, activation_name=activation_name,
                              padding='same', kernel_init='he_normal')
