from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import BatchNormalization
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras


DROPOUT = 0.5


def up_block(input_layer, layer_size, concat_layer, batch_norm=True,
             activation="relu", padding='same', kernel_init='he_normal'):

    up = Conv2D(layer_size, 2, activation=activation, padding=padding,
                kernel_initializer=kernel_init)(UpSampling2D(size=(2, 2))(input_layer))

    merge = concatenate([concat_layer, up], axis=3)

    conv = Conv2D(layer_size, 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_init)(merge)

    if batch_norm:
        conv = BatchNormalization()(conv)

    conv = Conv2D(layer_size, 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_init)(conv)

    return conv


def down_block(input_layer, layer_size, dropout=0, batch_norm=True,
               activation="relu", padding='same', kernel_init='he_normal'):

    conv = Conv2D(layer_size, 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_init)(input_layer)

    if batch_norm:
        conv = BatchNormalization()(conv)

    conv = Conv2D(layer_size, 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_init)(conv)

    if not dropout == 0:
        conv = Dropout(0.5)(conv)

    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return pool, conv


def straight_block(input_layer, layer_size, dropout=0, batch_norm=True,
                   activation="relu", padding='same', kernel_init='he_normal'):

    conv = Conv2D(layer_size, 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_init)(input_layer)

    if batch_norm:
        conv = BatchNormalization()(conv)

    conv = Conv2D(layer_size, 3, activation=activation, padding=padding,
                  kernel_initializer=kernel_init)(conv)

    if not dropout == 0:
        conv = Dropout(0.5)(conv)

    return conv


def unet3(input_size=(400, 400, 3), layers=[16, 32, 64, 128], pretrained_weights=None):

    inputs = Input(input_size)
    pool1, conv1 = down_block(inputs, layers[0])
    # conv1 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = BatchNormalization()(conv1)
    # conv1 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool2, conv2 = down_block(pool1, layers[1])
    # conv2 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    pool3, conv3 = down_block(pool2, layers[2], dropout=DROPOUT)
    # conv3 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # conv3 = BatchNormalization()(conv3)
    # conv3 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # drop3 = Dropout(0.5)(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = straight_block(pool3, layers[3], dropout=DROPOUT)
    #conv4 = Conv2D(layers[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    #conv4 = BatchNormalization()(conv4)
    #conv4 = Conv2D(layers[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    #drop4 = Dropout(0.5)(conv4)

    conv5 = up_block(conv4, layers[2], conv3)
    #up5 = Conv2D(layers[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop4))
    #merge5 = concatenate([drop3, up5], axis=3)
    #conv5 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    #conv5 = BatchNormalization()(conv5)
    #conv5 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    conv6 = up_block(conv5, layers[1], conv2)
    #up6 = Conv2D(layers[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    #merge6 = concatenate([conv2, up6], axis=3)
    #conv6 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    #conv6 = BatchNormalization()(conv6)
    #conv6 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    conv7 = up_block(conv6, layers[0], conv1)
    #up7 = Conv2D(layers[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    #merge7 = concatenate([conv1, up7], axis=3)
    #conv7 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    #conv7 = BatchNormalization()(conv7)
    #conv7 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv8 = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=conv8)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


unet3()


def unet(input_size=(400, 400, 3), layers=[16, 32, 64, 128], pretrained_weights=None):

    inputs = Input(input_size)
    conv1 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(layers[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(layers[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    up5 = Conv2D(layers[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([drop3, up5], axis=3)
    conv5 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    # conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(layers[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(layers[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    # conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv8 = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=conv8)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def unet2(input_size=(400, 400, 3), layers=[16, 32, 64, 128], pretrained_weights=None):

    inputs = Input(input_size)
    conv1 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(layers[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(layers[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    up5 = Conv2D(layers[2], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([drop3, up5], axis=3)
    conv5 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(layers[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(layers[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(layers[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(layers[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(layers[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv8 = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=conv8)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


unet2()
