from model import *
from data_process import *
from generator import *


in_size = 200
layers = list(map(lambda x: x * 2, [16, 32, 64, 128]))

model = unet(input_size=(in_size, in_size, 3), layers=layers, pretrained_weights=None)

train_data_gen = train_generator(in_size)
validation_data_gen = validation_generator(in_size)

model_checkpoint = ModelCheckpoint('weights.hdf5', monitor='acc', verbose=1, save_best_only=True)
Learning_reduction = ReduceLROnPlateau(monitor='acc', factor=0.3, patience=10, verbose=1, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)
Early_Stopping = EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto',
                               baseline=None, restore_best_weights=False)

model.fit_generator(train_data_gen, steps_per_epoch=500, epochs=70, callbacks=[
                    Learning_reduction, Early_Stopping, model_checkpoint],
                    validation_steps=20, validation_data=validation_data_gen)
