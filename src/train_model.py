from model import *
from data_process import *
from generator import *
from training import *


images, label = load_image_train()

in_size = 400
model = unet(input_size=(in_size, in_size, 3), pretrained_weights=None)
#train_image_generator = train_generator2(images, label, 1)
train_image_generator = train_generator(in_size)

model_checkpoint = ModelCheckpoint('weights.hdf5', monitor='acc', verbose=1, save_best_only=True)
Learning_reduction = ReduceLROnPlateau(monitor='acc', factor=0.3, patience=10, verbose=1, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)
Early_Stopping = EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto',
                               baseline=None, restore_best_weights=False)

model.fit_generator(train_image_generator, steps_per_epoch=10000, epochs=50, callbacks=[
                    Learning_reduction, Early_Stopping, model_checkpoint])
