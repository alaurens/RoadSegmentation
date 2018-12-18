from model import *
from data_process import *
from generator import *
from paths_to_data import *
import os
import csv
from logs_process import log_info


iter = 240
epoch_step = 10
steps_per_epoch = 300

if not os.path.exists(LOGS_PATH):
    os.mkdir(LOGS_PATH)

layers_size = [64, 128, 256, 512]

for in_size in [200, 400]:
    train_data_gen = train_generator(in_size)
    validation_data_gen = validation_generator(in_size)

    for i in [1, 2]:
        layers = list(map(lambda x: i*x, layers_size))

        model = unet(input_size=(in_size, in_size, 3), layers=layers)

        Learning_reduction = ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=10,
                                               verbose=1, mode='auto',
                                               min_delta=0.0001, cooldown=0, min_lr=0)
        Early_Stopping = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=10,
                                       verbose=1, mode='auto', baseline=None,
                                       restore_best_weights=False)
        last_epoch = 0
        for epochs in range(1, 10):

            iter = iter+1
            init_epoch = last_epoch
            last_epoch = epoch_step*epochs

            history = History()
            model_checkpoint = ModelCheckpoint(LOGS_PATH + '/weights' + str(iter) + '.hdf5',
                                               monitor='val_acc', verbose=1, save_best_only=True)

            model.fit_generator(train_data_gen, steps_per_epoch=steps_per_epoch,
                                epochs=last_epoch, initial_epoch=init_epoch,
                                callbacks=[Learning_reduction, history,
                                           Early_Stopping, model_checkpoint],
                                validation_steps=20, validation_data=validation_data_gen,
                                use_multiprocessing=True)

            hist = history.history
            log_info(iter, in_size, layers, last_epoch, steps_per_epoch,
                     hist['acc'], hist['val_acc'],
                     hist['loss'], hist['val_loss'])
