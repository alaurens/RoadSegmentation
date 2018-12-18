from model import *
from data_process import *
from generator import *
from paths_to_data import *
import os
import csv


def log_info(iter, in_size, layers, epochs, steps_per_epoch, unet_num, acc_list,
             val_acc_list, loss_list, val_loss_list):
    if not os.path.exists(LOGS_PATH):
        os.mkdir(LOGS_PATH)
    if unet_num == 1:
        batch_norm = "batch norm"
    else:
        batch_norm = "not batch norm"
    with open(LOGS_PATH + '/2log' + str(iter) + '.csv', mode='w') as log_file:
        log_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        log_writer.writerow(['input size', 'layers', 'number epochs',
                             'step per epoch', 'accuracy',
                             'loss', 'validation accuracy',
                             'validation loss'])
        log_writer.writerow([str(in_size),
                             '[' + ' '.join(str(l) for l in layers) + ']',
                             str(epochs),
                             str(steps_per_epoch),
                             '[' + ' '.join("{:.6f}".format(l) for l in acc_list) + ']',
                             '[' + ' '.join("{:.6f}".format(l) for l in loss_list) + ']',
                             '[' + ' '.join("{:.6f}".format(l) for l in val_acc_list) + ']',
                             '[' + ' '.join("{:.6f}".format(l) for l in val_loss_list) + ']'])


iter = 0
epoch_step = 10
if not os.path.exists(LOGS_PATH):
    os.mkdir(LOGS_PATH)
layers_size = [32, 64, 128, 256]


for in_size in [160, 200, 400]:

    train_data_gen = train_generator(in_size)
    validation_data_gen = validation_generator(in_size)
    test_data_gen = test_generator(in_size)

    for i in range(1, 3):

        layers = list(map(lambda x: (i)*x, layers_size))

        for steps_per_epoch in [300]:
            for unet_num in range(0, 2):
                # if unet_num ==:
                #    model = unet(input_size=(in_size, in_size, 3),
                #                 layers=layers,
                #                 pretrained_weights=None)
                # else:
                model = unet2(input_size=(in_size, in_size, 3),
                              layers=layers,
                              pretrained_weights=None)

                Learning_reduction = ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=10,
                                                       verbose=1, mode='auto',
                                                       min_delta=0.0001, cooldown=0, min_lr=0)
                Early_Stopping = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=10,
                                               verbose=1, mode='auto',
                                               baseline=None, restore_best_weights=False)
                last_epoch = 0
                for epochs in range(1, 7):

                    iter = iter+1
                    init_epoch = last_epoch
                    last_epoch = epoch_step*epochs

                    history = History()
                    model_checkpoint = ModelCheckpoint(LOGS_PATH + '/2weights' + str(iter) + '.hdf5',
                                                       monitor='val_acc', verbose=1, save_best_only=True)

                    model.fit_generator(train_data_gen, steps_per_epoch=steps_per_epoch,
                                        epochs=last_epoch, initial_epoch=init_epoch,
                                        callbacks=[Learning_reduction, history,
                                                   Early_Stopping, model_checkpoint],
                                        validation_steps=20, validation_data=validation_data_gen,
                                        use_multiprocessing=True)

                    hist = history.history
                    log_info(iter, in_size, layers, last_epoch, steps_per_epoch,
                             unet_num, hist['acc'], hist['val_acc'],
                             hist['loss'], hist['val_loss'])
