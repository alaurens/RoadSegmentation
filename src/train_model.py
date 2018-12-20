from model import *
from data_process import *
from generator import *
from paths_to_data import *
import os
import csv
from logs_process import log_info

# Number of iteration up to now for the logging
iter = 323

# Boolean to decide if relabeling of the mask image
relabel_mask = True
# Number of epoch at a time before logging
epoch_step = 10
# Number of steps per epoch
steps_per_epoch = 1000
# Number of epochs
num_epoch = 200

# Create the log file if necessary
if not os.path.exists(LOGS_PATH):
    os.mkdir(LOGS_PATH)

# Select either the first or the second model and the number of filter per
# convolution layer as well as the patch size

# model_num = 1
# layers = [256, 512, 1024, 2048]
# in_sizes = [400]

model_num = 2
layers = [64, 128, 256, 512, 1024]
in_sizes = [400]

# If you want to start training from pre-trained weights
pre_weights = WEIGHTS_PATH + '/weights322.hdf5'

# Activation function for the convolutional layers
activation = 'relu'

# Loop through the patcvh sizes
for in_size in in_sizes:

    # Initialize the train image generator and the validation image generator
    train_data_gen = train_generator(in_size, relabel_mask=relabel_mask)
    validation_data_gen = validation_generator(in_size, relabel_mask=relabel_mask)

    # Select the desired model and initlaize it
    if model_num == 1:
        model = unet_3_pool(input_size=(in_size, in_size, 3), layers=layers,
                            activation=activation, pretrained_weights=pre_weights)
    else:
        model = unet_4_pool(input_size=(in_size, in_size, 3), layers=layers,
                            activation=activation, pretrained_weights=pre_weights)

    # Define the callbacks for the keras training
    # Reduce the learning rate if stop improving
    Learning_reduction = ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=10,
                                           verbose=1, mode='auto', min_delta=0.0001,
                                           cooldown=0, min_lr=0)
    # Stop if training stops imprving for to long
    Early_Stopping = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=10,
                                   verbose=1, mode='auto', baseline=None,
                                   restore_best_weights=True)

    # Between each set of epoch keep the last epoch done
    last_epoch = 0
    # Go through a fraction of the total number of epoch
    for epochs in range(1, int(num_epoch/epoch_step) + 1):
        # Increase the iteration for the log file
        iter = iter+1

        # Starting and last epoch for this round
        init_epoch = last_epoch
        last_epoch = epoch_step*epochs

        # Reinitilizes the history each time
        history = History()
        # Callback for the model checkpoint to save the weights obtained on this set of epoch
        model_checkpoint = ModelCheckpoint(LOGS_PATH + '/weights' + str(iter) + '.hdf5',
                                           monitor='val_acc', verbose=1, save_best_only=True)

        # Train the model
        model.fit_generator(train_data_gen, steps_per_epoch=steps_per_epoch,
                            epochs=last_epoch, initial_epoch=init_epoch,
                            callbacks=[Learning_reduction, history,
                                       Early_Stopping, model_checkpoint],
                            validation_steps=20, validation_data=validation_data_gen,
                            use_multiprocessing=False)

        # Get the history and save to the logs
        hist = history.history
        log_info(iter, in_size, layers, last_epoch, steps_per_epoch,
                 hist['acc'], hist['val_acc'],
                 hist['loss'], hist['val_loss'], activation, relabel_mask)
