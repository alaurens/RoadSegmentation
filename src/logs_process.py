import csv
import ast
import os
from paths_to_data import LOGS_PATH

"""
This file contains all the functions to process the logs that we generate during training
as well as the function to generate the logs themselves
"""


def get_all_maxes(acc, loss, val_acc, val_loss):
    """Get the max value of the strings of accuracy,loss,validation accuracy and
    validation loss that have the format "[a b c e d ...]" """
    # Get all the numbers for each array, make sure to remove '[' and ']'
    acc = [float(num) for num in acc[1:-2].split(' ')]
    loss = [float(num) for num in loss[1:-2].split(' ')]
    val_acc = [float(num) for num in val_acc[1:-2].split(' ')]
    val_loss = [float(num) for num in val_loss[1:-2].split(' ')]
    # Return a string with the max value of each array
    return [str(max(acc)), str(min(loss)), str(max(val_acc)), str(min(val_loss))]


def transform_line(line, start, end):
    """Gets a line from a log and transforms it """
    r = line.split(',')
    # Get the accuracy, loss ,validation accuracy and loss
    acc, loss, val_acc, val_loss = r[start:end]
    # Get the max values of the previous elements
    r[start:end] = get_all_maxes(acc, loss, val_acc, val_loss)
    # Recreate the line
    line = ','.join(r) + '\n'
    return line


def combine_all_logs():
    """Combine all the individual log files into one large file"""
    # The number of files to take
    num_files = 350
    fout = open(LOGS_PATH + "/combined.csv", "w")
    # first file:
    f = open(LOGS_PATH + "/log1.csv")
    header = f.readline()

    # Get the index of accuration and validation loss elements in the array
    start = header.split(',').index("accuracy")
    if "validation loss" in header.split(','):
        end = header.split(',').index("validation loss") + 1
    else:
        end = header.split(',').index("validation loss\n") + 1

    fout.write('log_num,' + header)
    # now the rest:
    for num in range(1, num_files+1):
        f = open(LOGS_PATH + "/log"+str(num)+".csv")
        f.__next__()  # skip the header
        for line in f:
            line = str(num) + ',' + transform_line(line, start, end)
            fout.write(line)
        f.close()  # not really needed
    fout.close()


def log_info(iter, in_size, layers, epochs, steps_per_epoch, acc_list,
             val_acc_list, loss_list, val_loss_list, activation, relabel_mask):
    """Logs different information for each epoch"""
    if not os.path.exists(LOGS_PATH):
        os.mkdir(LOGS_PATH)
    with open(LOGS_PATH + '/log' + str(iter) + '.csv', mode='w') as log_file:
        log_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Write the header
        log_writer.writerow(['input size', 'layers', 'number epochs',
                             'step per epoch', 'accuracy',
                             'loss', 'validation accuracy',
                             'validation loss', 'activation', 'relabel mask'])
        # Write the different elements
        log_writer.writerow([str(in_size),
                             '[' + ' '.join(str(l) for l in layers) + ']',
                             str(epochs),
                             str(steps_per_epoch),
                             '[' + ' '.join("{:.6f}".format(l) for l in acc_list) + ']',
                             '[' + ' '.join("{:.6f}".format(l) for l in loss_list) + ']',
                             '[' + ' '.join("{:.6f}".format(l) for l in val_acc_list) + ']',
                             '[' + ' '.join("{:.6f}".format(l) for l in val_loss_list) + ']',
                             activation,
                             relabel_mask])


if __name__ == '__main__':
    combine_all_logs()
