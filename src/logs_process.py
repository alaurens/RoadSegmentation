import csv
import ast
import os
from paths_to_data import LOGS_PATH


def get_all_maxes(acc, loss, val_acc, val_loss):
    acc = [float(num) for num in acc[1:-2].split(' ')]
    loss = [float(num) for num in loss[1:-2].split(' ')]
    val_acc = [float(num) for num in val_acc[1:-2].split(' ')]
    val_loss = [float(num) for num in val_loss[1:-2].split(' ')]
    return [str(max(acc)), str(min(loss)), str(max(val_acc)), str(min(val_loss))]


def transform_line(line):
    r = line.split(',')
    acc, loss, val_acc, val_loss = r[4:]
    r[4:] = get_all_maxes(acc, loss, val_acc, val_loss)
    line = ','.join(r) + '\n'
    return line


def combine_all_logs():
    num_files = 255
    fout = open(LOGS_PATH + "/combined.csv", "w")
    # first file:
    f = open(LOGS_PATH + "/log1.csv")
    fout.write('log_num,' + f.readline())
    # now the rest:
    for num in range(1, num_files+1):
        f = open(LOGS_PATH + "/log"+str(num)+".csv")
        f.__next__()  # skip the header
        for line in f:
            line = str(num) + ',' + transform_line(line)
            fout.write(line)
        f.close()  # not really needed
    fout.close()


def log_info(iter, in_size, layers, epochs, steps_per_epoch, acc_list,
             val_acc_list, loss_list, val_loss_list):
    if not os.path.exists(LOGS_PATH):
        os.mkdir(LOGS_PATH)
    with open(LOGS_PATH + '/log' + str(iter) + '.csv', mode='w') as log_file:
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
