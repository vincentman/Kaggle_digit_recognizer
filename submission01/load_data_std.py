import csv
import numpy as np
from keras.utils import to_categorical

train_data_ratio = 0.7
def get_data(is_train, file_path):
    xlist, ylist = [], []
    data_ratio = train_data_ratio if is_train else 1-train_data_ratio
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)  # skip header
        count = 0
        for line in csv_reader:
            if count >= 42000*data_ratio:
                break
            xlist.append(line[1:785])
            ylist.append(line[0])
            count += 1
    x_train = np.asarray(xlist).reshape((len(xlist), 28, 28, 1)).astype('float32') / 255
    y_train = to_categorical(np.asarray(ylist, dtype=np.float32), 10)
    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train

def get_submit_data(file_path):
    xlist, ylist = [], []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)  # skip header
        for line in csv_reader:
            xlist.append(line[:784])
    x_submit = np.asarray(xlist).reshape((len(xlist), 28, 28, 1)).astype('float32') / 255
    print(x_submit.shape)
    return x_submit

