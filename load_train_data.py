import csv
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import pandas as pd
import math

def get_data(is_train, file_path):
    if is_train:
        data_ratio = 0.7 if is_train else 0.3
    pd_csv = pd.read_csv(file_path)
    data_size = math.ceil(42000*data_ratio)
    x_train = pd_csv.iloc[:data_size, 1:].values.astype('float32')
    y_train = pd_csv.iloc[:data_size, 0].values.astype('float32')
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    # y_train = to_categorical(y_train, 10)

    print(x_train.shape)
    print(y_train.shape)
    return x_train, y_train