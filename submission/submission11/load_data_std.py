import csv
import numpy as np
from keras.utils import to_categorical
import math
import pandas as pd

train_data_ratio = 0.9


def get_data(is_train):
    data = pd.read_csv("../train.csv")
    train_data_size = math.ceil(len(data) * train_data_ratio)
    if is_train:
        df_model = data.iloc[:train_data_size]
    else:
        df_model = data.iloc[train_data_size:]
    y = df_model.label
    x = df_model.drop('label', axis=1)
    x = x.values.reshape(len(x.values), 28, 28, 1).astype('float32') / 255
    y = to_categorical(y, 10)
    return x, y


def get_submit_data():
    data = pd.read_csv("../test.csv")
    x = data.values.reshape(len(data.values), 28, 28, 1).astype('float32') / 255
    return x
