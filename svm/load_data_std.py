# from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import pandas as pd
import math

def get_data(is_train, file_path):
    data_ratio = 0.7 if is_train else 0.3
    pd_csv = pd.read_csv(file_path)
    data_size = math.ceil(42000*data_ratio)
    feature_start_idx = 1 if is_train else 0
    x = pd_csv.iloc[:data_size, feature_start_idx:].values.astype('float32')
    y = pd_csv.iloc[:data_size, 0].values.astype('float32')
    sc = StandardScaler()
    sc.fit(x)
    x = sc.transform(x)
    # y = to_categorical(y, 10)

    train_or_test = 'train' if is_train else 'test'
    print("x_{}.shape = {}".format(train_or_test, x.shape))
    print("y_{}.shape = {}".format(train_or_test, y.shape))
    return x, y