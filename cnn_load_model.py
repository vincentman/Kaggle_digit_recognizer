import csv
import numpy as np
from keras.utils import to_categorical

import time
start = time.time()
xlist, ylist = [], []
with open("train.csv", 'r') as file:
    csv_reader = csv.reader(file, delimiter=',')
    next(csv_reader)  # skip header
    count = 0
    for line in reversed(list(csv_reader)):
        if count >= 42000*0.3:
            break
        xlist.append(line[1:785])
        ylist.append(line[0])
        count += 1
x_test = np.asarray(xlist).reshape((len(xlist), 28, 28, 1)).astype('float32') / 255
y_test = to_categorical(np.asarray(ylist, dtype=np.float32), 10)

print(x_test.shape)
print(y_test.shape)

from keras.models import load_model
model = load_model('cnn_train_2_model.h5')
scores = model.evaluate(x_test , y_test)
print(scores[1])

