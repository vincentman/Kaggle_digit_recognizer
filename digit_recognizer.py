from sklearn.cross_validation import train_test_split
import csv
import numpy as np
from keras.utils import to_categorical

csv_reader = csv.reader("train.csv", delimiter=',')
xlist, ylist = [], []
for line in csv_reader:
    xlist.append(line[1:784])
    ylist.append(line[0])
x_train = np.asarray(xlist).reshape((len(xlist),28,28,1)).astype('float32')/255
y_train = to_categorical(np.asarray(ylist, dtype=np.float32), 10)
print(x_train)

