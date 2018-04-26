from sklearn.model_selection import train_test_split
import csv
import numpy as np
from keras.utils import to_categorical

import time
start = time.time()
xlist, ylist = [], []
with open("train.csv", 'r') as file:
    csv_reader = csv.reader(file, delimiter=',')
    next(csv_reader)  # skip header
    for line in csv_reader:
        xlist.append(line[1:785])
        ylist.append(line[0])
x_train_all = np.asarray(xlist).reshape((len(xlist), 28, 28, 1)).astype('float32') / 255
y_train_all = to_categorical(np.asarray(ylist, dtype=np.float32), 10)
x_train_all /= 255

x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.3, random_state=11)
print(x_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=36,
#                  kernel_size=(5,5),
#                  padding='same',
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
print(model.summary())


model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
from keras.optimizers import Adam
# learning_rate = 0.0001
# adam = Adam(lr=learning_rate)
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,metrics=['accuracy'])
train_history=model.fit(x=x_train,
                        y=y_train, validation_split=0.2,
                        epochs=20, batch_size=128, verbose=2)

import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc,ylabel):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig = plt.gcf()
    fig.savefig('./cnn_train_1_{}.png'.format(ylabel), dpi=300)
    # plt.show()
show_train_history('acc','val_acc','accuracy')
show_train_history('loss','val_loss','loss')

end = time.time()
print('elapsed training time: {} min, {} sec '.format(int((end - start)/60), int((end - start)%60)))

model.save('cnn_train_1_model.h5')
