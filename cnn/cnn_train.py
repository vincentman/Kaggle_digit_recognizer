import csv
import time
import numpy as np
from keras.utils import to_categorical

xlist, ylist = [], []
with open("../train.csv", 'r') as file:
    csv_reader = csv.reader(file, delimiter=',')
    next(csv_reader)  # skip header
    count = 0
    for line in csv_reader:
        if count >= 42000*0.7:
            break
        xlist.append(line[1:785])
        ylist.append(line[0])
        count += 1
x_train = np.asarray(xlist).reshape((len(xlist), 28, 28, 1)).astype('float32') / 255
y_train = to_categorical(np.asarray(ylist, dtype=np.float32), 10)

print(x_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=36,
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
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
print(model.summary())

epochs = 20
from keras.optimizers import Adam
# learning_rate = 0.0001
# adam = Adam(lr=learning_rate)
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
start = time.time()
train_history=model.fit(x=x_train,
                        y=y_train, validation_split=0.2,
                        epochs=epochs, batch_size=128, verbose=2)

import matplotlib.pyplot as plt
def show_train_history(train_acc, validation_acc, ylabel):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[validation_acc])
    final_epoch_train_acc = train_history.history[train_acc][epochs - 1]
    final_epoch_validation_acc = train_history.history[validation_acc][epochs - 1]
    # plt.text(train_history.epoch[epochs-1], final_epoch_train_acc-0.01, 'train = {:.3f}'.format(final_epoch_train_acc))
    # plt.text(train_history.epoch[epochs-1], final_epoch_validation_acc-0.03, 'valid = {:.3f}'.format(final_epoch_validation_acc))
    plt.text(train_history.epoch[epochs-1], final_epoch_train_acc, 'train = {:.3f}'.format(final_epoch_train_acc))
    plt.text(train_history.epoch[epochs-1], final_epoch_validation_acc-0.01, 'valid = {:.3f}'.format(final_epoch_validation_acc))
    plt.title('Train History')
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.xlim(xmax=epochs+1)
    plt.legend(['train', 'validation'], loc='upper left')
    fig = plt.gcf()
    fig.savefig('./cnn_train_2_{}.png'.format(ylabel), dpi=300)
    plt.clf()
    # plt.show()
show_train_history('acc','val_acc','accuracy')
show_train_history('loss','val_loss','loss')

end = time.time()
print('elapsed training time: {} min, {} sec '.format(int((end - start)/60), int((end - start)%60)))

model.save('cnn_train_model.h5')
