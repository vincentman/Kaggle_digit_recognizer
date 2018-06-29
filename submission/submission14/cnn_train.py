import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.regularizers import l2
from cnn.load_data_std import get_data
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

x_train, y_train = get_data(True)
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)

model = Sequential()
model.add(Conv2D(filters=36,
                 kernel_size=(3, 3),
                 padding='same',
                 input_shape=(28, 28, 1),
                 activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
# model.add(Dense(256, activation='relu'))
model.add(Dense(units=512))
# model.add(Dense(units=256, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
print(model.summary())

epochs = 30
# from keras.optimizers import Adam
# learning_rate = 0.0001
# adam = Adam(lr=learning_rate)
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor='val_loss', patience=3)
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=1e-5)
callbacks = [learning_rate_reduction]
# callbacks = None
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

start = time.time()
train_history = model.fit(x=x_train,
                          y=y_train, validation_split=0.2,
                          epochs=epochs, batch_size=32, verbose=2,
                          callbacks=callbacks)


def show_train_history(train_acc, validation_acc, ylabel):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[validation_acc])
    final_epoch_train_acc = train_history.history[train_acc][epochs - 1]
    final_epoch_validation_acc = train_history.history[validation_acc][epochs - 1]
    plt.text(train_history.epoch[epochs - 1], final_epoch_train_acc, 'train = {:.3f}'.format(final_epoch_train_acc))
    plt.text(train_history.epoch[epochs - 1], final_epoch_validation_acc - 0.01,
             'valid = {:.3f}'.format(final_epoch_validation_acc))
    plt.title('Train History')
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    plt.xlim(xmax=epochs + 1)
    plt.legend(['train', 'validation'], loc='upper left')
    fig = plt.gcf()
    fig.savefig('./cnn_train_{}.png'.format(ylabel), dpi=100)
    plt.clf()
    # plt.show()
    return final_epoch_train_acc, final_epoch_validation_acc


train_acc, validation_acc = show_train_history('acc', 'val_acc', 'accuracy')
train_loss, validation_loss = show_train_history('loss', 'val_loss', 'loss')

end = time.time()
elapsed_train_time = 'elapsed training time: {} min, {} sec '.format(int((end - start) / 60), int((end - start) % 60))
print(elapsed_train_time)

model.save('cnn_train_model.h5')

with open('cnn_train_info.txt', 'w') as file:
    file.write(elapsed_train_time + '\n')
    file.write('train accuracy = {}, validation accuracy = {}\n'.format(train_acc, validation_acc))
    file.write('train loss = {}, validation loss = {}\n'.format(train_loss, validation_loss))
