from cnn.load_data_std import get_data
x_test, y_test = get_data(False, '../train.csv')

from keras.models import load_model
model = load_model('cnn_train_model.h5')
scores = model.evaluate(x_test , y_test)
print('CNN, test score: ', scores[1])

