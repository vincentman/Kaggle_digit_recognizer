from cnn.load_data_std import get_data
from keras.models import load_model

x_test, y_test = get_data(False)

model = load_model('cnn_train_model.h5')
scores = model.evaluate(x_test , y_test)
print('CNN, test score: ', scores[1])

with open('cnn_predict_info.txt', 'w') as file:
    file.write('test accuracy = {}\n'.format(scores[1]))