from cnn.load_data_std import get_submit_data
from keras.models import load_model
import pandas as pd

x_submit = get_submit_data()
print('x_submit.shape: ', x_submit.shape)

model = load_model('cnn_train_model.h5')
prediction = model.predict_classes(x_submit)
print('CNN, 10 of prediction for submission = ', prediction[:10])

df = pd.DataFrame(prediction)
df.index += 1
df.index.name = 'ImageId'
df.columns = ['Label']
df.to_csv('submission.csv', header=True)
