from cnn.load_data_std import get_submit_data
x_submit = get_submit_data('../test.csv')

from keras.models import load_model
model = load_model('cnn_train_model.h5')
prediction = model.predict_classes(x_submit)
print('CNN, prediction[:10]: ', prediction[:10])

import pandas as pd
df=pd.DataFrame(prediction)
df.index += 1
df.index.name = 'ImageId'
df.columns=['Label']
df.to_csv('submission.csv', header=True)
