from load_train_data import get_data

x_train, y_train = get_data(True, 'train.csv')

from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0, verbose=True)
svm.fit(x_train, y_train)
print(svm.score(x_train, y_train))