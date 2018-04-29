from svm.load_data_std import get_data

x_train, y_train = get_data(True, '../train.csv')

from sklearn.externals import joblib
pca = joblib.load('pca_dump.pkl')
x_train_pca = pca.transform(x_train)

import time
start = time.time()
from sklearn.svm import SVC
# svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10, verbose=True)
# svm.fit(x_train_pca, y_train)
# from sklearn.linear_model import SGDClassifier
# svm = SGDClassifier(loss='hinge', alpha=0.1, verbose=0)
# n_batch = 500
# n_loop = len(x_train_pca)//n_batch
# n_last_batch = len(x_train_pca)%n_batch
# for i in range(1, n_loop+1):
#     svm.partial_fit(x_train_pca[(i-1)*n_batch:i*n_batch, :], y_train[(i-1)*n_batch:i*n_batch], classes=np.arange(0, 10))
# if n_last_batch!=0:
#     svm.partial_fit(x_train_pca[n_batch*n_loop:, :], y_train[n_batch*n_loop:])
# print('SVM with PCA, train score: ', svm.score(x_train_pca, y_train))
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_range = [0.01, 0.1, 1.0, 10.0]
param_grid = [{'C': param_range, 'gamma': param_range, 'kernel': ['rbf']}]
svm = SVC(random_state=0, verbose=True)
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator=svm,
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=5)
gs.fit(x_train_pca, y_train)
print('SVM with PCA, train score: ', gs.best_score_)
end = time.time()
print('SVM, elapsed training time: {} min, {} sec '.format(int((end - start)/60), int((end - start)%60)))

# joblib.dump(gs.best_estimator_, 'svm_dump.pkl')
joblib.dump(svm, 'svm_dump.pkl')




from svm.load_data_std import get_data
x_test, y_test = get_data(False, 'test.csv')
# print('SVM with PCA, test score: ', gs.best_estimator_.score(x_test, y_test))
print('SVM with PCA, test score: ', svm.score(pca.transform(x_test), y_test))