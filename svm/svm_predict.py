from svm.load_data_std import get_data
x_test, y_test = get_data(False, '../test.csv')

from sklearn.externals import joblib
pca = joblib.load('pca_dump.pkl')
x_test_pca = pca.transform(x_test)

from sklearn.externals import joblib
svm = joblib.load('svm_dump.pkl')

# print(svm.predict(x_test_pca))
print('SVM with PCA, test score: ', svm.score(x_test_pca, y_test))