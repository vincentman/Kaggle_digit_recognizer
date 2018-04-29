from sklearn.decomposition import PCA
from svm.load_data_std import get_data
import time

x_train, y_train = get_data(True, '../train.csv')

start = time.time()
# pca = PCA(n_components=16)
# pca = PCA(n_components=64, svd_solver='randomized',
#           whiten=True)
pca = PCA(n_components=0.8)
pca.fit(x_train)
print('PCA, explained_variance_ratio:\n', pca.explained_variance_ratio_)
print('PCA, explained_variance_ratio sum: ', pca.explained_variance_ratio_.sum())
print('PCA, n_components_: ', pca.n_components_)

end = time.time()
print('PCA, elapsed training time: {} min, {} sec '.format(int((end - start)/60), int((end - start)%60)))

from sklearn.externals import joblib
joblib.dump(pca, 'pca_dump.pkl')