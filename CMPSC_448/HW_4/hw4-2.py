import numpy as np
import csv
from sklearn.decomposition import PCA as skpca
import matplotlib.pyplot as plt


def load_data():
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """

    np_data = np.genfromtxt('iris.data', delimiter=',')
    np_data = np.delete(np_data, 4, axis=1)
    sepal_proportions = np_data[:, 0] / np_data[:, 1]
    petal_proportions = np_data[:, 2] / np_data[:, 3]
    return np.array([sepal_proportions, petal_proportions]).T


labels = []
with open('iris.data', 'r') as f:
    raw_data = list(csv.reader(f))
    for datapoint in raw_data:
        if datapoint[4] == 'Iris-setosa':
            labels.append(0)
        elif datapoint[4] == 'Iris-versicolor':
            labels.append(1)
        else:
            labels.append(2)

iris_data = np.genfromtxt('iris.data', delimiter=',')
iris_data = np.delete(iris_data, 4, axis=1)
iris_proportions = load_data()

var_ratios_unprocessed = []
for i in range(1, 5):
    pca = skpca(n_components=i)
    pca.fit(iris_data)
    var_ratios_unprocessed.append(pca.explained_variance_ratio_)

var_ratios_preprocessed = []
for i in range(1, 3):
    pca = skpca(n_components=i)
    pca.fit(iris_data)
    var_ratios_preprocessed.append(pca.explained_variance_ratio_)

pca = skpca(n_components=2)
pca.fit(iris_data)
projected = pca.transform(iris_data)

plt.scatter(x=projected[:, 0], y=projected[:, 1], c=labels)
plt.show()
