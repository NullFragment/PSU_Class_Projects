import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from numpy import random
import csv


def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    random.seed(3306)
    init_centers = []
    center_idxs = [random.randint(0, X.shape[0])]
    init_centers.append(X[center_idxs[0]])
    for center in range(1, k):
        # Use cdist to calculate euclidean distance for each (data point, center) pair
        dists = cdist(X, np.array([init_centers[-1]]))
        norm_dists = dists / dists.max()
        new_center_idxs = np.where(norm_dists > random.rand())[0]
        if np.isin(new_center_idxs, center_idxs).all():
            init_centers.append(X[new_center_idxs[0]])
        else:
            for new_idx in new_center_idxs:
                if new_idx not in center_idxs:
                    center_idxs.append(new_idx)
                    init_centers.append(X[new_idx])
                    break
    return np.array(init_centers)


def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """
    old_centers = k_init(X, k)
    new_centers = []  # Set to different value for while loop
    iter = 0
    while iter < max_iter:
        data_assigns = assign_data2clusters(X, old_centers)
        new_centers = calculate_centers(old_centers, X, data_assigns)
        iter += 1
        if (old_centers == new_centers).all():
            break
        else:
            old_centers = new_centers
    return new_centers


def calculate_centers(C, X, data_map):
    """ Calculates the new centers for each cluster based on the data map
    Parameters
    ----------
    C: array, shape(k ,d)
        The final cluster centers

    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    Returns
    -------
    C_new: array, shape(k ,d)
        The updated cluster centers

    """
    C_new = np.zeros(C.shape)
    for center in range(0, C.shape[0]):
        num_pts = 0  # number of items in cluster
        center_val = np.zeros(X.shape[1])
        for point in range(0, X.shape[0]):
            if data_map[point][center] == 1:
                center_val += X[point]
                num_pts += 1
            if num_pts > 0:
                C_new[center] = center_val / num_pts
            else:
                C_new[center] = C[center]
    return C_new


def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The current cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    # Use cdist to calculate euclidean distance for each (data point, center) pair
    if C.shape[0] == 1:
        C = np.array([C[-1]])
    dists = cdist(X, C)

    # Find index of centers for best fit
    closest_centers = np.argmin(dists, axis=1)

    # Encode to one-hot format
    # Code for one-hot encoding found at
    # https://stackoverflow.com/questions/29831489/numpy-1-hot-array
    # This code was used because SK-learn was not allowed to be imported as sklearn has a
    # built in one-hot encoder
    data_map = np.zeros(dists.shape)
    data_map[np.arange(dists.shape[0]), closest_centers] = 1
    return data_map


def compute_objective(X, C):
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
    errors = np.zeros(X.shape[0])
    center_map = assign_data2clusters(X, C)
    for center in range(0, C.shape[0]):
        for point in range(0, X.shape[0]):
            if center_map[point][center] == 1:
                errors[point] = np.linalg.norm(X[point] - C[center])
    accuracy = sum(errors) / X.shape[0]
    return (accuracy)


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

    # with open('iris.data', 'r') as f:
    #     raw_data = list(csv.reader(f))
    np_data = np.genfromtxt('iris.data', delimiter=',')
    np_data = np.delete(np_data, 4, axis=1)
    sepal_proportions = np_data[:, 0] / np_data[:, 1]
    petal_proportions = np_data[:, 2] / np_data[:, 3]
    return np.array([sepal_proportions, petal_proportions]).T


def plot_accuracy_per_clusters():
    iris_proportions = load_data()
    accuracies = []
    for k in range(1, 6):
        new_centers = k_means_pp(iris_proportions, k, 50)
        accuracies.append(compute_objective(iris_proportions, new_centers))
    plt.plot(np.arange(1, 6), accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Clusters')
    plt.show()


def plot_accuracy_and_clusters():
    iris_proportions = load_data()
    accuracies = []
    new_centers = []
    for iter in range(1, 51):
        new_centers = k_means_pp(iris_proportions, 3, iter)
        accuracies.append(compute_objective(iris_proportions, new_centers))

    # Get cluster one-hot form for final centers and convert back to categorical numbers
    clusters = assign_data2clusters(iris_proportions, new_centers)
    labels = np.argmax(clusters, axis=1)

    # Plot accuracy vs iterations plot
    plt.plot(np.arange(1, 51), accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.show()

    # Plot scatterplot of categorized data
    plt.scatter(x=iris_proportions[:, 0], y=iris_proportions[:, 1], c=labels)
    plt.xlabel('Sepal Proportion')
    plt.ylabel('Petal Proportion')
    plt.show()


plot_accuracy_per_clusters()
plot_accuracy_and_clusters()
