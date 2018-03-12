import random
import numpy as np
import matplotlib.pyplot as plt
from src.homework2 import *


def load_data():
    data = np.load('data.npy')
    x, y = np.split(data, 2, 1)
    return x, y


def get_width(arr):
    """
        Getwidth returns the width of a given array
    """
    if len(arr.shape) > 1:
        return arr.shape[1]
    else:
        return 1


def init_values(x, y):
    """
        Initializes the weight matrix and other useful values for gradient descent

        Args:
            x - matrix of m samples by n dimensional data
            y - matrix of m samples by p categorical data

        Returns:
            x_dim - number of dimensions in x data
            y_cat - number of categories in y data
            num_samples - total number of samples
            w - weight matrix
    """
    x_dim = get_width(x)
    y_cat = get_width(y)
    num_samples = x.shape[0]
    w = np.random.rand(x_dim, y_cat)
    return x_dim, y_cat, num_samples, w


if __name__ == '__main__':
    # Initialize weights and data values
    x_data, y_data = load_data()
    n, p, m, weights = init_values(x_data, y_data)

    # Running BGD:
    w_bgd_1, hist_bgd_1 = bgd_l2(x_data, y_data, weights, 0.05, 0.10, 0.001, 50)
    w_bgd_2, hist_bgd_2 = bgd_l2(x_data, y_data, weights, 0.10, 0.01, 0.001, 50)
    w_bgd_3, hist_bgd_3 = bgd_l2(x_data, y_data, weights, 0.10, 0.00, 0.001, 100)
    w_bgd_4, hist_bgd_4 = bgd_l2(x_data, y_data, weights, 0.10, 0.00, 0.000, 100)

    # Plotting BGD:
    plt.clf()
    plt.subplots_adjust(hspace=.5)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(hist_bgd_1, linestyle='-')
    plt.plot(hist_bgd_2, linestyle=':')
    plt.plot(hist_bgd_3, linestyle='-.')
    plt.plot(hist_bgd_4, linestyle='--')
    plt.xlabel("Time Steps")
    plt.ylabel("Objective Function [f(w)]")
    plt.title("BGD Cost")
    plt.legend(['Parameter Set 1', 'Parameter Set 2', 'Parameter Set 3', 'Parameter Set 4'], loc='upper right')

    # Running SGD:
    w_sgd_1, hist_sgd_1 = sgd_l2(x_data, y_data, weights, 1, 0.10, 0.05, 800)
    w_sgd_2, hist_sgd_2 = sgd_l2(x_data, y_data, weights, 1, 0.01, 0.01, 800)
    w_sgd_3, hist_sgd_3 = sgd_l2(x_data, y_data, weights, 1, 0.00, 0.00, 40)
    w_sgd_4, hist_sgd_4 = sgd_l2(x_data, y_data, weights, 1, 0.00, 0.00, 800)

    # Plotting SGD:
    plt.subplot(212)
    plt.plot(hist_sgd_1, linestyle='-')
    plt.plot(hist_sgd_2, linestyle=':')
    plt.plot(hist_sgd_3, linestyle='-.')
    plt.plot(hist_sgd_4, linestyle='--')
    plt.xlabel("Time Steps")
    plt.ylabel("Objective Function [f(w)]")
    plt.title("SGD Cost")
    plt.legend(['Parameter Set 1', 'Parameter Set 2', 'Parameter Set 3', 'Parameter Set 4'], loc='upper right')
    plt.show()
