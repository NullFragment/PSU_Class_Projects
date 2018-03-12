import math
import random
import numpy as np


def fun_g(x, y, w, delta):
    """
        Computes the given g(w,x,y) function

        Args:
            x - matrix of m samples by n dimensions
            y - matrix of m samples by p categorical data
            w - weight matrix of n dimensions by p categories
            delta - given delta value

        Returns:
            result of given function g
    """
    xw = np.multiply(x, w)
    if abs(y - xw) < delta:
        return 0
    elif y >= (xw + delta):
        return pow((y - xw - delta), 2)
    else:
        return pow((y - xw + delta), 2)


def grad_g(x, y, w, delta):
    """
        Computes the gradient of the given g(w,x,y) function

        Args:
            x - matrix of m samples by n dimensions
            y - matrix of m samples by p categorical data
            w - weight matrix of n dimensions by p categories
            delta - given delta value

        Returns:
            result of grad g wrt w
    """
    xw = np.multiply(x, w)
    if abs(y - xw) < delta:
        return 0
    elif y >= (xw + delta):
        return 2 * (y - xw - delta).dot(x)
    else:
        return 2 * (y - xw + delta).dot(x)


def fun_f(x, y, w, delta, lam, sgd=False, idx=0):
    """
        Computes the given f(w) function

        Args:
            x - matrix of m samples by n dimensions
            y - matrix of m samples by p categorical data
            w - weight matrix of n dimensions by p categories
            delta - given delta value
            lam - given lambda regularization value

        Returns:
            result of given function f
    """
    m = x.shape[0]
    sum_g = 0
    if sgd:
        sum_g = fun_g(x[idx], y[idx], w, delta)
    else:
        for i in range(0, m):
            sum_g += fun_g(x[i], y[i], w, delta)
        sum_g = 1 / m * sum_g

    reg = lam * w.dot(w.T)[0, 0]
    return sum_g + reg


def grad_f(x, y, w, delta, lam, sgd=False, idx=0):
    """
        Computes the gradient of the given f(w) function

        Args:
            x - matrix of m samples by n dimensions
            y - matrix of m samples by p categorical data
            w - weight matrix of n dimensions by p categories
            delta - given delta value
            lam - given lambda regularization value

        Returns:
            result of grad f wrt w
    """
    m = x.shape[0]
    sum_grad_g = 0
    if sgd:
        sum_grad_g = grad_g(x[idx], y[idx], w, delta)
    else:
        for i in range(0, m):
            sum_grad_g += grad_g(x[i], y[i], w, delta)
        sum_grad_g = 1 / m * sum_grad_g
    reg = lam * 2 * np.sum(w)
    return (sum_grad_g + reg)


def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    history_fw = []
    new_w = np.copy(w)
    for i in range(0, num_iter):
        history_fw.append(fun_f(data, y, new_w, delta, lam)[0][0])
        new_w += eta * grad_f(data, y, new_w, delta, lam)
    return new_w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
    history_fw = []
    new_w = np.copy(w)
    m = data.shape[0]
    idx = i
    for j in range(0, num_iter):
        if i == -1:
            idx = np.random.randint(0, m)
        history_fw.append(fun_f(data, y, new_w, delta, lam)[0][0])
        new_w += (eta / math.sqrt(j+1)) * grad_f(data, y, new_w, delta, lam, True, idx)
    return new_w, history_fw