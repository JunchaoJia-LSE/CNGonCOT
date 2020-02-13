import numpy as np


def logistic(val):
    return np.exp(val) / np.sum(np.exp(val))


def random_on_simplex(dim):
    val = np.random.normal(size=dim)
    return logistic(val)
