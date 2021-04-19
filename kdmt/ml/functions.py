"""
Mathematical functions used to customize
computation in various places
"""
from math import exp, sqrt, pi
from typing import *

import numpy as np

def sigmoid(x):
    """Sigmoid squashing function for scalars"""
    return 1 / (1 + exp(-x))


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=1)

def inv_sigmoid(x):
    """Inverse sigmoid (logit) for scalars"""
    return -np.log(1 / x - 1)


def inv_softmax(x):
    """Inverse softmax (logit) for scalars"""

    if np.any(x < 0) or np.any(x > 1):
        raise ValueError('probability values have to be between [0, 1]')
    eta = np.log(x)
    iie = np.isinf(eta)
    if np.any(iie):
        eta[~iie] = eta[~iie] - eta[~iie].mean()
        return eta
    return eta - np.nanmean(eta, axis=1)[:, None]

