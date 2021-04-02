"""
Mathematical functions used to customize
computation in various places
"""
from math import exp, sqrt, pi
from typing import *

import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss

LOSS_BIAS = 0.9  # [0..1] where 1 is inf bias


def set_loss_bias(bias: float):
    """
    Changes the loss bias

    This allows customizing the acceptable tolerance between
    false negatives and false positives

    Near 1.0 reduces false positives
    Near 0.0 reduces false negatives
    """
    global LOSS_BIAS
    LOSS_BIAS = bias


def weighted_log_loss(y_true, y_pred) -> Any:
    """
    Binary crossentropy with a bias towards false negatives
    y_true: Target
    y_pred: Prediction
    """
    from tensorflow.keras import backend as K

    pos_loss = -(0 + y_true) * K.log(0 + y_pred + K.epsilon())
    neg_loss = -(1 - y_true) * K.log(1 - y_pred + K.epsilon())

    return LOSS_BIAS * K.mean(neg_loss) + (1. - LOSS_BIAS) * K.mean(pos_loss)


def log_loss(y_true, y_pred):
    return sklearn_log_loss(y_true, y_pred)


def weighted_mse_loss(y_true, y_pred) -> Any:
    """Standard mse loss with a weighting between false negatives and positives"""
    from tensorflow.keras import backend as K

    total = K.sum(K.ones_like(y_true))
    neg_loss = total * K.sum(K.square(y_pred * (1 - y_true))) / K.sum(1 - y_true)
    pos_loss = total * K.sum(K.square(1. - (y_pred * y_true))) / K.sum(y_true)

    return LOSS_BIAS * neg_loss + (1. - LOSS_BIAS) * pos_loss


def false_pos(y_true, y_pred) -> Any:
    """
    Metric for Keras that *estimates* false positives while training
    This will not be completely accurate because it weights batches
    equally
    """
    from tensorflow.keras import backend as K
    return K.sum(K.cast(y_pred * (1 - y_true) > 0.5, 'float')) / K.maximum(1.0, K.sum(1 - y_true))


def false_neg(y_true, y_pred) -> Any:
    """
    Metric for Keras that *estimates* false negatives while training
    This will not be completely accurate because it weights batches
    equally
    """
    from tensorflow.keras import backend as K
    return K.sum(K.cast((1 - y_pred) * (0 + y_true) > 0.5, 'float')) / K.maximum(1.0, K.sum(0 + y_true))


def load_keras() -> Any:
    """Imports Keras injecting custom functions to prevent exceptions"""
    from tensorflow import keras
    keras.losses.weighted_log_loss = weighted_log_loss
    keras.metrics.false_pos = false_pos
    keras.metrics.false_positives = false_pos
    keras.metrics.false_neg = false_neg
    return keras


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


def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    if std == 0:
        return 0
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))
