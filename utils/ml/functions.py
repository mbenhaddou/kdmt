"""
Mathematical functions used to customize
computation in various places
"""
from math import exp, sqrt, pi
from typing import *

import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss

LOSS_BIAS = 0.9  # [0..1] where 1 is inf bias



def pad_sentences(
        sequences: np.ndarray, max_length: int = None, padding_value: int = 0, padding_style="post"
) -> np.ndarray:
    """
    Pad input sequences up to max_length
    values are aligned to the right

    Args:
        sequences (iter): a 2D matrix (np.array) to pad
        max_length (int, optional): max length of resulting sequences
        padding_value (int, optional): padding value
        padding_style (str, optional): add padding values as prefix (use with 'pre')
            or postfix (use with 'post')

    Returns:
        input sequences padded to size 'max_length'
    """
    if isinstance(sequences, list) and len(sequences) > 0:
        try:
            sequences = np.asarray(sequences)
        except ValueError:
            print("cannot convert sequences into numpy array")
    assert hasattr(sequences, "shape")
    if len(sequences) < 1:
        return sequences
    if max_length is None:
        max_length = np.max([len(s) for s in sequences])
    elif max_length < 1:
        raise ValueError("max sequence length must be > 0")
    if max_length < 1:
        return sequences
    padded_sequences = np.ones((len(sequences), max_length), dtype=np.int32) * padding_value
    for i, sent in enumerate(sequences):
        if padding_style == "post":
            trunc = sent[-max_length:]
            padded_sequences[i, : len(trunc)] = trunc
        elif padding_style == "pre":
            trunc = sent[:max_length]
            padded_sequences[i, -trunc:] = trunc
    return padded_sequences.astype(dtype=np.int32)


def one_hot(mat: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert a 1D matrix of ints into one-hot encoded vectors.

    Arguments:
        mat (numpy.ndarray): A 1D matrix of labels (int)
        num_classes (int): Number of all possible classes

    Returns:
        numpy.ndarray: A 2D matrix
    """
    assert len(mat.shape) < 2 or isinstance(mat.shape, int)
    vec = np.zeros((mat.shape[0], num_classes))
    for i, v in enumerate(mat):
        vec[i][v] = 1.0
    return vec


def one_hot_sentence(mat: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert a 2D matrix of ints into one-hot encoded 3D matrix

    Arguments:
        mat (numpy.ndarray): A 2D matrix of labels (int)
        num_classes (int): Number of all possible classes

    Returns:
        numpy.ndarray: A 3D matrix
    """
    new_mat = []
    for i in range(mat.shape[0]):
        new_mat.append(one_hot(mat[i], num_classes))
    return np.asarray(new_mat)


def class_label_statistics(y):
    unique, counts = np.unique(y, return_counts=True)
    class_stats = dict(zip(unique, counts))

    return sorted(class_stats.items(), key=lambda x: -x[1])


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
