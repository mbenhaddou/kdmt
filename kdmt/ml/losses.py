import numpy as np

# Sample-based measures:
# Hamming Loss, Accuracy, Precision, Recall, F1-Score:
def calculate_hamming_loss(real_labels, predicted_labels):
    num_samples, num_classes = real_labels.shape
    xor_arrays = np.logical_xor(real_labels, predicted_labels).astype('float')
    sum_rows = np.sum(xor_arrays, axis=1) / num_classes
    hamming_loss = sum_rows.sum() / num_samples
    return hamming_loss


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


def weighted_log_loss(y_true, y_pred):
    """
    Binary crossentropy with a bias towards false negatives
    y_true: Target
    y_pred: Prediction
    """
    from tensorflow.keras import backend as K

    pos_loss = -(0 + y_true) * K.log(0 + y_pred + K.epsilon())
    neg_loss = -(1 - y_true) * K.log(1 - y_pred + K.epsilon())

    return LOSS_BIAS * K.mean(neg_loss) + (1. - LOSS_BIAS) * K.mean(pos_loss)

def weighted_mse_loss(y_true, y_pred) :
    """Standard mse loss with a weighting between false negatives and positives"""
    from tensorflow.keras import backend as K

    total = K.sum(K.ones_like(y_true))
    neg_loss = total * K.sum(K.square(y_pred * (1 - y_true))) / K.sum(1 - y_true)
    pos_loss = total * K.sum(K.square(1. - (y_pred * y_true))) / K.sum(y_true)

    return LOSS_BIAS * neg_loss + (1. - LOSS_BIAS) * pos_loss
