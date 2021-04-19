#!/usr/bin/env python

import numpy as np


def calculate_accuracy(real_labels, predicted_labels, average=True):
    num_samples = real_labels.shape[0]
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels), axis=1).astype('float')
    tp_fn_fp = np.sum(np.logical_or(real_labels, predicted_labels), axis=1).astype('float')
    acc_per_sample = true_positive / tp_fn_fp
    accuracy = np.sum(acc_per_sample) / num_samples
    if average:
        return accuracy
    else:
        return acc_per_sample


def calculate_precision(real_labels, predicted_labels, average=True):
    num_samples = real_labels.shape[0]
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels), axis=1).astype('float')
    num_real_labels = np.sum(real_labels, axis=1).astype('float')
    precision_per_sample = true_positive / num_real_labels
    precision = np.sum(precision_per_sample) / num_samples
    if average:
        return precision
    else:
        return precision_per_sample


def calculate_recall(real_labels, predicted_labels, average=True):
    num_samples = real_labels.shape[0]
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels), axis=1).astype('float')
    num_predicted_labels = np.sum(predicted_labels, axis=1).astype('float')
    num_predicted_labels[num_predicted_labels == 0] = np.nan
    recall_per_sample = true_positive / num_predicted_labels
    recall = np.nansum(recall_per_sample) / num_samples
    if average:
        return recall
    else:
        recall_per_sample[np.isnan(recall_per_sample)] = 0
        return recall_per_sample


def calculate_f1_score(real_labels, predicted_labels):
    num_samples = real_labels.shape[0]
    recall = calculate_recall(real_labels, predicted_labels, average=False)
    precision = calculate_precision(real_labels, predicted_labels, average=False)
    f1_num = 2 * recall * precision
    f1_den = recall + precision
    f1_den[f1_den == 0] = np.nan
    f1_score = f1_num / f1_den
    f1_score = np.nansum(f1_score) / num_samples
    return f1_score


# Label-based measures:
# Macro Precision, Recall and F1-Score
def calculate_macro_precision(real_labels, predicted_labels, average=True):
    num_classes = real_labels.shape[1]
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels), axis=0).astype('float')
    false_positive = np.sum(np.logical_and(np.logical_not(real_labels), predicted_labels), axis=0).astype('float')
    precision_den = true_positive + false_positive
    precision_den[precision_den == 0] = np.nan
    macro_precision_per_class = true_positive / precision_den
    macro_precision = np.nansum(macro_precision_per_class) / num_classes
    if average:
        return macro_precision
    else:
        macro_precision_per_class[np.isnan(macro_precision_per_class)] = 0
        return macro_precision_per_class


def calculate_macro_recall(real_labels, predicted_labels, average=True):
    num_classes = real_labels.shape[1]
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels), axis=0).astype('float')
    false_negative = np.sum(np.logical_and(real_labels, np.logical_not(predicted_labels)), axis=0).astype('float')
    recall_den = true_positive + false_negative
    recall_den[recall_den == 0] = np.nan
    macro_recall_per_class = true_positive / recall_den
    macro_recall = np.nansum(macro_recall_per_class) / num_classes
    if average:
        return macro_recall
    else:
        macro_recall_per_class[np.isnan(macro_recall_per_class)] = 0
        return macro_recall_per_class


def calculate_macro_f1_score(real_labels, predicted_labels, per_class=False):
    num_classes = real_labels.shape[1]
    recall = calculate_macro_recall(real_labels, predicted_labels, average=False)
    precision = calculate_macro_precision(real_labels, predicted_labels, average=False)
    f1_num = 2 * recall * precision
    f1_den = recall + precision
    f1_den[f1_den == 0] = np.nan
    f1_score = f1_num / f1_den
    if per_class:
        f1_score[np.isnan(f1_score)] = 0
        return f1_score
    else:
        f1_score = np.nansum(f1_score) / num_classes
        return f1_score


# Overall measures:
# Micro F1-Score, Precision and Recall
def calculate_micro_precision(real_labels, predicted_labels):
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels)).astype('float')
    false_positive = np.sum(np.logical_and(np.logical_not(real_labels), predicted_labels)).astype('float')
    precision_den = true_positive + false_positive
    if precision_den == 0:
        precision = 0
    else:
        precision = true_positive / precision_den
    return precision


def calculate_micro_recall(real_labels, predicted_labels):
    true_positive = np.sum(np.logical_and(real_labels, predicted_labels)).astype('float')
    false_negative = np.sum(np.logical_and(real_labels, np.logical_not(predicted_labels))).astype('float')
    recall_den = true_positive + false_negative
    if recall_den == 0:
        recall = 0
    else:
        recall = true_positive / recall_den
    return recall


def calculate_micro_f1_score(real_labels, predicted_labels):
    precision = calculate_micro_precision(real_labels, predicted_labels)
    recall = calculate_micro_recall(real_labels, predicted_labels)
    f1_num = 2 * precision * recall
    f1_den = precision + recall
    if f1_den == 0:
        f1_score = 0
    else:
        f1_score = f1_num / f1_den
    return f1_score


def calculate_micro_f1_score_from_precision_recall(precision, recall):
    f1_num = 2 * precision * recall
    f1_den = precision + recall
    if f1_den == 0:
        f1_score = 0
    else:
        f1_score = f1_num / f1_den
    return f1_score


def calculate_class_label_statistics(y):
    unique, counts = np.unique(y, return_counts=True)
    class_stats = dict(zip(unique, counts))

    return sorted(class_stats.items(), key=lambda x: -x[1])


def calculate_false_pos(y_true, y_pred):
    """
    Metric for Keras that *estimates* false positives while training
    This will not be completely accurate because it weights batches
    equally
    """
    from tensorflow.keras import backend as K
    return K.sum(K.cast(y_pred * (1 - y_true) > 0.5, 'float')) / K.maximum(1.0, K.sum(1 - y_true))


def calculate_false_neg(y_true, y_pred):
    """
    Metric for Keras that *estimates* false negatives while training
    This will not be completely accurate because it weights batches
    equally
    """
    from tensorflow.keras import backend as K
    return K.sum(K.cast((1 - y_pred) * (0 + y_true) > 0.5, 'float')) / K.maximum(1.0, K.sum(0 + y_true))

def calculate_hamming_score(true_labels, predicts):
    """ Computes de hamming score, which is known as the label-based accuracy,
    designed for multi-label problems. It's defined as the number of correctly
    predicted y_values divided by all classified y_values.
    """
    if not hasattr(true_labels, 'shape'):
        true_labels = np.asarray(true_labels)
    if not hasattr(predicts, 'shape'):
        predicts = np.asarray(predicts)
    N, L = true_labels.shape
    return np.sum((true_labels == predicts) * 1.) / N / L


def calculate_j_index(true_labels, predicts):
    """ Computes the Jaccard Index of the given set, which is also called the
    'intersection over union' in multi-label settings. It's defined as the
    intersection between the true label's set and the prediction's set,
    divided by the sum, or union, of those two sets.
    Parameters
    ----------
    true_labels: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the true y_values for all the classification tasks and for
        n_samples.
    predicts: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the predictions for all the classification tasks and for
        n_samples.
    Returns
    -------
    float
        The J-index, or 'intersection over union', for the given sets.
    """
    if not hasattr(true_labels, 'shape'):
        true_labels = np.asarray(true_labels)
    if not hasattr(predicts, 'shape'):
        predicts = np.asarray(predicts)
    N, L = true_labels.shape
    s = 0.0
    for i in range(N):
        inter = sum((true_labels[i, :] * predicts[i, :]) > 0) * 1.
        union = sum((true_labels[i, :] + predicts[i, :]) > 0) * 1.
        if union > 0:
            s += inter / union
        elif np.sum(true_labels[i, :]) == 0:
            s += 1.
    return s * 1. / N


def calculate_exact_match(true_labels, predicts):
    """ This is the most strict metric for the multi label setting. It's defined
    as the percentage of samples that have all their y_values correctly classified.
    Parameters
    ----------
    true_labels: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the true y_values for all the classification tasks and for
        n_samples.
    predicts: numpy.ndarray of shape (n_samples, n_target_tasks)
        A matrix with the predictions for all the classification tasks and for
        n_samples.
    Returns
    -------
    float
        The exact match percentage between the given sets.
    """
    if not hasattr(true_labels, 'shape'):
        true_labels = np.asarray(true_labels)
    if not hasattr(predicts, 'shape'):
        predicts = np.asarray(predicts)
    N, L = true_labels.shape
    return np.sum(np.sum((true_labels == predicts) * 1, axis=1) == L) * 1. / N
