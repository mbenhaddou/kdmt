import numpy as np

from kdmt.ml.metrics.plot.util import set_default_ax
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import precision_score
from kdmt.lists import isiter
from kdmt.ml.metrics.plot import binarize
from kdmt.ml.metrics.plot import validate

@set_default_ax
def metrics_at_thresholds(fn, y_true, y_score, n_thresholds=10, start=0.0,
                          ax=None):
    """Plot metrics at increasing thresholds
    """
    th, m = compute_at_thresholds(fn, y_true, y_score, n_thresholds,
                                  start)

    ax.plot(th, np.array(m).T, '.--')
    ax.legend([fn_.__name__ for fn_ in fn])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric value')
    ax.grid()

    return ax


def compute_at_thresholds(fn, y_true, y_score, n_thresholds=10, start=0.0):
    """
    Given scores, binarize them at different thresholds, then compute
    metrics
    """
    if isiter(fn):
        (thresholds,
         Y_pred) = binarize.scores_at_thresholds(y_score,
                                                 n_thresholds=n_thresholds)
        metrics = [np.array([fn_(y_true, y_pred) for y_pred in Y_pred])
                   for fn_ in fn]
        return thresholds, metrics
    else:
        (thresholds,
         Y_pred) = binarize.scores_at_thresholds(y_score,
                                                 n_thresholds=n_thresholds)
        metrics = np.array([fn(y_true, y_pred) for y_pred in Y_pred])
        return thresholds, metrics


def confusion_matrix(y_true, y_pred, target_names, normalize=False):
    if any((val is None for val in (y_true, y_pred))):
        raise ValueError("y_true and y_pred are needed to plot confusion "
                         "matrix")

    # calculate how many names you expect
    values = set(list(y_true)).union(set(list(y_pred)))
    expected_len = len(values)
    if target_names is not None:
        len_target = len(target_names)
    if target_names is not None and (expected_len != len_target):
        raise ValueError(('Data cointains {} different values, but target'
                          ' names contains {} values.'.format(expected_len,
                                                              len(target_names)
                                                              )))

    # if the user didn't pass target_names, create generic ones
    if target_names is not None:
        values = list(values)
        values.sort()
        target_names = ['Class {}'.format(v) for v in values]

    cm = sklearn_confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def classification_report(y_true, y_pred, target_names, normalize=False):
    if any((val is None for val in (y_true, y_pred))):
        raise ValueError("y_true and y_pred are needed to plot confusion "
                         "matrix")

    # calculate how many names you expect
    values = set(y_true).union(set(y_pred))
    expected_len = len(values)
    if target_names is not None:
        len_target = len(target_names)
    if target_names is not None and (expected_len != len_target):
        raise ValueError(('Data cointains {} different values, but target'
                          ' names contains {} values.'.format(expected_len,
                                                              len(target_names)
                                                              )))

    # if the user didn't pass target_names, create generic ones
    if target_names is None:
        values = list(values)
        values.sort()
        target_names = values

    cr = sk_classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    return cr


@validate.argument_is_proportion('top_proportion')
def precision_at(y_true, y_score, top_proportion, ignore_nas=False):
    '''
    Calculates precision at a given proportion.
    Only supports binary classification.
    '''
    # Sort scores in descending order
    scores_sorted = np.sort(y_score)[::-1]

    # Based on the proportion, get the index to split the data
    # if value is negative, return 0
    cutoff_index = max(int(len(y_true) * top_proportion) - 1, 0)
    # Get the cutoff value
    cutoff_value = scores_sorted[cutoff_index]

    # Convert scores to binary, by comparing them with the cutoff value
    scores_binary = np.array([int(y >= cutoff_value) for y in y_score])
    # Calculate precision using sklearn function
    if ignore_nas:
        precision = __precision(y_true, scores_binary)
    else:
        precision = precision_score(y_true, scores_binary)

    return precision, cutoff_value


def __precision(y_true, y_pred):
    '''
        Precision metric tolerant to unlabeled data in y_true,
        NA values are ignored for the precision calculation
    '''
    # make copies of the arrays to avoid modifying the original ones
    y_true = np.copy(y_true)
    y_pred = np.copy(y_pred)

    # precision = tp/(tp+fp)
    # True nehatives do not affect precision value, so for every missing
    # value in y_true, replace it with 0 and also replace the value
    # in y_pred with 0
    is_nan = np.isnan(y_true)
    y_true[is_nan] = 0
    y_pred[is_nan] = 0
    precision = precision_score(y_true, y_pred)
    return precision


@validate.argument_is_proportion('top_proportion')
def tp_at(y_true, y_score, top_proportion):
    y_pred = binarize.scores_at_top_proportion(y_score, top_proportion)
    tp = (y_pred == 1) & (y_true == 1)
    return tp.sum()


@validate.argument_is_proportion('top_proportion')
def fp_at(y_true, y_score, top_proportion):
    y_pred = binarize.scores_at_top_proportion(y_score, top_proportion)
    fp = (y_pred == 1) & (y_true == 0)
    return fp.sum()


@validate.argument_is_proportion('top_proportion')
def tn_at(y_true, y_score, top_proportion):
    y_pred = binarize.scores_at_top_proportion(y_score, top_proportion)
    tn = (y_pred == 0) & (y_true == 0)
    return tn.sum()


@validate.argument_is_proportion('top_proportion')
def fn_at(y_true, y_score, top_proportion):
    y_pred = binarize.scores_at_top_proportion(y_score, top_proportion)
    fn = (y_pred == 0) & (y_true == 1)
    return fn.sum()


@validate.argument_is_proportion('top_proportion')
def labels_at(y_true, y_score, top_proportion, normalize=False):
    '''
        Return the number of labels encountered in the top  Y proportion
    '''
    # Get indexes of scores sorted in descending order
    indexes = np.argsort(y_score)[::-1]

    # Sort true values in the same order
    y_true_sorted = y_true[indexes]

    # Grab top x proportion of true values
    cutoff_index = max(int(len(y_true_sorted) * top_proportion) - 1, 0)
    # add one to index to grab values including that index
    y_true_top = y_true_sorted[:cutoff_index + 1]

    # Count the number of non-nas in the top x proportion
    # we are returning a count so it should be an int
    values = int((~np.isnan(y_true_top)).sum())

    if normalize:
        values = float(values) / (~np.isnan(y_true)).sum()

    return values
