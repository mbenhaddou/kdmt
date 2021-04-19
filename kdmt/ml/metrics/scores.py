import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score as classification_f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score


def logloss(y_true, y_predicted, sample_weight=None):
    epsilon = 1e-6
    y_predicted = sp.maximum(epsilon, y_predicted)
    y_predicted = sp.minimum(1 - epsilon, y_predicted)
    ll = log_loss(y_true, y_predicted, sample_weight=sample_weight)
    return ll


def rmse(y_true, y_predicted, sample_weight=None):
    val = mean_squared_error(y_true, y_predicted, sample_weight=sample_weight)
    return np.sqrt(val) if val > 0 else -np.Inf


def rmsle(y_true, y_predicted, sample_weight=None):
    val = mean_squared_log_error(y_true, y_predicted, sample_weight=sample_weight)
    return np.sqrt(val) if val > 0 else -np.Inf


def negative_auc(y_true, y_predicted, sample_weight=None):
    val = roc_auc_score(y_true, y_predicted, sample_weight=sample_weight)
    return -1.0 * val


def negative_r2(y_true, y_predicted, sample_weight=None):
    val = r2_score(y_true, y_predicted, sample_weight=sample_weight)
    return -1.0 * val


def negative_f1(y_true, y_predicted, sample_weight=None):

    if isinstance(y_true, pd.DataFrame):
        y_true = np.array(y_true)
    if isinstance(y_predicted, pd.DataFrame):
        y_predicted = np.array(y_predicted)

    average = None
    if len(y_predicted.shape) == 1:
        y_predicted = (y_predicted > 0.5).astype(int)
        average = "binary"
    else:
        y_predicted = np.argmax(y_predicted, axis=1)
        average = "micro"

    val = f1_score(y_true, y_predicted, sample_weight=sample_weight, average=average)

    return -val


def negative_average_precision(y_true, y_predicted, sample_weight=None):

    if isinstance(y_true, pd.DataFrame):
        y_true = np.array(y_true)
    if isinstance(y_predicted, pd.DataFrame):
        y_predicted = np.array(y_predicted)

    val = average_precision_score(y_true, y_predicted, sample_weight=sample_weight)

    return -val


def negative_spearman(y_true, y_predicted, sample_weight=None):
    # sample weight is ignored
    c, _ = sp.stats.spearmanr(y_true, y_predicted)
    return -c


def negative_pearson(y_true, y_predicted, sample_weight=None):
    # sample weight is ignored
    if isinstance(y_true, pd.DataFrame):
        y_true = np.array(y_true).ravel()
    if isinstance(y_predicted, pd.DataFrame):
        y_predicted = np.array(y_predicted).ravel()
    return -np.corrcoef(y_true, y_predicted)[0, 1]

def simple_accuracy(preds, labels):
    """return simple accuracy
    """
    return (preds == labels).mean()


def accuracy(preds, labels):
    """return simple accuracy in expected dict format
    """
    acc = simple_accuracy(preds, labels)
    return {"acc": acc}


def acc_and_f1(preds, labels):
    """return accuracy and f1 score
    """
    acc = simple_accuracy(preds, labels)
    f1 = classification_f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    """get pearson and spearman correlation
    """
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }
