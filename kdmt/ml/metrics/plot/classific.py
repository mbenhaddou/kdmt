"""
Plotting functions for classifier models
"""
import matplotlib.pyplot as plt
import numpy as np
from kdmt.ml.importance import feature_importances
from kdmt.ml.metrics.plot.matplotlib import bar
from kdmt.ml.metrics.classification import confusion_matrix as k_confusion_matrix
from kdmt.ml.metrics.classification import precision_at
from kdmt.lists import is_column_vector, is_row_vector
from kdmt.ml.metrics.plot.heatmap import default_heatmap


def confusion_matrix_ax(y_true, y_pred, target_names=None, normalize=False,
                        cmap=None, ax=None):
    """
    Plot confustion matrix.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Correct target values (ground truth).
    y_pred : array-like, shape = [n_samples]
        Target predicted classes (estimator predictions).
    target_names : list
        List containing the names of the target classes. List must be in order
        e.g. ``['Label for class 0', 'Label for class 1']``. If ``None``,
        generic labels will be generated e.g. ``['Class 0', 'Class 1']``
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes
    normalize : bool
        Normalize the confusion matrix
    cmap : matplotlib Colormap
        If ``None`` uses a modified version of matplotlib's OrRd colormap.

    Notes
    -----
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html


    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/conf_mat.py

    """

    if target_names is None:
        target_names = set(y_true)
    cm = k_confusion_matrix(y_true, y_pred, target_names=target_names, normalize=normalize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    np.set_printoptions(precision=2)

    if ax is None:
        ax = plt.gca()

    # this (y, x) may sound counterintuitive. The reason is that
    # in a matrix cell (i, j) is in row=i and col=j, translating that
    # to an x, y plane (which matplotlib uses to plot), we need to use
    # i as the y coordinate (how many steps down) and j as the x coordinate
    # how many steps to the right.
    for (y, x), v in np.ndenumerate(cm):
        try:
            label = '{:.2}'.format(v)
        except:
            label = v
        ax.text(x, y, label, horizontalalignment='center',
                verticalalignment='center')

    if cmap is None:
        cmap = default_heatmap()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(target_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(target_names)

    title = 'Confusion matrix'
    if normalize:
        title += ' (normalized)'
    ax.set_title(title)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return ax


# Receiver operating characteristic (ROC) with cross validation
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#example-model-selection-plot-roc-crossval-py


# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def feature_importances_ax(data, top_n=None, feature_names=None,
                        orientation='horizontal', ax=None):
    """
    Get and order feature importances from a scikit-learn model
    or from an array-like structure. If data is a scikit-learn model with
    sub-estimators (e.g. RandomForest, AdaBoost) the function will compute the
    standard deviation of each feature.

    Parameters
    ----------
    data : sklearn model or array-like structure
        Object to get the data from.
    top_n : int
        Only get results for the top_n features.
    feature_names : array-like
        Feature names
    orientation: ('horizontal', 'vertical')
        Bar plot orientation
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------
    .. plot:: ../../examples/feature_importances.py

    """
    if data is None:
        raise ValueError('data is needed to plot feature importances. '
                         'When plotting using the evaluator you need to pass '
                         'an estimator ')

    # If no feature_names is provided, assign numbers
    res = feature_importances(data, top_n, feature_names)

    ax = bar.plot(res.importance, orientation, res.feature_name,
                  error=None if not hasattr(res, 'std_') else res.std_)
    ax.set_title("Feature importances")
    return ax


def precision_at_proportions(y_true, y_score, ax=None):
    """
    Plot precision values at different proportions.

    Parameters
    ----------
    y_true : array-like
        Correct target values (ground truth).
    y_score : array-like
        Target scores (estimator predictions).
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    if any((val is None for val in (y_true, y_score))):
        raise ValueError('y_true and y_score are needed to plot precision at '
                         'proportions')

    if ax is None:
        ax = plt.gca()

    y_score_is_vector = is_column_vector(y_score) or is_row_vector(y_score)
    if not y_score_is_vector:
        y_score = y_score[:, 1]

    # Calculate points
    proportions = [0.01 * i for i in range(1, 101)]
    precs_and_cutoffs = [precision_at(y_true, y_score, p) for p in proportions]
    precs, cutoffs = zip(*precs_and_cutoffs)

    # Plot and set nice defaults for title and axis labels
    ax.plot(proportions, precs)
    ax.set_title('Precision at various proportions')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Proportion')
    ticks = [0.1 * i for i in range(1, 11)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_ylim([0, 1.0])
    ax.set_xlim([0, 1.0])
    return ax
