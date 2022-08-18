"""
Plotting functions for classifier models
"""
import numpy as np
from kdmt.ml.importance import feature_importances
from kdmt.ml.metrics.plot.matplotlib import bar
from kdmt.ml.metrics.classification import confusion_matrix as k_confusion_matrix
from kdmt.ml.metrics.classification import precision_at
from kdmt.lists import is_column_vector, is_row_vector
from kdmt.ml.metrics.plot.heatmap import default_heatmap
import matplotlib.pyplot as plt
import seaborn as sns


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


def make_confusion_matrix_plot(conf_mat,
                               group_names=None,
                               categories='auto',
                               count=True,
                               percent=True,
                               cbar=True,
                               xyticks=True,
                               xyplotlabels=True,
                               sum_stats=True,
                               figsize=None,
                               cmap='Blues',
                               title=None,
                               file_name=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    conf_mat:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(conf_mat.size)]

    if group_names and len(group_names) == conf_mat.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in conf_mat.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in conf_mat.flatten() / np.sum(conf_mat)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(conf_mat.shape[0], conf_mat.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(conf_mat) / float(np.sum(conf_mat))

        # if it is a binary confusion matrix, show some more stats
        if len(conf_mat) == 2:
            # Metrics for Binary Confusion Matrices
            precision = conf_mat[1, 1] / sum(conf_mat[:, 1])
            recall = conf_mat[1, 1] / sum(conf_mat[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(conf_mat, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    if file_name is not None:
        plt.savefig(file_name)


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
