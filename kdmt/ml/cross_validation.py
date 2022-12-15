import numbers
from sklearn.model_selection._validation import _score, _check_is_permutation, _enforce_prediction_order, _normalize_score_results
import numpy as np
import scipy.sparse as sp
from joblib import Parallel

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.utils.validation import _check_fit_params
from sklearn.utils.validation import _num_samples
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._split import check_cv
from sklearn.preprocessing import LabelEncoder


def cross_val_predict_score(
    estimator,
    X,
    y=None,
    *,
    scoring=None,
    groups=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    method="predict",
    error_score=np.nan,
):

    if not hasattr(estimator, 'predict_proba'):
        method = "predict"

    """Generate cross-validated estimates for each input data point.

    The data is split according to the cv parameter. Each sample belongs
    to exactly one test set, and its prediction is computed with an
    estimator fitted on the corresponding training set.

    Passing these predictions into an evaluation metric may not be a valid
    way to measure generalization performance. Results can differ from
    :func:`cross_validate` and :func:`cross_val_score` unless all tests sets
    have equal size and the metric decomposes over samples.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and
        predicting are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    method : {'predict', 'predict_proba', 'predict_log_proba', \
              'decision_function'}, default='predict'
        The method to be invoked by `estimator`.

    Returns
    -------
    predictions : ndarray
        This is the result of calling `method`. Shape:

            - When `method` is 'predict' and in special case where `method` is
              'decision_function' and the target is binary: (n_samples,)
            - When `method` is one of {'predict_proba', 'predict_log_proba',
              'decision_function'} (unless special case above):
              (n_samples, n_classes)
            - If `estimator` is :term:`multioutput`, an extra dimension
              'n_outputs' is added to the end of each shape above.

    See Also
    --------
    cross_val_score : Calculate score for each CV split.
    cross_validate : Calculate one or more scores and timings for each CV
        split.

    Notes
    -----
    In the case that one or more classes are absent in a training portion, a
    default score needs to be assigned to all instances for that class if
    ``method`` produces columns per class, as in {'decision_function',
    'predict_proba', 'predict_log_proba'}.  For ``predict_proba`` this value is
    0.  In order to ensure finite output, we approximate negative infinity by
    the minimum finite float value for the dtype in other cases.

    Examples
    --------

    """
    X, y, groups = indexable(X, y, groups)
    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    splits = list(cv.split(X, y, groups))

    test_indices = np.concatenate([test for _, test in splits])
    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError("cross_val_predict only works for partitions")

    # If classification methods produce multiple columns of output,
    # we need to manually encode classes to ensure consistent column ordering.
    encode = (
        method in ["decision_function", "predict_proba", "predict_log_proba"]
        and y is not None
    )
    if encode:
        y = np.asarray(y)
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    predictions_and_scores = parallel(
        delayed(_fit_score_and_predict)(
            clone(estimator), X, y, train, test, verbose, fit_params, method, scorers, error_score
        )
        for train, test in splits
    )
    scores=[pred[1] for pred in predictions_and_scores]
    predictions=[pred[0] for pred in predictions_and_scores]

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        # `predictions` is a list of method outputs from each fold.
        # If each of those is also a list, then treat this as a
        # multioutput-multiclass task. We need to separately concatenate
        # the method outputs for each label into an `n_labels` long list.
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)

    test_scores_dict = _normalize_score_results(scores)

    ret = {}
    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]



    if method=='predict_proba':
        predictions =np.column_stack((np.argmax(predictions, axis=-1), np.max(predictions, axis=-1)))
    else:
        predictions=np.array(predictions)

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions], ret
    else:
        return predictions[inv_test_indices], ret


def _fit_score_and_predict(estimator, X, y, train, test, verbose, fit_params, method, scorer, error_score):
    """Fit estimator and predict values for a given dataset split.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

        .. versionchanged:: 0.20
            X is only required to be an object with finite length or shape now

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    verbose : int
        The verbosity level.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    method : str
        Invokes the passed method name of the passed estimator.

    Returns
    -------
    predictions : sequence
        Result of calling 'estimator.method'
    """

    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    func = getattr(estimator, method)
    predictions = func(X_test)
    test_scores = _score(estimator, X_test, y_test, scorer, error_score)

    encode = (
        method in ["decision_function", "predict_proba", "predict_log_proba"]
        and y is not None
    )

    if encode:
        if isinstance(predictions, list):
            predictions = [
                _enforce_prediction_order(
                    estimator.classes_[i_label],
                    predictions[i_label],
                    n_classes=len(set(y[:, i_label])),
                    method=method,
                )
                for i_label in range(len(predictions))
            ]
        else:
            # A 2D y array should be a binary label indicator matrix
            n_classes = len(set(y)) if y.ndim == 1 else y.shape[1]
            predictions = _enforce_prediction_order(
                estimator.classes_, predictions, n_classes, method
            )

    return (predictions, test_scores)



