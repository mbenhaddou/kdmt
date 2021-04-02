import numpy as np


def mcut_thresholding(dataset):
    num_samps = dataset.shape[0]
    sorted_dataset = np.sort(dataset)
    diff_scores = np.diff(sorted_dataset, axis=1)
    # Get the index of the maximum differences:
    maxdiff_1 = np.argmax(diff_scores, axis=1).reshape(1, -1).T
    maxdiff_2 = maxdiff_1 + 1
    row_index = np.array(range(num_samps)).reshape(1, -1).T
    mcut = (sorted_dataset[(row_index, maxdiff_1)] + sorted_dataset[(row_index, maxdiff_2)]) / 2
    mcut = np.reshape(mcut, mcut.shape[0])
    pred_labels = np.greater(dataset, mcut[:, None]).astype(int)
    return pred_labels


def prior_prob_per_cat(training_labels, num_cats):
    num_samples = training_labels.shape[0]
    priorprobs = np.zeros((num_cats, 1), dtype='float')
    for i in range(num_cats):
        num_curr_label = training_labels[:, i].sum()
        priorprobs[i - 1] = float(num_curr_label) / num_samples
    return priorprobs


def pcut_thresholding(scores, prior_probs, x):
    num_labels = scores.shape[1]
    set_length = scores.shape[0]
    pred_labels = np.zeros((set_length, num_labels))
    # Sort texts per category (low to high) and get the sorted indices:
    sorted_scores = scores.argsort(axis=0)
    for cat in range(num_labels):
        prior = prior_probs[cat]
        k = int(np.floor(prior * x * num_labels))
        # print k
        cat_scores = sorted_scores[:, cat]
        cat_scores = np.flipud(cat_scores)
        indices = cat_scores[:k]
        pred_labels[indices, cat] = 1
    return pred_labels


def rcut_thresholding(scores, threshold):
    num_labels = scores.shape[1]
    set_length = scores.shape[0]
    pred_labels = np.zeros(scores.shape, dtype='uint8')
    # Get the indices of the sorted array of texts:
    sort_index = np.argsort(scores, axis=1)
    # Reverse the order of the columns to get High to Low order:
    sort_index = np.fliplr(sort_index)
    labels = sort_index[:, :threshold]
    row_index = np.array(range(set_length)).reshape(1, -1).T
    pred_labels[(row_index, labels)] = 1
    return pred_labels


def rtcut_thresholding(scores, threshold):
    num_labels = scores.shape[1]
    set_length = scores.shape[0]
    pred_labels = np.zeros((set_length, num_labels), dtype='uint8')
    # Get the indices of the sorted array of texts:
    sort_index = np.argsort(scores, axis=1)
    # Reverse the order of the columns to get High to Low order:
    sort_index = np.fliplr(sort_index)
    labels = sort_index[:, :threshold]
    row_index = np.array(range(set_length)).reshape(1, -1).T
    pred_labels[(row_index, labels)] = 1
    return pred_labels


# scut_thresholding calculates the thresholds for each category
def scut_thresholding(scores, labels):
    num_labels = scores.shape[1]
    set_length = scores.shape[0]
    # print set_length
    # Calculate one threshold for each label:
    thresholds = np.zeros(num_labels)
    for lbl in range(num_labels):
        scores_lbl = scores[:, lbl]
        real_lbl = labels[:, lbl].astype('int')
        opt_th = 0.0
        min_mse = np.Inf
        for th in np.linspace(0, 1, 1000):
            pred_lbl = (scores_lbl >= th).astype('int')
            mse = (np.power(pred_lbl - real_lbl, 2).sum()).astype('float') / set_length
            # print "Threshold: %f - MSE: %f" % (th, mse)
            if mse < min_mse:
                min_mse = mse
                opt_th = th
        print("Label %d - Threshold: %f - Min MSE: %f" % (lbl + 1, opt_th, min_mse))
        thresholds[lbl] = opt_th
    return thresholds


def apply_thresholds(scores, thresholds):
    num_labels = scores.shape[1]
    set_length = scores.shape[0]
    pred_labels = np.zeros((set_length, num_labels), dtype='uint8')
    for lbl in range(num_labels):
        scores_lbl = scores[:, lbl]
        th = thresholds[lbl]
        pred_labels[:, lbl] = (scores_lbl >= th).astype('uint8')
    return pred_labels


