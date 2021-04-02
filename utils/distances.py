import copy

import numpy as np


def hamming_distance(x, y, weights=None):
    """
    calculate hamming distance
    :param x: a sample
    :param y: another sample
    :param weights: the weights for each feature
    :return: hamming distance
    """
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    distance = np.sum(((1 - np.sign(x * y)) / 2) * weights)
    return distance

def euclidean_distance(x, y, weights=None):
    """
    calulate euclidean distance
    :param x: a sample
    :param y: another sample
    :param weights: the weights for each feature
    :return: euclidean distance
    """
    assert len(x) == len(y)
    if weights is None:
        weights = np.ones(len(x))
    temp_x = copy.deepcopy(x)
    temp_y = copy.deepcopy(y)
    distance = np.sqrt(np.sum(np.power(temp_x - temp_y, 2) * weights))
    return distance

def braycurtis_distance( a, b):
    return np.sum(np.fabs(a - b)) / np.sum(np.fabs(a + b))

def canberra_distance( a, b):
    return np.sum(np.fabs(a - b) / (np.fabs(a) + np.fabs(b)))

def chebyshev_distance( a, b):
    return np.amax(a - b)

def cityblock_distance( a, b):
    return manhattan_distance(a, b)

def correlation_distance( a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    return 1.0 - np.mean(a * b) / np.sqrt(np.mean(np.square(a)) * np.mean(np.square(b)))

def cosine_distance( a, b):
    return 1 - np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))

def dice_distance( a, b):
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))

def jaccard_distance( u, v):
    return np.double(np.bitwise_and((u != v), np.bitwise_or(u != 0, v != 0)).sum()) / np.double(np.bitwise_or(u != 0, v != 0).sum())

def kulsinski_distance( a, b):
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return (ntf + nft - ntt + len(a)) / (ntf + nft + len(a))

def mahalanobis_distance( a, b, vi):
    return np.sqrt(np.dot(np.dot((a - b), vi),(a - b).T))

def manhattan_distance( a, b):
    return np.sum(np.fabs(a - b))

def matching_distance( a, b):
    return hamming_distance(a, b)

def minkowski_distance( a, b, p):
    return np.power(np.sum(np.power(np.fabs(a - b), p)), 1 / p)

def rogerstanimoto_distance( a, b):
    nff = ((1 - a) * (1 - b)).sum()
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return float(2.0 * (ntf + nft)) / float(ntt + nff + (2.0 * (ntf + nft)))

def russellrao_distance( a, b):
    return float(len(a) - (a * b).sum()) / len(a)

def seuclidean_distance( a, b, V):
    return np.sqrt(np.sum((a - b) ** 2 / V))

def sokalmichener_distance( a, b):
    nff = ((1 - a) * (1 - b)).sum()
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return float(2.0 * (ntf + nft)) / float(ntt + nff + 2.0 * (ntf + nft))

def sokalsneath_distance( a, b):
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return float(2.0 * (ntf + nft)) / np.array(ntt + 2.0 * (ntf + nft))

def sqeuclidean_distance( a, b):
    return np.sum(np.dot((a - b), (a - b)))

def wminkowski_distance( a, b, p, w):
    return np.power(np.sum(np.power(np.fabs(w * (a - b)), p)), 1 / p)

def yule_distance( a, b):
    nff = ((1 - a) * (1 - b)).sum()
    nft = ((1 - a) * b).sum()
    ntf = (a * (1 - b)).sum()
    ntt = (a * b).sum()
    return float(2.0 * ntf * nft / np.array(ntt * nff + ntf * nft))