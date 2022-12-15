import time
import timeit

import numpy as np
from kdmt import distances
def has_duplicate_col(matrix):
    """
    to checkout whether there are same cols in a matrix
    :param matrix: coding matrix
    :return: true or false
    """
    col_count = matrix.shape[1]
    for i in range(col_count):
        for j in range(i+1, col_count):
            if np.all([matrix[:, i] == matrix[:, j]]) or np.all([matrix[:, i] == -matrix[:, j]]):
                return True
    return False


def has_duplicate_row(matrix):
    """
    to checkout whether there are same rows in a matrix
    :param matrix: coding matrix
    :return: true or false
    """
    row_count = matrix.shape[0]
    for i in range(row_count):
        for j in range(i+1, row_count):
            if np.all([matrix[i] == matrix[j]]) or np.all([matrix[i] == -matrix[j]]):
                return True
    return False


def get_closet_vector(vector, matrix, distance="euclidean", weights=None):
    """
    find the closet coding vector in matrix
    :param vector: a predicted vector
    :param matrix: type ndarray, coding matrix
    :param distance: a callable object to calculate distance
    :param weights: the weights for each feature
    :return: the index corresponding to closet coding vector

    example:
    -------
    >>> from kdmt.matrix import get_closet_vector
    >>> import numpy as np
    >>> mat=np.array([[1, 2, 3],[4,5, 6],[7, 8, 9]])
    >>> print(get_closet_vector([4, 4, 6], mat))
    """
    if not callable(distance):
        distance=eval('distances.{}_distance'.format(distance))
    else:
        distance=distance

    d = np.inf
    index = None
    for i in range(matrix.shape[0]):
        if distance(vector, matrix[i], weights) < d:
            d = distance(vector, matrix[i], weights)
            index = i
    return index


def euclidean_distance(X, Y):
    """Efficiently calculates the euclidean distance
    between two vectors using Numpys einsum function.

    Parameters
    ----------
    X : array, (n_samples x d_dimensions)
    Y : array, (m_samples x d_dimensions)

    Returns
    -------
    D : array, (n_samples, n_samples)
    """
    XX = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
    YY = np.einsum('ij,ij->i', Y, Y)


    D_squared=XX + YY - 2 * np.dot(X, Y.T)
    zero_mask = np.less(D_squared, 0.0)
    D_squared[zero_mask] = 0.0
    return np.sqrt(D_squared)


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

if __name__=='__main__':
    R=np.random.rand(20, 20)

    matrices=split(R, 10, 10)

    print('x')