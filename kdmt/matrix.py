import numpy as np

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
