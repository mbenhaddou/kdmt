
class ConfusionMatrix():
    """ ConfusionMatrix
    This structure constitutes a confusion matrix, or an error matrix. It is
    represented by a matrix of shape (n_labels, n_labels), in a simple, one
    classification task context.
    """

    def __init__(self, n_targets=None, dtype=np.int64):
        super().__init__()
        if n_targets is not None:
            self.n_targets = n_targets
        else:
            self.n_targets = 0
        self.sample_count = 0
        self.dtype = dtype

        self.confusion_matrix = np.zeros((self.n_targets, self.n_targets), dtype=dtype)

    def restart(self, n_targets):
        if n_targets is None:
            self.n_targets = 0
        else:
            self.n_targets = n_targets
        self.confusion_matrix = np.zeros((self.n_targets, self.n_targets))
        self.sample_count = 0
        pass

    def _update(self, i, j):
        self.confusion_matrix[i, j] += 1
        self.sample_count += 1
        return True

    def update(self, i=None, j=None):
        """ update
        Increases by one the count of occurrences in one of the ConfusionMatrix's
        cells.
        Parameters
        ---------
        i: int
            The index of the row to be updated.
        j: int
            The index of the column to be updated.
        Returns
        -------
        bool
            True if the update was successful and False if it was unsuccessful,
            case in which a index is out of range.
        Notes
        -----
        No IndexError or IndexOutOfRange errors raised.
        """
        if i is None or j is None:
            return False

        else:
            m, n = self.confusion_matrix.shape
            if (i <= m) and (i >= 0) and (j <= n) and (j >= 0):
                return self._update(i, j)

            else:
                max_value = np.max(i, j)
                if max_value > m + 1:
                    return False

                else:
                    self.reshape(max_value, max_value)
                    return self._update(i, j)

    def remove(self, i=None, j=None):
        """ remove
        Decreases by one the count of occurrences in one of the ConfusionMatrix's
        cells.
        Parameters
        ----------
        i: int
            The index of the row to be updated.
        j: int
            The index of the column to be updated.
        Returns
        -------
        bool
            True if the removal was successful and False otherwise.
        Notes
        -----
        No IndexError or IndexOutOfRange errors raised.
        """
        if i is None or j is None:
            return False

        m, n = self.confusion_matrix.shape
        if (i <= m) and (i >= 0) and (j <= n) and (j >= 0):
            return self._remove(i, j)

        else:
            return False

    def _remove(self, i, j):
        self.confusion_matrix[i, j] = self.confusion_matrix[i, j] - 1
        self.sample_count -= 1
        return True

    def reshape(self, m, n):
        i, j = self.confusion_matrix.shape

        if (m != n) or (m < i) or (n < j):
            return False
        aux = self.confusion_matrix.copy()
        self.confusion_matrix = np.zeros((m, n), self.dtype)

        for p in range(i):
            for q in range(j):
                self.confusion_matrix[p, q] = aux[p, q]

        return True

    def shape(self):
        """ shape
        Returns
        -------
        tuple
            The confusion matrix's shape.
        """
        return self.confusion_matrix.shape

    def value_at(self, i, j):
        """ value_at
        Parameters
        ----------
        i: int
            An index from one of the matrix's rows.
        j: int
            An index from one of the matrix's columns.
        Returns
        -------
        int
            The current occurrence count at position [i, j].
        """
        return self.confusion_matrix[i, j]

    def row(self, r):
        """ row
        Parameters
        ----------
        r: int
            An index from one of the matrix' rows.
        Returns
        -------
        numpy.array
            The complete row indexed by r.
        """
        return self.confusion_matrix[r: r + 1, :]

    def column(self, c):
        """ column
        Parameters
        ----------
        c: int
            An index from one of the matrix' columns.
        Returns
        -------
        numpy.array
            The complete column indexed by c.
        """
        return self.confusion_matrix[:, c: c + 1]

    def get_sum_main_diagonal(self):
        """ Computes the sum of occurrences in the main diagonal.
        Returns
        -------
        int
            The occurrence count in the main diagonal.
        """
        m, n = self.confusion_matrix.shape
        sum_main_diagonal = 0
        for i in range(m):
            sum_main_diagonal += self.confusion_matrix[i, i]
        return sum_main_diagonal

    @property
    def matrix(self):
        if self.confusion_matrix is not None:
            return self.confusion_matrix
        else:
            return None

    @property
    def _sample_count(self):
        return self.sample_count

    def get_info(self):
        return 'ConfusionMatrix: n_targets: ' + str(self.n_targets) + \
               ' - sample_count: ' + str(self.sample_count) + \
               ' - dtype: ' + str(self.dtype)

    def get_class_type(self):
        return 'collection'


class MOLConfusionMatrix():
    """
    This structure represents a confusion matrix, or an error matrix. It is
    represented by a matrix of shape (n_targets, n_labels, n_labels). It
    basically works as an individual ConfusionMatrix for each of the
    classification tasks in a multi label environment. Thus, n_labels is
    always 2 (binary).

    The first dimension defines which classification task it keeps track of.
    The second dimension is associated with the true y_values, while the other
    is associated with the predictions. For example, an entry in position
    [2, 1, 2] represents a miss classification in the classification task of
    index 2, where the true label was index 1, but the prediction was index 2.

    This structure is used to keep updated statistics from a multi output
    classifier's performance, which allows to compute different evaluation
    metrics.

    """

    def __init__(self, n_targets=None, dtype=np.int64):
        super().__init__()
        if n_targets is not None:
            self.n_targets = n_targets
        else:
            self.n_targets = 0
        self.dtype = dtype
        self.confusion_matrix = np.zeros((self.n_targets, 2, 2), dtype=dtype)

    def restart(self, n_targets):
        if n_targets is None:
            self.n_targets = 0
        else:
            self.n_targets = n_targets
        self.confusion_matrix = np.zeros((self.n_targets, 2, 2), dtype=self.dtype)
        pass

    def _update(self, target, true, pred):
        self.confusion_matrix[int(target), int(true), int(pred)] += 1
        return True

    def update(self, target=None, true=None, pred=None):
        """ update

        Increases by one the occurrence count in one of the matrix's positions.
        As entries arrive, it may reshape the matrix to correctly accommodate all
        possible y_values.

        The count will be increased in the matrix's [label, true, pred] position.

        Parameters
        ----------
        target: int
            A classification task's index.

        true: int
            A true label's index.

        pred: int
            A prediction's index

        Returns
        -------
        bool
            True if the update was successful, False otherwise.

        """

        if target is None or true is None or pred is None:
            return False
        else:
            m, n, p = self.confusion_matrix.shape
            if (target < m) and (target >= 0) and (true < n) and (true >= 0) and (pred < p) and (pred >= 0):
                return self._update(target, true, pred)
            else:
                try:
                    if (true > 1) or (true < 0) or (pred > 1) or (pred < 0):
                        return False
                    if target > m:
                        return False
                    else:
                        self.reshape(target + 1, 2, 2)
                        return self._update(target, true, pred)
                except Exception as e:
                    print(e)

    def remove(self, target=None, true=None, pred=None):
        """ remove

        Decreases by one the occurrence count in one of the matrix's positions.

        The count will be increased in the matrix's [label, true, pred] position.

        Parameters
        ----------
        target: int
            A classification task's index.

        true: int
            A true label's index.

        pred: int
            A prediction's index

        Returns
        -------
        bool
            True if the removal was successful, False otherwise.

        """
        if true is None or pred is None or target is None:
            return False
        m, n, p = self.confusion_matrix.shape
        if (target <= m) and (target >= 0) and (true <= n) and (true >= 0) and (pred >= 0) and (pred <= p):
            return self._remove(target, true, pred)
        else:
            return False

    def _remove(self, target, true, pred):
        self.confusion_matrix[target, true, pred] = self.confusion_matrix[target, true, pred] - 1
        return True

    def reshape(self, target, m, n):
        t, i, j = self.confusion_matrix.shape
        if (target > t + 1) or (m != n) or (m != 2) or (m < i) or (n < j):
            return False
        aux = self.confusion_matrix.copy()
        self.confusion_matrix = np.zeros((target, m, n), self.dtype)
        for w in range(t):
            for p in range(i):
                for q in range(j):
                    self.confusion_matrix[w, p, q] = aux[w, p, q]
        return True

    def shape(self):
        return self.confusion_matrix.shape

    def value_at(self, target, i, j):
        """ value_at

        Parameters
        ----------
        target: int
            An index from one of classification's tasks.

        i: int
            An index from one of the matrix's rows.

        j: int
            An index from one of the matrix's columns.

        Returns
        -------
        int
            The current occurrence count at position [label, i, j].

        """
        return self.confusion_matrix[target, i, j]

    def row(self, r):
        """ row

        Parameters
        ----------
        r: int
            An index from one of the matrix' rows.

        Returns
        -------
        numpy.array
            The complete row indexed by r.

        """
        return self.confusion_matrix[r:r + 1, :]

    def column(self, c):
        """ column

        Parameters
        ----------
        c: int
            An index from one of the matrix' columns.

        Returns
        -------
        numpy.array
            The complete column indexed by c.

        """
        return self.confusion_matrix[:, c:c + 1]

    def target(self, t):
        """ label

        Parameters
        ----------
        t: int
            An index from one of the matrix' label.

        Returns
        -------
        numpy.ndarray
            The complete label indexed by t.

        """
        return self.confusion_matrix[t, :, :]

    def get_sum_main_diagonal(self):
        """ get_sum_main_diagonal

        Computes the sum of occurrences in all the main diagonals.

        Returns
        -------
        int
            The occurrence count in the main diagonals.

        """
        t, m, n = self.confusion_matrix.shape
        sum_main_diagonal = 0
        for i in range(t):
            sum_main_diagonal += self.confusion_matrix[i, 0, 0]
            sum_main_diagonal += self.confusion_matrix[i, 1, 1]
        return sum_main_diagonal

    def get_total_sum(self):
        """ get_total_sum

        Returns
        ------
        int
            The sum of occurrences in the matrix.

        """
        return np.sum(self.confusion_matrix)

    def get_total_discordance(self):
        """ get_total_discordance

        The total discordance is defined as all the occurrences where a miss
        classification was detected. In other words it's the sum of all cells
        indexed by [t, i, j] where i and j are different.

        Returns
        -------
        float
            The total discordance from all label's matrices.

        """
        return self.get_total_sum() - self.get_sum_main_diagonal()

    @property
    def matrix(self):
        if self.confusion_matrix is not None:
            return self.confusion_matrix
        else:
            return None

    def get_info(self):
        return 'MOLConfusionMatrix: n_targets: ' + str(self.n_targets) + \
               ' - total_sum: ' + str(self.get_total_sum()) + \
               ' - total_discordance: ' + str(self.get_total_discordance()) + \
               ' - dtype: ' + str(self.dtype)

    def get_class_type(self):
        return 'collection'
