import numpy as np


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


class Deque():
    """ Deque
    A simple buffer used to keep track of a limited number of unitary entries. It
    updates the buffer following a FIFO method, meaning that when the buffer is
    full and a new entry arrives, the oldest entry is pushed out of the queue.
    In theory it keeps track of simple, primitive objects, such as numeric values,
    but in practice it can be used to store any kind of object.
    For this framework the FastBuffer is mainly used to keep track of true y_values
    and predictions in a classification task context, so that we can keep updated
    statistics about the task being executed.
    Parameters
    ----------
    max_size: int
        Maximum size of the queue.
    object_list: list
        An initial list. Optional. If given the queue will be started with the
        values from this list.
    """

    def __init__(self, max_size, object_list=None):
        super().__init__()
        # Default values
        self.current_size = 0
        self.max_size = None
        self.buffer = []

        self.configure(max_size, object_list)

    def get_class_type(self):
        return 'data_structure'

    def configure(self, max_size, object_list):
        self.max_size = max_size
        if isinstance(object_list, list):
            self.add_element(object_list)

    def add_element(self, element_list):
        """ add_element
        Adds a new entry to the buffer. In case there are more elements in the
        element_list parameter than there is free space in the queue, elements
        from the queue are iteratively popped from the queue and appended to
        a list, which in the end is returned.
        """
        if (self.current_size + len(element_list)) <= self.max_size:
            for i in range(len(element_list)):
                self.buffer.append(element_list[i])
            self.current_size += len(element_list)
            return None

        else:
            aux = []
            for element in element_list:
                if self.is_full():
                    aux.append(self.get_next_element())
                self.buffer.append(element)
                self.current_size += 1
            return aux

    def get_next_element(self):
        """ get_next_element
        Pop the head of the queue.
        Returns
        -------
        int or float
            The first element in the queue.
        """
        result = None
        if len(self.buffer) > 0:
            result = self.buffer.pop(0)
            self.current_size -= 1
        return result

    def clear_queue(self):
        self._clear_all()

    def _clear_all(self):
        del self.buffer[:]
        self.buffer = []
        self.current_size = 0
        self.configure(self.max_size, None)

    def is_full(self):
        return self.current_size == self.max_size

    def is_empty(self):
        return self.current_size == 0

    def get_current_size(self):
        return self.current_size

    def peek(self):
        """ peek
        Peek the head of the queue, without removing or altering it.
        Returns
        -------
        int or float
            The head of the queue.
        """
        try:
            return self.buffer[0]
        except IndexError:
            return None

    def get_deque(self):
        return self.buffer


class ComplexDeque():
    """ ComplexDeque

    A complex buffer used to keep track of a limited number of complex entries. It
    updates the buffer following a FIFO method, meaning that when the buffer is
    full and a new entry arrives, the oldest entry is pushed out of the queue.

    We use the term complex entry to specify that each entry is a set of n
    predictions, one for each classification task. This structure is used to keep
    updated statistics from a multi output context.

    Parameters
    ----------
    max_size: int
        Maximum size of the queue.

    width: int
        The width from a complex entry, in other words how many classification
        tasks are there to keep track of.

    Examples
    --------
    It works similarly to the FastBuffer structure, except that it keeps track
    of more than one value per entry. For a complete example, please see
    skmultiflow.evaluation.measure_collection.WindowMultiTargetClassificationMeasurements'
    implementation, where the FastComplexBuffer is used to keep track of the
    MultiOutputLearner's statistics.

    """

    def __init__(self, max_size, width):
        super().__init__()
        # Default values
        self.current_size = 0
        self.max_size = None
        self.width = None
        self.buffer = []

        self.configure(max_size, width)

    def get_class_type(self):
        return 'data_structure'

    def configure(self, max_size, width):
        self.max_size = max_size
        self.width = width

    def add_element(self, element_list):
        """ add_element

        Adds a new entry to the buffer. In case there are more elements in the
        element_list parameter than there is free space in the queue, elements
        from the queue are iteratively popped from the queue and appended to
        a list, which in the end is returned.

        Parameters
        ----------
        element_list: list or numpy.array
            A list with all the elements that are to be added to the queue.

        Returns
        -------
        list
            If no elements need to be popped from the queue to make space for new
            entries there is no return. On the other hand, if elements need to be
            removed, they are added to an auxiliary list, and that list is returned.

        """
        is_list = True
        dim = 1
        if hasattr(element_list, 'ndim'):
            dim = element_list.ndim
        if (dim > 1) or hasattr(element_list[0], 'append'):
            size, width = 0, 0
            if hasattr(element_list, 'append'):
                size, width = len(element_list), len(element_list[0])
            elif hasattr(element_list, 'shape'):
                is_list = False
                size, width = element_list.shape
            self.width = width
            if width != self.width:
                return None
        else:
            size, width = 0, 0
            if hasattr(element_list, 'append'):
                size, width = 1, len(element_list)
            elif hasattr(element_list, 'size'):
                is_list = False
                size, width = 1, element_list.size
            self.width = width
            if width != self.width:
                return None

        if not is_list:
            if size == 1:
                items = [element_list.tolist()]
            else:
                items = element_list.tolist()
        else:
            if size == 1:
                items = [element_list]
            else:
                items = element_list

        if (self.current_size + size) <= self.max_size:
            for i in range(size):
                self.buffer.append(items[i])
            self.current_size += size
            return None
        else:
            aux = []
            for element in items:
                if self.is_full():
                    aux.append(self.get_next_element())
                self.buffer.append(element)
                self.current_size += 1
            return aux

    def get_next_element(self):
        """ get_next_element

        Pop the head of the queue.

        Returns
        -------
        tuple
            The first element of the queue.

        """
        result = None
        if len(self.buffer) > 0:
            result = self.buffer.pop(0)
            self.current_size -= 1
        return result

    def clear_queue(self):
        self._clear_all()

    def _clear_all(self):
        del self.buffer[:]
        self.buffer = []
        self.current_size = 0
        self.configure(self.max_size, None)

    def print_queue(self):
        print(self.buffer)

    def is_full(self):
        return self.current_size == self.max_size

    def is_empty(self):
        return self.current_size == 0

    def get_current_size(self):
        return self.current_size

    def peek(self):
        """ peek

        Peek the head of the queue, without removing or altering it.

        Returns
        -------
        tuple
            The head of the queue.

        """
        try:
            return self.buffer[0]
        except IndexError:
            return None

    def get_queue(self):
        return self.buffer

    def get_info(self):
        return 'FastBuffer: max_size: ' + str(self.max_size) \
               + ' - current_size: ' + str(self.current_size) \
               + ' - width: ' + str(self.width)
