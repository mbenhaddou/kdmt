
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

class Table():
    def __init__(self, content, header):
        try:
            self._tabulate = __import__('tabulate').tabulate
        except:
            raise ImportError('tabulate is required to use the table module')
        self.content = content
        self.header = header

    def to_html(self):
        return self._tabulate(self.content, headers=self.header,
                              tablefmt='html')

    def __str__(self):
        return self._tabulate(self.content, headers=self.header,
                              tablefmt='grid')

    def _repr_html_(self):
        return self.to_html()

    def __repr__(self):
        return str(self)
