_get_attr_raise_on_attribute_error = "RAISE ON EXCEPTION"





def import_module(dotted_path):
    """
    Imports the specified module based on the
    dot notated import path for the module.
    """
    import importlib

    module_parts = dotted_path.split('.')
    module_path = '.'.join(module_parts[:-1])
    module = importlib.import_module(module_path)

    return getattr(module, module_parts[-1])

def initialize_class(data, *args, **kwargs):
    """
    :param data: A string or dictionary containing a import_path attribute.
    """
    if isinstance(data, dict):
        import_path = data.get('import_path')
        data.update(kwargs)
        Class = import_module(import_path)

        return Class(*args, **data)
    else:
        Class = import_module(data)

        return Class(*args, **kwargs)


def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj


def module_path_from_object(o):
    """Returns the fully qualified class path of the instantiated object."""
    return o.__class__.__module__ + "." + o.__class__.__name__


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Notes
    -----
    Code from sklearn

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState instance'.format(seed))



def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        print('package '+package+' is not installed. Installing...')
        pip.main(['install', package])

def get_attr(obj, string_rep, default=_get_attr_raise_on_attribute_error, separator="."):
    """ getattr via a chain of attributes like so:
    >>> import datetime
    >>> some_date = datetime.date.today()
    >>> get_attr(some_date, "month.numerator.__doc__")
    'int(x[, base]) -> integer\n\nConvert a string or number to an integer, ...
    """
    attribute_chain = string_rep.split(separator)

    current_obj = obj

    for attr in attribute_chain:
        try:
            current_obj = getattr(current_obj, attr)
        except AttributeError:
            if default is _get_attr_raise_on_attribute_error:
                raise AttributeError(
                    "Bad attribute \"{}\" in chain: \"{}\"".format(attr, string_rep)
                )
            return default

    return current_obj


class ImmutableWrapper(object):
    _obj = None
    _recursive = None

    def __init__(self, obj, recursive):
        self._obj = obj
        self._recursive = recursive

    def __setattr__(self, name, val):
        if name == "_obj" and self._obj is None:
            object.__setattr__(self, name, val)
            return
        elif name == "_recursive" and self._recursive is None:
            object.__setattr__(self, name, val)
            return

        raise AttributeError("This object has been marked as immutable; you cannot set its attributes.")

    def __getattr__(self, name):
        if self._recursive:
            return immutable(getattr(self._obj, name), recursive=self._recursive)

        return getattr(self._obj, name)

    def __repr__(self):
        return "<Immutable {}: {}>".format(self._obj.__class__.__name__, self._obj.__repr__())


def immutable(obj, recursive=True):
    """wraps the argument in a pass-through class that disallows all attribute
    setting. If the `recursive` flag is true, all attribute accesses will
    return an immutable-wrapped version of the "real" attribute."""
    return ImmutableWrapper(obj, recursive)
