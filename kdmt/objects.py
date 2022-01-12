import re
import pip
import numpy as np
import numbers
from copy import copy
import inspect
import pydoc
from inspect import signature
_get_attr_raise_on_attribute_error = "RAISE ON EXCEPTION"


def isinstance_of_class(obj):
    return hasattr(obj, '__dict__')

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

def class_name(obj):
    class_name = str(type(obj))
    class_name = re.search(".*'(.+?)'.*", class_name).group(1)
    return class_name

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        print('package '+package+' is not installed. Installing...')
        pip.main(['install', package])
        try:
            __import__(package)
        except:
            print('package ' + package + ' installation failed.')

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

def map_parameters_in_fn_call(args, kwargs, func):
    """
    Based on function signature, parse args to to convert them to key-value
    pairs and merge them with kwargs
    Any parameter found in args that does not match the function signature
    is still passed.
    Missing parameters are filled with their default values
    """
    # Get missing parameters in kwargs to look for them in args
    args_spec = inspect.getargspec(func).args
    params_all = set(args_spec)
    params_missing = params_all - set(kwargs.keys())

    if 'self' in args_spec:
        offset = 1
    else:
        offset = 0

    # Get indexes for those args
    idxs = [args_spec.index(name) for name in params_missing]

    # Parse args
    args_parsed = dict()

    for idx in idxs:
        key = args_spec[idx]

        try:
            value = args[idx - offset]
        except IndexError:
            pass
        else:
            args_parsed[key] = value

    parsed = copy(kwargs)
    parsed.update(args_parsed)

    # fill default values
    default = {k: v.default for k, v
               in signature(func).parameters.items()
               if v.default != inspect._empty}

    to_add = set(default.keys()) - set(parsed.keys())

    default_to_add = {k: v for k, v in default.items() if k in to_add}
    parsed.update(default_to_add)

    return parsed

def class_from_module_path(module_path):
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects. """
    import importlib

    # load the module, will raise ImportError if module cannot be loaded
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition('.')
        try:
            m = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            return getattr(m, class_name)
        except:
            print('The module: '+module_path+ "could not be found. Please install any missing libraries")
            return None
    else:
        return globals()[module_path]

def load_data_object(data, **kwargs):
    """
    Load Object From Dict
    Args:
        data:
        **kwargs:

    Returns:

    """
    module_name = f"{data['__module__']}.{data['__class_name__']}"
    obj = pydoc.locate(module_name)(**data['config'], **kwargs)
    if hasattr(obj, '_override_load_model'):
        obj._override_load_model(data)

    return obj
