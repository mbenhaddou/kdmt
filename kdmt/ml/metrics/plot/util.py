import matplotlib.pyplot as plt
from decorator import decorator
from kdmt.objects import map_parameters_in_fn_call
import inspect

from collections import defaultdict
from copy import copy
from inspect import signature
from itertools import product
from kdmt.lists import can_iterate

@decorator
def set_default_ax(func, *args, **kwargs):
    params = map_parameters_in_fn_call(args, kwargs, func)

    if 'ax' not in params:
        raise Exception('ax is not a parameter in {}'.format(func))

    if params['ax'] is None:
        params['ax'] = plt.gca()

    return func(**params)

def _group_by(data, criteria):
    """
        Group objects in data using a function or a key
    """
    if isinstance(criteria, str):
        criteria_str = criteria

        def criteria(x):
            return x[criteria_str]

    res = defaultdict(list)
    for element in data:
        key = criteria(element)
        res[key].append(element)
    return res


def _get_params_value(params):
    """
        Given an iterator (k1, k2), returns a function that when called
        with an object obj returns a tuple of the form:
        ((k1, obj.parameters[k1]), (k2, obj.parameters[k2]))
    """
    # sort params for consistency
    ord_params = sorted(params)

    def fn(obj):
        l = []
        for p in ord_params:
            try:
                l.append((p, obj.parameters[p]))
            except:
                raise ValueError('{} is not a valid parameter'.format(p))
        return tuple(l)

    return fn


def _sorted_map_iter(d):
    ord_keys = sorted(d.keys())
    for k in ord_keys:
        yield (k, d[k])


def _product(k, v):
    """
        Perform the product between two objects
        even if they don't support iteration
    """
    if not can_iterate(k):
        k = [k]
    if not can_iterate(v):
        v = [v]
    return list(product(k, v))


def _mapping_to_tuple_pairs(d):
    """
        Convert a mapping object (such as a dictionary) to tuple pairs,
        using its keys and values to generate the pairs and then generating
        all possible combinations between those
        e.g. {1: (1,2,3)} -> (((1, 1),), ((1, 2),), ((1, 3),))
    """
    # order the keys, this will prevent different implementations of Python,
    # return different results from the same dictionary since the order of
    # iteration depends on it
    t = []
    ord_keys = sorted(d.keys())
    for k in ord_keys:
        t.append(_product(k, d[k]))
    return tuple(product(*t))


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
