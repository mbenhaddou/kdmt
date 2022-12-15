import collections.abc
from copy import copy, deepcopy
import itertools

from kdmt.infer import is_list_of_one_element, is_dict_of_one_element
def nested_dict_get_values(key, dictionary):
    if hasattr(dictionary, 'items'):
        for k, v in dictionary.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in nested_dict_get_values(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in nested_dict_get_values(key, d):
                        yield result


def nested_dict_has_key(key, dictionary):

    if hasattr(dictionary, 'items'):
        for k, v in dictionary.items():
            if k == key:
                return True
            if isinstance(v, dict):
                if nested_dict_has_key(key, v):
                    return True
            elif isinstance(v, list):
                for d in v:
                    if nested_dict_has_key(key, d):
                        return True



def nested_dict_get_key_path( key, dict_obj, path=[]):
    """
    Description
    -----------
    Use this function to get the full path of a key in a dictionary.
    the dictionay can be a simple dictionary or a nested dictionary that may also contain lists of other nested dictionnaries.

    Parameters
    ----------
    @param dict_obj: the dictionary that can con contain the key
    @param key: the target key
    @return: a list containg the path sequence that leads to the key

    Example
    -------

    >>>from kdmt.dict import nested_dict_get_key_path
    >>> dictionary ={
        'k1': 'v1',
        'k2': 'v2',
        'k3':[{
            'k31': 'new_v31',
            'k32': 'v32'
        },
        {
            'k41': 'new_v31',
            'k42': 'v32'
        }
        ]
    }

    >>> print(nested_dict_get_key_path(dictionary, 'k31')
    #['k3', 0, 1]
    """
    q = [(dict_obj, [])]
    while q:
        n, p = q.pop(0)
        if p and p[-1]==key:
            return p
        if isinstance(n, dict):
            for k, v in n.items():
                q.append((v, p+[k]))
        elif isinstance(n, list):
            for i, v in enumerate(n):
                q.append((v, p+[i]))


def nested_dict_set_key_value(key, dict_obj, new_value):
    """
    Description
    -----------
    Use this function to set the value of a nested dictinary based on key path.
    the dictionay can be a simple dictionary or a nested dictionary that may also contain lists of other nested dictionnaries.

    Parameters
    ----------
    @param dict_obj: the dictionary that can con contain the key
    @param key: the target key
    @new_value: the new value that will be stored in key
    @return: a list containg the path sequence that leads to the key

    Example
    -------

    >>>from kdmt.dict import nested_dict_set_key_value
    >>> dictionary ={
        'k1': 'v1',
        'k2': 'v2',
        'k3':[{
            'k31': 'new_v31',
            'k32': 'v32'
        },
        {
            'k41': 'new_v31',
            'k42': 'v32'
        }
        ]
    }
    >>> print(nested_dict_set_key_value2(['k3', 0, 'k31', 'k332'], new, 'new_V32'))
    #{'k1': 'v1', 'k2': 'v2', 'k3': [{'k31': {'k331': 'new_v31', 'k332': 'new_V32'}, 'k32': 'v32'}, {'k41': 'new_v31', 'k42': 'v32'}]}
    """
    if not isinstance(key, list):
        key=[key]

    new_dic=dict_obj
    for sub_key in key[:-1]:
        new_dic = new_dic[sub_key]

    new_dic[key[-1]] = new_value
    return dict_obj

def nested_dict_get_value(key, dictionary):

    if hasattr(dictionary, 'items'):
        for k, v in dictionary.items():
            if k == key:
                return v
            if isinstance(v, dict):
                val= nested_dict_get_value(key, v)
                if val:
                     return val
            elif isinstance(v, list):
                for d in v:
                    val = nested_dict_get_value(key, d)
                    if val:
                        return val

def update(origin, new):
    new=deepcopy(new)
    for k, v in new.items():
        if isinstance(v, collections.abc.Mapping):
            origin[k] = update(origin.get(k, {}), v)
        else:
            origin[k] = v
    return origin


def get_keys_for_value(dictionary, value):
    return [key for key,val in dictionary.items() if val == value]


def format_dict(_dict, tidy=True):
    """
    This function format a dict. If the main dict or a deep dict has only on element
     {"col_name":{0.5: 200}} we get 200
    :param _dict: dict to be formatted
    :param tidy:
    :return:
    """

    if tidy is True:
        levels = 2
        while (levels >= 0):
            levels -= 1
            if is_list_of_one_element(_dict):
                _dict = _dict[0]
            elif is_dict_of_one_element(_dict):
                _dict = next(iter(_dict.values()))
            else:
                return _dict

    else:
        if is_list_of_one_element(_dict):
            return _dict[0]
        else:
            return _dict


if __name__=="__main__":
    dic={
        'k1': 'v1',
        'k2': 'v2',
        'k3':{
            'k31': 'v31'
        }
    }
    new ={
        'k1': 'v1',
        'k2': 'v2',
        'k3':[{
            'k31': {
            'k331': 'new_v31',
            'k332': 'v32'
        },
            'k32': 'v32'
        },
        {
            'k41': 'new_v31',
            'k42': 'v32'
        }
        ]
    }
    print(nested_dict_set_key_value(['k3', 0, 'k31', 'k332'], new, 'new_V32'))

 #   nested_dict_set_key_value(r, new, 'bla')



def zip_dict(*dicts):
  """Iterate over items of dictionaries grouped by their keys."""
  for key in set(itertools.chain(*dicts)):  # set merge all keys
    # Will raise KeyError if the dict don't have the same keys
    yield key, tuple(d[key] for d in dicts)


import copy


class Dict(dict):

    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in kwargs.items():
            __self[key] = __self._hook(val)

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError("'Dict' object attribute "
                                 "'{0}' is read-only".format(name))
        else:
            self[name] = value

    def __setitem__(self, name, value):
        super(Dict, self).__setitem__(name, value)
        try:
            p = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')

    def __add__(self, other):
        if not self.keys():
            return other
        else:
            self_type = type(self).__name__
            other_type = type(other).__name__
            msg = "unsupported operand type(s) for +: '{}' and '{}'"
            raise TypeError(msg.format(self_type, other_type))

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __getattr__(self, item):
        if item not in self:
            return None
        return self.__getitem__(item)

    def __missing__(self, name):
        return self.__class__(__parent=self, __key=name)

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else
                    item for item in value)
            else:
                base[key] = value
        return base

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if ((k not in self) or
                    (not isinstance(self[k], dict)) or
                    (not isinstance(v, dict))):
                self[k] = v
            else:
                self[k].update(v)

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default
