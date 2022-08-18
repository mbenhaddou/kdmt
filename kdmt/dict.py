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