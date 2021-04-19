import collections.abc

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
    for k, v in new.items():
        if isinstance(v, collections.abc.Mapping):
            origin[k] = update(origin.get(k, {}), v)
        else:
            origin[k] = v
    return origin


def get_keys_for_value(dictionary, value):
    return [key for key,val in dictionary.items() if val == value]

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
        'k3':{
            'k31': 'new_v31',
            'k32': 'v32'
        }
    }
    print(update(dic, new))