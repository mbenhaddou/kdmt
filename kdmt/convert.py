import numpy as np



def convert_numpy(value):
    if isinstance(value, (dict,)):
        for key in value:
            value[key] = convert_numpy(value[key])
        return value
    elif isinstance(value, (list, set, tuple)):
        return value.__class__(map(convert_numpy, value))
    elif isinstance(value, (np.generic,)):
        return np.asscalar(value)
    elif hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    else:
        return value
