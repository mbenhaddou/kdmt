import pandas as pd
from collections.abc import Iterable
import numpy as np



def is_int(x):
    """Check if x is of integer type, but not boolean"""
    # boolean are subclasses of integers in Python, so explicitly exclude them
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def is_float(x):
    """Check if x is of float type"""
    return isinstance(x, (float, np.floating))


def get_first_non_nan_value(input_value):
    """Return the first not ``nan`` value when possible.

    Convert to ``pandas.Series`` if the ``input_value`` is not already. This helps to detect
    easier the ``nan`` values. We filter the values that are ``nans`` since pandas does not
    detect them properly in their ``_guess_datetime_format_for_array``. Also there is a bug in
    ``pandas`` that does not support ``numpy.str_`` data type, that is why we use
    ``pandas.Series`` and convert the data type to ``string`` and then to ``numpy.ndarray``.

    Args:
       input_value (pandas.Series, np.ndarray, list, or str):
            Input to return the first non ``nan`` value.

    Returns:
        str or ``nan``:
            Returns either the first valid value or ``nan``.
    """
    value = input_value
    if not isinstance(value, pd.Series):
        value = pd.Series(input_value)

    value = value[~value.isna()]
    value = value.astype(str).to_numpy()
    if len(value):
        return value[0]

    if isinstance(input_value, Iterable) and not isinstance(input_value, str):
        return input_value[0]

    return input_value


def is_numerical_type(value):
    """Determine if the input is numerical or not.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.

    Returns:
        bool:
            True if the input is numerical, False if not.
    """
    return pd.isna(value) | pd.api.types.is_float(value) | pd.api.types.is_integer(value)


def is_boolean_type(value):
    """Determine if the input is a boolean or not.

    Args:
        value (int, str, datetime, bool):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a boolean, False if not.
    """
    return True if pd.isna(value) | (value is True) | (value is False) else False


def alnum_or_num(text):
    return any(char.isdigit() for char in text)
