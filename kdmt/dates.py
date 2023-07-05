"""Useful things to do with dates"""
import datetime
import math
from dateutil.parser import parse
import pandas as pd
from pandas.core.tools.datetimes import _guess_datetime_format_for_array
from typing import Iterable
def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def get_datetime_format(value):
    """Get the ``strftime`` format for a given ``value``.

    This function returns the ``strftime`` format of a given ``value`` when possible.
    If the ``_guess_datetime_format_for_array`` from ``pandas.core.tools.datetimes`` is
    able to detect the ``strftime`` it will return it as a ``string`` if not, a ``None``
    will be returned.

    Args:
        value (pandas.Series, np.ndarray, list, or str):
            Input to attempt detecting the format.

    Return:
        String representing the datetime format in ``strftime`` format or ``None`` if not detected.
    """
    if not isinstance(value, pd.Series):
        value = pd.Series(value)

    value = value[~value.isna()]
    value = value.astype(str).to_numpy()
    return _guess_datetime_format_for_array(value)


def is_datetime_type(value):
    """Determine if the input is a datetime type or not.

    Args:
        value (pandas.DataFrame, int, str or datetime):
            Input to evaluate.

    Returns:
        bool:
            True if the input is a datetime type, False if not.
    """
    if isinstance(value, Iterable) and not isinstance(value, str):
        value = get_first_non_nan_value(value)

    return (
        pd.api.types.is_datetime64_any_dtype(value)
        or isinstance(value, pd.Timestamp)
        or isinstance(value, datetime)
        or bool(get_datetime_format([value]))
    )


def validate_datetime_format(column, datetime_format):
    """Determine the values of the column that match the datetime format.

    Args:
        column (pd.Series):
            Column to evaluate.
        datetime_format (str):
            The datetime format.

    Returns:
        pd.Series:
            Series of booleans, with True if the value matches the format, False if not.
    """
    pandas_datetime_format = datetime_format.replace('%-', '%')
    datetime_column = pd.to_datetime(
        column,
        errors='coerce',
        format=pandas_datetime_format
    )
    valid = pd.isna(column) | ~pd.isna(datetime_column)

    return set(column[~valid])


def convert_to_timedelta(column):
    """Convert a ``pandas.Series`` to one with dtype ``timedelta``.

    ``pd.to_timedelta`` does not handle nans, so this function masks the nans, converts and then
    reinserts them.

    Args:
        column (pandas.Series):
            Column to convert.

    Returns:
        pandas.Series:
            The column converted to timedeltas.
    """
    nan_mask = pd.isna(column)
    column[nan_mask] = 0
    column = pd.to_timedelta(column)
    column[nan_mask] = pd.NaT
    return column



def date_from_string(string, format_string=None,return_datetime=True ):
    """Runs through a few common string formats for datetimes,
    and attempts to coerce them into a datetime. Alternatively,
    format_string can provide either a single string to attempt
    or an iterable of strings to attempt."""
    formats_string = [
            "%Y-%m-%d",
            "%m-%d-%Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ]
    if isinstance(format_string, str):
        formats_string.insert(0, format_string)






    for format in formats_string:
        try:
            if return_datetime:
                return datetime.datetime.strptime(string, format)
            else:
                return datetime.datetime.strptime(string, format).date()
        except ValueError:
            parse(string)

    raise ValueError("Could not produce date from string: {}".format(string))


def to_datetime(plain_date, hours=0, minutes=0, seconds=0, ms=0):
    """given a datetime.date, gives back a datetime.datetime"""
    # don't mess with datetimes
    if isinstance(plain_date, datetime.datetime):
        return plain_date
    return datetime.datetime(
        plain_date.year,
        plain_date.month,
        plain_date.day,
        hours,
        minutes,
        seconds,
        ms,
    )


def days_ago(days, give_datetime=True):
    delta = datetime.timedelta(days=days)
    dt = datetime.datetime.now() - delta
    if give_datetime:
        return dt
    else:
        return dt.date()


def days_ahead(days, give_datetime=True):
    delta = datetime.timedelta(days=days)
    dt = datetime.datetime.now() + delta
    if give_datetime:
        return dt
    else:
        return dt.date()


def days_ahead_from_date(date, days, date_format, give_datetime=True):
    date_t=date
    if isinstance(date, str):
        date_t=date_from_string(date, date_format)
    elif not isinstance(date, datetime.datetime):
        raise Exception("the given date parameter is not a string of a datetime")
    delta = datetime.timedelta(days=days)
    dt = date_t + delta
    if give_datetime:
        return dt
    else:
        return dt.date()


def datetime_to_integer(date_time, date_format='%d-%m-%Y %H:%M:%S', day_min_unit='minute'):


    if day_min_unit == 'hour':
        intra_day_intervals=24
    elif day_min_unit == 'minute':
        intra_day_intervals=24*60
    elif day_min_unit == 'second':
        intra_day_intervals=86400
    else:
        raise Exception("wrong 'day_min_unit' value execting: 'hour','minute' or 'second', getting "+ str(day_min_unit))

    if not isinstance(date_time, datetime.datetime):
        if isinstance(date_time, str):
            dt=datetime.datetime.strptime(date_time, date_format)
        else:
            raise(Exception("Date value should be either a datetime or a string."))
    else:
        dt=date_time

    year = dt.year
    month = dt.month
    day  = dt.day
    remaining_day = (dt.minute*60 + dt.hour*60*60+dt.second)/86400.0

    weekday = datetime.date(year, month, day).weekday()


    if intra_day_intervals is None:
        return weekday
    else:
        return int((weekday+remaining_day) * intra_day_intervals)


def integer_to_datetime(num_integer, reference_date, day_min_unit='minute', date_format='%d-%m-%Y %H:%M:%S'):
    """Convert integer return from solver to readable date for logging or visualization.

    Args:
        num_integer (_type_): _description_
        within_day (bool, optional): _description_. Defaults to True.
        num_hours_per_day (_type_, optional): _description_. Defaults to HOURS_PER_DAY_MODEL.
        start_hour_of_day (int, optional): _description_. Defaults to 0.

    Returns:
        string: _description_
    """

    if day_min_unit=='minute':
        day_units=24*60
    elif day_min_unit=='hour':
        day_units=24
    elif day_min_unit=='second':
        day_units=86400
    day = math.floor(num_integer / day_units)
    plus_number = num_integer % day_units


    if day_min_unit=='minute':
        plus_number=plus_number*60
    elif day_min_unit=='hour':
        plus_number=plus_number*60*60


    minutes, seconds = divmod(plus_number, 60)

    hours, minutes = divmod(minutes, 60)





    new_date= date_from_string(reference_date, date_format)
    new_date = new_date + datetime.timedelta(days=day, hours=hours, minutes=minutes,seconds=seconds)


    return new_date

if __name__ == "__main__":
    num=days_ahead_from_date("02-09-2022 16:20:00", 4, date_format='%d-%m-%Y %H:%M:%S')
    print(num)
#    print(integer_to_datetime(num, reference_date="29-08-2022 00:00:00", day_min_unit='hour'))