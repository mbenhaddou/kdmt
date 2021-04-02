"""Useful things to do with dates"""
import datetime


def date_from_string(string, format_string=None):
    """Runs through a few common string formats for datetimes,
    and attempts to coerce them into a datetime. Alternatively,
    format_string can provide either a single string to attempt
    or an iterable of strings to attempt."""

    if isinstance(format_string, str):
        return datetime.datetime.strptime(string, format_string).date()

    elif format_string is None:
        format_string = [
            "%Y-%m-%d",
            "%m-%d-%Y",
            "%m/%d/%Y",
            "%d/%m/%Y",
        ]

    for format in format_string:
        try:
            return datetime.datetime.strptime(string, format).date()
        except ValueError:
            continue

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
