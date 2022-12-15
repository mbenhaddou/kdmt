import re
from abc import ABC
import inspect
import numpy as np
import pandas as pd
from typing import Callable, Set
from kdmt.lists import one_tuple_to_val, val_to_list
from kdmt.infer import  is_list, is_list_of_list, is_list_of_int, is_list_of_str

def get_function_params(function: Callable) -> Set[str]:
    return inspect.signature(function).parameters



class MiscFunctions(ABC):
    """
    Functions for internal use or to be called using 'F': `from optimus.functions import F`
    Note: some methods needs to be static so they can be passed to a Dask worker.
    """
    def from_delayed(self, delayed):
        return delayed[0]

    def to_delayed(self, delayed):
        return [delayed]

    def apply_delayed(self, series, func, *args, **kwargs):
        result = self.to_delayed(series)
        result = [partition.apply(func, *args, **kwargs) for partition in result]
        result = self.from_delayed(result)
        result.index = series.index
        return result

    def map_delayed(self, series, func, *args, **kwargs):
        result = self.to_delayed(series)
        result = [partition.map(func, *args, **kwargs) for partition in result]
        result = self.from_delayed(result)
        result.index = series.index
        return result

    @staticmethod
    def sort_df(dfd, cols, ascending):
        """
        Sort rows taking into account one column
        """
        return dfd.sort_values(cols, ascending=ascending)

    @staticmethod
    def reverse_df(dfd):
        """
        Reverse rows
        """
        return dfd[::-1]

    @staticmethod
    def append(dfd, dfd2):
        """
        Append two dataframes
        """
        return dfd.append(dfd2)

    def _new_series(self, *args, **kwargs):
        """
        Creates a new series (also known as column)
        """
        return self._functions.Series(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return one_tuple_to_val((*(a for a in args), *(kwargs[k] for k in kwargs)))

    @staticmethod
    def to_dict(series) -> dict:
        """
        Convert series to a Python dictionary
        """
        return series.to_dict()

    @staticmethod
    def to_items(series) -> dict:
        """
        Convert series to a list of tuples [(index, value), ...]
        """
        df = series.reset_index()
        return df.to_dict(orient='split')['data']

    def _to_boolean(self, series):
        """
        Converts series to bool
        """
        return series.map(lambda v: bool(v), na_action=None).astype('bool')

    def to_boolean(self, series):
        return self._to_boolean(series)

    def _to_boolean_none(self, series):
        """
        Converts series to boolean
        """
        return series.map(lambda v: bool(v), na_action='ignore').astype('object')

    def to_boolean_none(self, series):
        return self._to_boolean_none(series)

    def _to_float(self, series):
        """
        Converts a series values to floats
        """
        try:
            return self._new_series(np.vectorize(fast_float)(series, default=np.nan).flatten())
        except:
            return self._new_series(self._functions.to_numeric(series, errors='coerce')).astype('float')

    def to_float(self, series):
        return self._to_float(series)

    def _to_integer(self, series, default=0):
        """
        Converts a series values to integers
        """
        try:
            return self._new_series(np.vectorize(fast_int)(series, default=default).flatten())
        except:
            return self._new_series(self._functions.to_numeric(series, errors='coerce').fillna(default)).astype('int')

    def to_integer(self, series, default=0):
        return self._to_integer(series, default)

    def _to_datetime(self, series, format):
        return series

    def to_datetime(self, series, format):
        return self._to_datetime(series, format)

    @staticmethod
    def to_string(series):
        return series.astype(str)

    @staticmethod
    def to_string_accessor(series):
        return series.astype(str).str

    @staticmethod
    def duplicated(dfd, keep, subset):
        return dfd.duplicated(keep=keep, subset=subset)

    def date_format(self, series):
        dtype = str(series.dtype)
        if dtype in self.constants.STRING_TYPES:
            import pydateinfer
            return pydateinfer.infer(self.compute(series).values)
        elif dtype in self.constants.DATETIME_TYPES:
            return True

        return False

    def min(self, series, numeric=False, string=False):
        if numeric:
            series = self.to_float(series)

        if string or str(series.dtype) in self.constants.STRING_TYPES:
            return self.to_string(series.dropna()).min()
        else:
            return series.min()

    def max(self, series, numeric=False, string=False):

        if numeric:
            series = self.to_float(series)

        if string or str(series.dtype) in self.constants.STRING_TYPES:
            return self.to_string(series.dropna()).max()
        else:
            return series.max()

    def mean(self, series):
        return self.to_float(series).mean()

    @staticmethod
    def mode(series):
        return series.mode()

    @staticmethod
    def crosstab(series, other):
        return pd.crosstab(series, other)

    def std(self, series):
        return self.to_float(series).std()

    def sum(self, series):
        return self.to_float(series).sum()

    def cumsum(self, series):
        return self.to_float(series).cumsum()

    def cumprod(self, series):
        return self.to_float(series).cumprod()

    def cummax(self, series):
        return self.to_float(series).cummax()

    def cummin(self, series):
        return self.to_float(series).cummin()

    def var(self, series):
        return self.to_float(series).var()

    def count_uniques(self, series, estimate=False):
        return self.to_string(series).nunique()

    def unique_values(self, series, estimate=False):
        # Cudf can not handle null so we fill it with non zero values.
        return self.to_string(series).unique()

    @staticmethod
    def count_na(series):
        return series.isna().sum()
        # return {"count_na": {col_name:  for col_name in columns}}
        # return np.count_nonzero(_df[_serie].isnull().values.ravel())
        # return cp.count_nonzero(_df[_serie].isnull().values.ravel())

    @staticmethod
    def count_zeros(series, *args):
        raise NotImplemented

    @staticmethod
    def kurtosis(series):
        raise NotImplemented

    @staticmethod
    def skew(series):
        raise NotImplemented

    # import dask.dataframe as dd
    def mad(self, series, error=False, more=False, estimate=False):

        _series = self.to_float(series).dropna()

        if not estimate:
            _series = self.compute(_series)

        if not len(_series):
            return np.nan
        else:
            median_value = _series.quantile(0.5)
            mad_value = (_series - median_value).abs().quantile(0.5)
            if more:
                return {"mad": mad_value, "median": median_value}
            else:
                return mad_value

    # TODO: dask seems more efficient triggering multiple .min() task, one for every column
    # cudf seems to be calculate faster in on pass using df.min()
    def range(self, series):
        series = self.to_float(series)
        return {"min": series.min(), "max": series.max()}

    def percentile(self, series, values, error, estimate=False):

        _series = self.to_float(series).dropna()

        if not estimate:
            _series = self.compute(_series)

        if not len(_series):
            return np.nan
        else:
            @self.delayed
            def format_percentile(_s):
                if hasattr(_s, "to_dict"):
                    return _s.to_dict()
                else:
                    return _s

            return format_percentile(_series.quantile(values))

    # def radians(series):
    #     return series.to_float().radians()
    #
    # def degrees(series, *args):
    #     return call(series, method_name="degrees")

    ###########################

    def z_score(self, series):
        t = self.to_float(series)
        return t - t.mean() / t.std(ddof=0)

    def modified_z_score(self, series, estimate):
        series = self.to_float(series)

        _series = series.dropna()

        if not estimate:
            _series = self.compute(_series)

        if not len(_series):
            return np.nan
        else:
            median = _series.quantile(0.5)
            mad = (_series - median).abs().quantile(0.5)

            return abs(0.6745 * (series - median) / mad)

    def clip(self, series, lower_bound, upper_bound):
        return self.to_float(series).clip(float(lower_bound), float(upper_bound))

    def cut(self, series, bins, labels, default):
        if is_list_of_int(bins):
            return pd.cut(self.to_float(series), bins, include_lowest=True, labels=labels)
        elif is_list_of_str(bins):
            conditions = [series.str.contains(i) for i in bins]

            return np.select(conditions, labels, default=default)

    def qcut(self, series, quantiles):
        return pd.qcut(series, quantiles)

    def abs(self, series):
        return self.to_float(series).abs()

    def exp(self, series):
        return self.to_float(series).exp()

    def len(self, series):
        return self.to_string_accessor(series).len()

    @staticmethod
    def sqrt(series):
        raise NotImplemented

    def mod(self, series, other):
        return self.to_float(series).mod(other)

    def round(self, series, decimals):
        return self.to_float(series).round(decimals)

    def pow(self, series, exponent):
        return self.to_float(series).pow(exponent)

    def floor(self, series):
        return self.to_float(series).floor()


    @staticmethod
    def radians(series):
        return np.radians(series)


    @staticmethod
    def degrees(series):
        return np.degrees(series)

    @staticmethod
    def ln(series):
        return np.log(series)

    @staticmethod
    def log(series, base):
        return np.log(series)

    @staticmethod
    def ceil(series):
        return np.ceil(series)

    @staticmethod
    def sin(self, series):
        return np.sin(series)

    @staticmethod
    def cos(self, series):
        return np.cos(series)

    @staticmethod
    def tan(self, series):
        return np.tan(series)

    @staticmethod
    def asin(self, series):
        return np.asin(series)

    @staticmethod
    def acos(self, series):
        return np.acos(series)

    @staticmethod
    def atan(self, series):
        return np.atan(series)

    @staticmethod
    def sinh(self, series):
        return np.sinh(series)

    @staticmethod
    def cosh(self, series):
        raise NotImplemented

    @staticmethod
    def tanh(self, series):
        return np.tanh(series)

    @staticmethod
    def asinh(self, series):
        return np.asinh(series)

    @staticmethod
    def acosh(self, series):
        return np.acosh(series)

    @staticmethod
    def atanh(self, series):
        return np.atanh(series)

    # Strings
    def match(self, series, regex):
        return self.to_string_accessor(series).match(regex)


    def lower(self, series):
        return self.to_string_accessor(series).lower()


    def upper(self, series):
        return self.to_string_accessor(series).upper()


    def title(self, series):
        return self.to_string_accessor(series).title()

    def capitalize(self, series):
        return self.to_string_accessor(series).capitalize()

    def pad(self, series, width, side, fillchar=""):
        return self.to_string_accessor(series).pad(width, side, fillchar)

    def extract(self, series, regex):
        return self.to_string_accessor(series).extract(regex)

    def slice(self, series, start, stop, step):
        return self.to_string_accessor(series).slice(start, stop, step)

    def trim(self, series):
        return self.to_string_accessor(series).strip()

    def strip_html(self, series):
        return self.to_string(series).replace('<.*?>', '', regex=True)

    def replace_chars(self, series, search, replace_by, ignore_case):
        search = val_to_list(search, convert_tuple=True)
        if ignore_case:
            str_regex = [r'(?i)%s' % re.escape(s) for s in search]
        else:
            str_regex = [r'%s' % re.escape(s) for s in search]
        return self.to_string(series).replace(str_regex, replace_by, regex=True)

    def replace_words(self, series, search, replace_by, ignore_case):
        search = val_to_list(search, convert_tuple=True)
        if ignore_case:
            str_regex = [r'(?i)\b%s\b' % re.escape(s) for s in search]
        else:
            str_regex = [r'\b%s\b' % re.escape(s) for s in search]
        return self.to_string(series).replace(str_regex, replace_by, regex=True)

    def replace_full(self, series, search, replace_by, ignore_case):
        search = val_to_list(search, convert_tuple=True)
        if ignore_case:
            str_regex = [r'(?i)^%s$' % re.escape(s) for s in search]
        else:
            str_regex = [r'^%s$' % re.escape(s) for s in search]
        return series.replace(str_regex, replace_by, regex=True)

    def replace_values(self, series, search, replace_by, ignore_case):
        search = val_to_list(search, convert_tuple=True)

        if ignore_case:
            regex = True
            search = [(r'(?i)%s' % re.escape(s)) for s in search]
        else:
            regex = False

        if is_list(replace_by) and is_list_of_list(search):
            for _s, _r in zip(search, replace_by):
                series = series.replace(_s, _r, regex=regex)

        else:
            series = series.replace(search, replace_by, regex=regex)

        return series

    def replace_regex_chars(self, series, search, replace_by, ignore_case):
        search = val_to_list(search, convert_tuple=True)
        if ignore_case:
            str_regex = [r'(?i)%s' % s for s in search]
        else:
            str_regex = [r'%s' % s for s in search]
        return self.to_string(series).replace(str_regex, replace_by, regex=True)

    def replace_regex_words(self, series, search, replace_by, ignore_case):
        search = val_to_list(search, convert_tuple=True)
        if ignore_case:
            str_regex = [r'(?i)\b%s\b' % s for s in search]
        else:
            str_regex = [r'\b%s\b' % s for s in search]
        return self.to_string(series).replace(str_regex, replace_by, regex=True)

    def replace_regex_full(self, series, search, replace_by, ignore_case):
        search = val_to_list(search, convert_tuple=True)
        if ignore_case:
            str_regex = [r'(?i)^%s$' % s for s in search]
        else:
            str_regex = [r'^%s$' % s for s in search]
        return series.replace(str_regex, replace_by, regex=True)

    def remove_numbers(self, series):
        return self.to_string_accessor(series).replace(r'\d+', '', regex=True)

    def remove_white_spaces(self, series):
        return self.to_string_accessor(series).replace(" ", "")

    def remove_urls(self, series):
        return self.to_string_accessor(series).replace(r"https?://\S+|www\.\S+", "", regex=True)

    def normalize_spaces(self, series):
        return self.to_string_accessor(series).replace(r" +", " ", regex=True)


    # @staticmethod# def len(series):
    #     return series.str.len()

    def normalize_chars(self, series):
        raise NotImplemented

    def find(self, sub, start=0, end=None):
        series = self.series
        return self.to_string_accessor(series).find(sub, start, end)

    def rfind(self, series, sub, start=0, end=None):
        return self.to_string_accessor(series).rfind(sub, start, end)

    def left(self, series, position):
        return self.to_string_accessor(series)[:position]

    def right(self, series, position):
        return self.to_string_accessor(series)[-1 * position:]

    def mid(self, series, _start, _n):
        return self.to_string_accessor(series)[_start:_n]

    def starts_with(self, series, pat):
        return self.to_string_accessor(series).startswith(pat)

    def ends_with(self, series, pat):
        return self.to_string_accessor(series).endswith(pat)

    def contains(self, series, pat):
        return self.to_string_accessor(series).contains(pat)

    def char(self, series, _n):
        return self.to_string_accessor(series)[_n]

    @staticmethod
    def unicode(series):
        raise NotImplemented

    @staticmethod
    def exact(series, pat):
        return series == pat

    # dates
    def year(self, series, format):
        """
        Extract the year from a series of dates
        :param series:
        :param format: "%Y-%m-%d HH:mm:ss"
        :return:
        """
        return self.to_datetime(series, format=format).dt.year

    def month(self, series, format):
        """
        Extract the month from a series of dates
        :param series:
        :param format: "%Y-%m-%d HH:mm:ss"
        :return:
        """
        return self.to_datetime(series, format=format).dt.month

    def day(self, series, format):
        """
        Extract the day from a series of dates
        :param series:
        :param format: "%Y-%m-%d HH:mm:ss"
        :return:
        """
        return self.to_datetime(series, format=format).dt.day

    def hour(self, series, format):
        """
        Extract the hour from a series of dates
        :param series:
        :param format: "%Y-%m-%d HH:mm:ss"
        :return:
        """
        return self.to_datetime(series, format=format).dt.hour

    def minute(self, series, format):
        """
        Extract the minute from a series of dates
        :param series:
        :param format: "%Y-%m-%d HH:mm:ss"
        :return:
        """
        return self.to_datetime(series, format=format).dt.minute

    def second(self, series, format):
        """
        Extract the second from a series of dates
        :param series:
        :param format: "%Y-%m-%d HH:mm:ss"
        :return:
        """
        return self.to_datetime(series, format=format).dt.second

    def weekday(self, series, format):
        """
        Extract the weekday from a series of dates
        :param series:
        :param format: "%Y-%m-%d HH:mm:ss"
        :return:
        """
        return self.to_datetime(series, format=format).dt.weekday

    @staticmethod
    def format_date(self, current_format=None, output_format=None):
        raise NotImplemented

    @staticmethod
    def td_between(self, value=None, date_format=None):
        raise NotImplemented

    def years_between(self, series, value=None, date_format=None):
        return self.td_between(series, value, date_format).dt.days / 365.25

    def months_between(self, series, value=None, date_format=None):
        return self.td_between(series, value, date_format).dt.days / 30.436875

    def days_between(self, series, value=None, date_format=None):
        return self.td_between(series, value, date_format).dt.days

    def hours_between(self, series, value=None, date_format=None):
        series = self.td_between(series, value, date_format)
        return series.dt.days * 24.0 + series.dt.seconds / 3600.0

    def minutes_between(self, series, value=None, date_format=None):
        series = self.td_between(series, value, date_format)
        return series.dt.days * 1440.0 + series.dt.seconds / 60.0

    def seconds_between(self, series, value=None, date_format=None):
        series = self.td_between(series, value, date_format)
        return series.dt.days * 86400 + series.dt.seconds

    def domain(self, series):
        import url_parser
        return self.to_string(series).map(lambda v: url_parser.parse_url(v)["domain"], na_action=None)

    def top_domain(self, series):
        import url_parser
        return self.to_string(series).map(lambda v: url_parser.parse_url(v)["top_domain"], na_action=None)

    def sub_domain(self, series):
        import url_parser
        return self.to_string(series).map(lambda v: url_parser.parse_url(v)["sub_domain"], na_action=None)

    def url_scheme(self, series):
        import url_parser
        return self.to_string(series).map(lambda v: url_parser.parse_url(v)["protocol"], na_action=None)

    def url_path(self, series):
        import url_parser
        return self.to_string(series).map(lambda v: url_parser.parse_url(v)["path"], na_action=None)

    def url_file(self, series):
        import url_parser
        return self.to_string(series).map(lambda v: url_parser.parse_url(v)["file"], na_action=None)

    def url_query(self, series):
        import url_parser
        return self.to_string(series).map(lambda v: url_parser.parse_url(v)["query"], na_action=None)

    def url_fragment(self, series):
        import url_parser
        return self.to_string(series).map(lambda v: url_parser.parse_url(v)["fragment"], na_action=None)

    def host(self, series):
        import url_parser
        return self.to_string(series).map(lambda v: url_parser.parse_url(v)["host"], na_action=None)

    def port(self, series):
        import url_parser
        return self.to_string(series).map(lambda v: url_parser.parse_url(v)["port"], na_action=None)

    def email_username(self, series):
        return self.to_string_accessor(series).split('@').str[0]

    def email_domain(self, series):
        return self.to_string_accessor(series).split('@').str[1]


    def date_formats(self, series):
        import pydateinfer
        return series.map(lambda v: pydateinfer.infer([v]))
