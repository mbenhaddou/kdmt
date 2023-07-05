import pandas as pd
import numpy as np
from kdmt.dates import get_datetime_format
import sys



def color_df(
    df: pd.DataFrame, color: str, names: list, axis: int = 1
):
    return df.style.apply(
        lambda x: [f"background: {color}" if (x.name in names) else "" for _ in x],
        axis=axis,
    )



MAX_DECIMALS = sys.float_info.dig - 1
INTEGER_BOUNDS = {
    'Int8': (-2**7, 2**7 - 1),
    'Int16': (-2**15, 2**15 - 1),
    'Int32': (-2**31, 2**31 - 1),
    'Int64': (-2**63, 2**63 - 1),
    'UInt8': (0, 2**8 - 1),
    'UInt16': (0, 2**16 - 1),
    'UInt32': (0, 2**32 - 1),
    'UInt64': (0, 2**64 - 1),
}


class NumericalFormatter:
    """Formatter for numerical data.

    Args:
        enforce_rounding (bool):
            Whether or not to learn what place to round to based on the data seen during ``fit``.
            If ``True``, the data returned by ``reverse_transform`` will be rounded to that place.
            Defaults to ``False``.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``.
            Defaults to ``False``.
        computer_representation (dtype):
            Accepts ``'Int8'``, ``'Int16'``, ``'Int32'``, ``'Int64'``, ``'UInt8'``, ``'UInt16'``,
            ``'UInt32'``, ``'UInt64'``, ``'Float'``.
            Defaults to ``'Float'``.
    """

    _dtype = None
    _min_value = None
    _max_value = None
    _rounding_digits = None

    def __init__(self, enforce_rounding=False, enforce_min_max_values=False,
                 computer_representation='Float'):
        self.enforce_rounding = enforce_rounding
        self.enforce_min_max_values = enforce_min_max_values
        self.computer_representation = computer_representation

    @staticmethod
    def _learn_rounding_digits(data):
        """Check if data has any decimals."""
        name = data.name
        data = np.array(data)
        roundable_data = data[~(np.isinf(data) | pd.isna(data))]

        # Doesn't contain numbers
        if len(roundable_data) == 0:
            return None

        # Doesn't contain decimal digits
        if ((roundable_data % 1) == 0).all():
            return 0

        # Try to round to fewer digits
        if (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
            for decimal in range(MAX_DECIMALS + 1):
                if (roundable_data == roundable_data.round(decimal)).all():
                    return decimal

        # Can't round, not equal after MAX_DECIMALS digits of precision
        print(
            f"No rounding scheme detected for column '{name}'."
            ' Synthetic data will not be rounded.'
        )
        return None

    def learn_format(self, column):
        """Learn the format of a column.

        Args:
            column (pandas.Series):
                Data to learn the format.
        """
        self._dtype = column.dtype
        if self.enforce_min_max_values:
            self._min_value = column.min()
            self._max_value = column.max()

        if self.enforce_rounding:
            self._rounding_digits = self._learn_rounding_digits(column)

    def format_data(self, column):
        """Format a column according to the learned format.

        Args:
            column (pd.Series):
                Data to format.

        Returns:
            numpy.ndarray:
                containing the formatted data.
        """
        column = column.copy().to_numpy()
        if self.enforce_min_max_values:
            column = column.clip(self._min_value, self._max_value)
        elif self.computer_representation != 'Float':
            min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
            column = column.clip(min_bound, max_bound)

        is_integer = np.dtype(self._dtype).kind == 'i'
        if self.enforce_rounding and self._rounding_digits is not None:
            column = column.round(self._rounding_digits)
        elif is_integer:
            column = column.round(0)

        if pd.isna(column).any() and is_integer:
            return column

        return column.astype(self._dtype)

class DatetimeFormatter:
    """Formatter for datetime data.

    Args:
        datetime_format (str):
            The strftime to use for parsing time. For more information, see
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
            If ``None`` it will attempt to learn it by itself. Defaults to ``None``.
    """

    def __init__(self, datetime_format=None):
        self.datetime_format = datetime_format

    def learn_format(self, column):
        """Learn the format of a column.

        Args:
            column (pandas.Series):
                Data to learn the format.
        """
        self._dtype = column.dtype
        if self.datetime_format is None:
            self.datetime_format = get_datetime_format(column)

    def format_data(self, column):
        """Format a column according to the learned format.

        Args:
            column (pd.Series):
                Data to format.

        Returns:
            numpy.ndarray:
                containing the formatted data.
        """
        if self.datetime_format:
            try:
                datetime_column = pd.to_datetime(column, format=self.datetime_format)
                column = datetime_column.dt.strftime(self.datetime_format)
            except ValueError:
                column = pd.to_datetime(column).dt.strftime(self.datetime_format)

        return column.astype(self._dtype)


def get_nan_component_value(row):
    """Check for NaNs in a pandas row.

    Outputs a concatenated string of the column names with NaNs.

    Args:
        row (pandas.Series):
            A pandas row.

    Returns:
        A concatenated string of the column names with NaNs.
    """
    columns_with_nans = []
    for column, value in row.items():
        if pd.isna(value):
            columns_with_nans.append(column)

    if columns_with_nans:
        return ', '.join(columns_with_nans)
    else:
        return 'None'


def compute_nans_column(table_data, list_column_names):
    """Compute a categorical column to the table_data indicating where NaNs are.

    Args:
        table_data (pandas.DataFrame):
            The table data.
        list_column_names (list):
            The list of column names to check for NaNs.

    Returns:
        A dict with the column name as key and the column indicating where NaNs are as value.
        Empty dict if there are no NaNs.
    """
    nan_column_name = '#'.join(list_column_names) + '.nan_component'
    column = table_data[list_column_names].apply(get_nan_component_value, axis=1)
    if not (column == 'None').all():
        return pd.Series(column, name=nan_column_name)

    return None


def revert_nans_columns(table_data, nan_column_name):
    """Reverts the NaNs in the table_data based on the categorical column.

    Args:
        table_data (pandas.DataFrame):
            The table data.
        nan_column (pandas.Series):
            The categorical columns indicating where the NaNs are.
    """
    combinations = table_data[nan_column_name].unique()
    for combination in combinations:
        if combination != 'None':
            column_names = [column_name.strip() for column_name in combination.split(',')]
            table_data.loc[table_data[nan_column_name] == combination, column_names] = np.nan

    return table_data.drop(columns=nan_column_name)