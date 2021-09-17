import datetime
import numpy as np
import pandas as pd

def json_converter(obj):
    """
    Custom converter to be used with json.dumps
    :param obj:
    :return:
    """

    if not pd.isnull(obj):
        if isinstance(obj, datetime.datetime):
            # return obj.strftime('%Y-%m-%d %H:%M:%S')
            return obj.isoformat()

        elif isinstance(obj, datetime.date):
            # return obj.strftime('%Y-%m-%d')
            return obj.isoformat()

        elif isinstance(obj, (np.generic,)):
            return np.asscalar(obj)
