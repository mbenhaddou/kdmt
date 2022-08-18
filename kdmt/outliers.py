import numpy as np
from scipy import stats


def mad_method(df, variable_name,threshold=4):
    #Takes two parameters: dataframe & variable of interest as string

    med = np.median(df[variable_name], axis = 0)
    mad = np.abs(stats.median_abs_deviation(df[variable_name]))
    threshold = threshold
    outlier = []

    for i, v in enumerate(df.loc[:,variable_name]):
        t = (v-med)/mad
        if t > threshold:
            outlier.append(i)
        else:
            continue
    return outlier




def z_score_method(df, variable_name,threshold=3):
    #Takes two parameters: dataframe & variable of interest as string

    z = np.abs(stats.zscore(df[variable_name]))
    threshold = threshold
    outlier = []

    for i, v in enumerate(z):
        if v > threshold:
            outlier.append(i)
        else:
            continue
    return outlier


# Tukey's method
def tukeys_method(df, variable):
    # Takes two parameters: dataframe & variable of interest as string
    q1 = df[variable].quantile(0.25)
    q3 = df[variable].quantile(0.75)
    iqr = q3 - q1
    inner_fence = 1.5 * iqr
    outer_fence = 3 * iqr

    # inner fence lower and upper end
    inner_fence_le = q1 - inner_fence
    inner_fence_ue = q3 + inner_fence

    # outer fence lower and upper end
    outer_fence_le = q1 - outer_fence
    outer_fence_ue = q3 + outer_fence

    outliers_prob = []
    outliers_poss = []
    for index, x in enumerate(df[variable]):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)
    for index, x in enumerate(df[variable]):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)
    return outliers_prob, outliers_poss

