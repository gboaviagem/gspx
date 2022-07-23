"""General utilities."""

import numpy as np
import operator
from scipy.stats import ks_2samp


def ks2(y_true, y_score):
    """Compute the Kolmogorov-Smirnov."""
    y_pred0 = y_score[y_true == 0]
    y_pred1 = y_score[y_true == 1]
    ks, _ = ks_2samp(y_pred0, y_pred1)
    return ks


def best_features(df, y, nbest=4, thres=0.3):
    """Find the features less correlated and with highest KS."""
    arr = df.values
    corr = df.corr() - np.eye(df.shape[1])
    all_cols = df.columns.tolist()

    _, ncols = arr.shape
    ks_ = {}
    for c in range(ncols):
        ks_[all_cols[c]] = ks2(y, arr[:, c])
    ks_ = dict(sorted(
        ks_.items(),
        key=operator.itemgetter(1),
        reverse=True))
    cols, _ = zip(*ks_.items())
    cols = list(cols)

    nbest = 4
    found = False
    best_cols = cols[:nbest]
    [cols.remove(c) for c in best_cols]
    while not found:
        this_corr = corr.loc[best_cols, best_cols]
        above_thres = (this_corr.abs() > thres).any()
        t = dict(reversed(list(above_thres.to_dict().items())))
        for k, v in t.items():
            if v is True:
                best_cols.remove(k)
                break
        if len(best_cols) == nbest:
            found = True
        else:
            while len(best_cols) < nbest:
                best_cols.append(cols[0])
                cols.remove(cols[0])
            if len(cols) == 0:
                break
    assert found is True, (
        f"Not able to find {nbest} columns with correlation "
        f"less than {thres}.")
    return best_cols
