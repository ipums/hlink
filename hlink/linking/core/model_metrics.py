# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
import math

import numpy as np


def f_measure(true_pos: int, false_pos: int, false_neg: int) -> float:
    return 2 * true_pos / (2 * true_pos + false_pos + false_neg)


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Given the counts of true positives (tp), true negatives (tn), false
    positives (fp), and false negatives (fn) for a model run, compute the
    Matthews Correlation Coefficient (MCC).
    """
    if (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) != 0:
        mcc = ((tp * tn) - (fp * fn)) / (
            math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )
    else:
        mcc = 0
    return mcc


def precision(tp: int, fp: int) -> float:
    if (tp + fp) == 0:
        precision = np.nan
    else:
        precision = tp / (tp + fp)

    return precision


def recall(tp: int, fn: int) -> float:
    if (tp + fn) == 0:
        recall = np.nan
    else:
        recall = tp / (tp + fn)

    return recall
