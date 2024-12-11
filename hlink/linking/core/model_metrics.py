# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
import math


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
