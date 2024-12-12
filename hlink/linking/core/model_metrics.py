# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
import math


def f_measure(true_pos: int, false_pos: int, false_neg: int) -> float:
    denominator = 2 * true_pos + false_pos + false_neg
    if denominator == 0:
        return math.nan
    return 2 * true_pos / denominator


def mcc(true_pos: int, true_neg: int, false_pos: int, false_neg: int) -> float:
    """
    Given the counts of true positives (true_pos), true negatives (true_neg), false
    positives (false_pos), and false negatives (false_neg) for a model run, compute the
    Matthews Correlation Coefficient (MCC).
    """
    if (
        math.sqrt(
            (true_pos + false_pos)
            * (true_pos + false_neg)
            * (true_neg + false_pos)
            * (true_neg + false_neg)
        )
    ) != 0:
        mcc = ((true_pos * true_neg) - (false_pos * false_neg)) / (
            math.sqrt(
                (true_pos + false_pos)
                * (true_pos + false_neg)
                * (true_neg + false_pos)
                * (true_neg + false_neg)
            )
        )
    else:
        mcc = 0
    return mcc


def precision(true_pos: int, false_pos: int) -> float:
    if (true_pos + false_pos) == 0:
        precision = math.nan
    else:
        precision = true_pos / (true_pos + false_pos)

    return precision


def recall(true_pos: int, false_neg: int) -> float:
    if (true_pos + false_neg) == 0:
        recall = math.nan
    else:
        recall = true_pos / (true_pos + false_neg)

    return recall
