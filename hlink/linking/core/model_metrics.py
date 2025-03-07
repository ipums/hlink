# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
"""
Metrics for evaluating the performance of a machine learning model. These
metrics operate on the "confusion matrix", which contains the four counts of
true positives, true negatives, false positives, and false negatives.
Throughout this module, we use the abbreviations true_pos, true_neg, false_pos,
and false_neg for these confusion matrix values.

All of these functions return math.nan in cases where they are not well-defined,
such as cases with division by zero.
"""

import math


def clamp(value: float, minimum: float, maximum: float) -> float:
    """
    Clamp the given value, forcing it to be between the minimum and maximum.
    """
    if minimum > maximum:
        raise ValueError("minimum is greater than maximum")

    return max(minimum, min(value, maximum))


def f_measure(true_pos: int, false_pos: int, false_neg: int) -> float:
    """
    Compute the F-measure, which is defined as the harmonic mean of precision
    and recall:

       2 * precision * recall / (precision + recall)

    Using the definitions of precision and recall, we can write this in terms of
    the confusion matrix entries as

      2 * true_pos / (2 * true_pos + false_pos + false_neg)
    """
    denominator = 2 * true_pos + false_pos + false_neg
    if denominator == 0:
        return math.nan
    return 2 * true_pos / denominator


def mcc(true_pos: int, true_neg: int, false_pos: int, false_neg: int) -> float:
    """
    Compute the Matthews Correlation Coefficient (MCC). This can be written as
    numerator / denominator, where

      numerator = true_pos * true_neg - false_pos * false_neg

    and

      denominator = sqrt(
        (true_pos + false_pos) *
        (true_pos + false_neg) *
        (true_neg + false_pos) *
        (true_neg + false_neg)
      )
    """
    denominator = math.sqrt(
        (true_pos + false_pos)
        * (true_pos + false_neg)
        * (true_neg + false_pos)
        * (true_neg + false_neg)
    )
    if denominator == 0:
        return math.nan

    numerator = true_pos * true_neg - false_pos * false_neg
    value = numerator / denominator
    return clamp(value, minimum=-1.0, maximum=1.0)


def precision(true_pos: int, false_pos: int) -> float:
    """
    Compute the precision, also known as the positive predictive value (PPV).
    This can be written in terms of the entries of the confusion matrix as

      true_pos / (true_pos + false_pos)
    """
    denominator = true_pos + false_pos
    if denominator == 0:
        return math.nan

    return true_pos / denominator


def recall(true_pos: int, false_neg: int) -> float:
    """
    Compute the recall, which can be written in terms of the entries of the
    confusion matrix as

      true_pos / (true_pos + false_neg)
    """
    denominator = true_pos + false_neg
    if denominator == 0:
        return math.nan

    return true_pos / denominator
