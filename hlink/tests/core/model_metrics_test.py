# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
import math

from hypothesis import assume, given
import hypothesis.strategies as st

from hlink.linking.core.model_metrics import mcc, precision, recall

NonNegativeInt = st.integers(min_value=0)


def test_mcc_example() -> None:
    tp = 3112
    fp = 205
    fn = 1134
    tn = 33259

    mcc_score = mcc(tp, tn, fp, fn)
    assert abs(mcc_score - 0.8111208) < 0.0001, "expected MCC to be near 0.8111208"


def test_precision_example() -> None:
    tp = 3112
    fp = 205

    precision_score = precision(tp, fp)
    assert (
        abs(precision_score - 0.9381972) < 0.0001
    ), "expected precision to be near 0.9381972"


@given(true_pos=NonNegativeInt, false_pos=NonNegativeInt)
def test_precision_between_0_and_1(true_pos: int, false_pos: int) -> None:
    """
    Under "normal circumstances" (there were at least some positive predictions)
    precision()'s range is the interval [0.0, 1.0].
    """
    assume(true_pos + false_pos > 0)
    precision_score = precision(true_pos, false_pos)
    assert 0.0 <= precision_score <= 1.0


def test_precision_no_positive_predictions() -> None:
    """
    When there are no positive predictions, true_pos=0 and false_pos=0, and
    precision is not well defined. In this case we return NaN.
    """
    precision_score = precision(0, 0)
    assert math.isnan(precision_score)


def test_recall_example() -> None:
    tp = 3112
    fn = 1134

    recall_score = recall(tp, fn)
    assert (
        abs(recall_score - 0.7329251) < 0.0001
    ), "expected recall to be near 0.7329251"


@given(true_pos=NonNegativeInt, false_neg=NonNegativeInt)
def test_recall_between_0_and_1(true_pos: int, false_neg: int) -> None:
    """
    Under "normal circumstances" (there is at least one true positive or false
    negative), the range of recall() is the interval [0.0, 1.0].
    """
    assume(true_pos + false_neg > 0)
    recall_score = recall(true_pos, false_neg)
    assert 0.0 <= recall_score <= 1.0


def test_recall_no_true_pos_or_false_neg() -> None:
    """
    When both true_pos and false_neg are 0, recall is not well defined, and we
    return NaN.
    """
    recall_score = recall(0, 0)
    assert math.isnan(recall_score)
