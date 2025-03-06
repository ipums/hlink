# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
import math

from hypothesis import assume, example, given
import hypothesis.strategies as st
import pytest

from hlink.linking.core.model_metrics import clamp, f_measure, mcc, precision, recall

NonNegativeInt = st.integers(min_value=0)
NegativeInt = st.integers(max_value=-1)
BoundedFloat = st.floats(allow_infinity=False, allow_nan=False)


def test_f_measure_example() -> None:
    true_pos = 3112
    false_pos = 205
    false_neg = 1134

    f_measure_score = f_measure(true_pos, false_pos, false_neg)
    assert (
        abs(f_measure_score - 0.8229539) < 0.0001
    ), "expected F-measure to be near 0.8229539"


def test_f_measure_all_zeroes() -> None:
    """
    When true_pos, false_pos, and false_neg are all 0, f_measure is undefined and
    returns NaN to indicate this.
    """
    f_measure_score = f_measure(0, 0, 0)
    assert math.isnan(f_measure_score)


@given(true_pos=NonNegativeInt, false_pos=NonNegativeInt, false_neg=NonNegativeInt)
def test_f_measure_between_0_and_1(
    true_pos: int, false_pos: int, false_neg: int
) -> None:
    assume(true_pos + false_pos + false_neg > 0)
    f_measure_score = f_measure(true_pos, false_pos, false_neg)
    assert 0.0 <= f_measure_score <= 1.0


@given(true_pos=NonNegativeInt, false_pos=NonNegativeInt, false_neg=NonNegativeInt)
def test_f_measure_is_harmonic_mean_of_precision_and_recall(
    true_pos: int, false_pos: int, false_neg: int
) -> None:
    precision_score = precision(true_pos, false_pos)
    recall_score = recall(true_pos, false_neg)

    assume(precision_score + recall_score > 0)

    f_measure_score = f_measure(true_pos, false_pos, false_neg)
    harmonic_mean = (
        2 * precision_score * recall_score / (precision_score + recall_score)
    )

    assert (
        abs(harmonic_mean - f_measure_score) < 0.0001
    ), f"harmonic mean is {harmonic_mean}, but F-measure is {f_measure_score}"


def test_mcc_example() -> None:
    true_pos = 3112
    false_pos = 205
    false_neg = 1134
    true_neg = 33259

    mcc_score = mcc(true_pos, true_neg, false_pos, false_neg)
    assert abs(mcc_score - 0.8111208) < 0.0001, "expected MCC to be near 0.8111208"


@given(
    true_pos=NonNegativeInt,
    true_neg=NonNegativeInt,
    false_pos=NonNegativeInt,
    false_neg=NonNegativeInt,
)
@example(true_pos=0, true_neg=0, false_pos=51, false_neg=2_070_366_244_862_899).via(
    "issue #187"
)
def test_mcc_is_between_negative_1_and_positive_1(
    true_pos: int, true_neg: int, false_pos: int, false_neg: int
) -> None:
    """
    Under "normal circumstances", where the denominator of the Matthews Correlation
    Coefficient isn't 0, its range is the interval [-1, 1].
    """
    assume(true_pos + false_pos > 0)
    assume(true_pos + false_neg > 0)
    assume(true_neg + false_pos > 0)
    assume(true_neg + false_neg > 0)

    mcc_score = mcc(true_pos, true_neg, false_pos, false_neg)
    assert -1.0 <= mcc_score <= 1.0


@pytest.mark.parametrize(
    "true_pos,true_neg,false_pos,false_neg",
    [(0, 0, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)],
)
def test_mcc_denom_zero(
    true_pos: int, true_neg: int, false_pos: int, false_neg: int
) -> None:
    """
    If the denominator of MCC is 0, it's not well-defined, and it returns NaN. This
    can happen in a variety of situations if at least 2 of the inputs are 0.
    """
    mcc_score = mcc(true_pos, true_neg, false_pos, false_neg)
    assert math.isnan(mcc_score)


def test_precision_example() -> None:
    true_pos = 3112
    false_pos = 205

    precision_score = precision(true_pos, false_pos)
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
    true_pos = 3112
    false_neg = 1134

    recall_score = recall(true_pos, false_neg)
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


def test_clamp_in_between() -> None:
    assert clamp(15, 10, 20) == 15


def test_clamp_less_than_minimum() -> None:
    assert clamp(1, 5, 10) == 5


def test_clamp_greater_than_maximum() -> None:
    assert clamp(200, 10, 30) == 30


@given(x=BoundedFloat, y=BoundedFloat, z=BoundedFloat)
def test_clamp_lies_within_bounds(x: float, y: float, z: float) -> None:
    assume(y <= z)
    assert y <= clamp(x, y, z) <= z


@given(x=BoundedFloat, y=BoundedFloat, z=BoundedFloat)
def test_clamp_error_when_minimum_greater_than_maximum(
    x: float, y: float, z: float
) -> None:
    assume(y > z)
    with pytest.raises(ValueError, match="minimum is greater than maximum"):
        clamp(x, y, z)
