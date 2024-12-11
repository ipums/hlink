# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from hlink.linking.core.model_metrics import mcc, precision, recall


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


def test_recall_example() -> None:
    tp = 3112
    fn = 1134

    recall_score = recall(tp, fn)
    assert (
        abs(recall_score - 0.7329251) < 0.0001
    ), "expected recall to be near 0.7329251"
