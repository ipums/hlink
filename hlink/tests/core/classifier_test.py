import pytest

from hlink.linking.core.classifier import choose_classifier

try:
    import xgboost  # noqa: F401
except ModuleNotFoundError:
    xgboost_available = False
else:
    xgboost_available = True


@pytest.mark.skipif(not xgboost_available, reason="requires the xgboost library")
def test_choose_classifier_supports_xgboost():
    """
    If the xgboost module is installed, then choose_classifier() supports a model
    type of "xgboost".
    """
    params = {
        "max_depth": 2,
        "eta": 0.5,
    }
    classifier, _post_transformer = choose_classifier("xgboost", params, "match")
    assert classifier.getLabelCol() == "match"
