# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from hlink.linking.core.classifier import choose_classifier
from hlink.tests.markers import requires_lightgbm, requires_xgboost


@requires_lightgbm
def test_choose_classifier_supports_lightgbm() -> None:
    params = {
        "maxDepth": 7,
        "numIterations": 5,
    }

    classifier, _post_transformer = choose_classifier("lightgbm", params, "match")
    assert classifier.getLabelCol() == "match"


@requires_xgboost
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
