# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pytest

try:
    import xgboost  # noqa: F401
except ModuleNotFoundError:
    xgboost_available = False
else:
    xgboost_available = True

try:
    import synapse.ml.lightgbm  # noqa: F401
except ModuleNotFoundError:
    lightgbm_available = False
else:
    lightgbm_available = True


requires_xgboost = pytest.mark.skipif(
    not xgboost_available, reason="requires the xgboost library"
)
"""For tests which require the xgboost library. This checks whether xgboost is
installed and skips the test if it is not."""

requires_lightgbm = pytest.mark.skipif(
    not lightgbm_available, reason="requires the lightgbm library"
)
