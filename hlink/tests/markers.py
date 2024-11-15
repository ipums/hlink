import pytest

try:
    import xgboost  # noqa: F401
except ModuleNotFoundError:
    xgboost_available = False
else:
    xgboost_available = True

requires_xgboost = pytest.mark.skipif(
    not xgboost_available, reason="requires the xgboost library"
)
"""For tests which require the xgboost library. This checks whether xgboost is
installed and skips the test if it is not."""
