import pytest

try:
    import synapse.ml.lightgbm  # noqa: F401
except ModuleNotFoundError:
    lightgbm_available = False
else:
    lightgbm_available = True

requires_lightgbm = pytest.mark.skipif(
    not lightgbm_available, reason="requires the lightgbm library"
)
