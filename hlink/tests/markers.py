# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

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
