# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink


class SparkError(Exception):
    """Catch any exceptions from Spark"""

    pass


class UsageError(Exception):
    """Incorrectly specified options"""

    pass


class DataError(Exception):
    """There is an issue in the source data that will cause a problem."""

    pass
