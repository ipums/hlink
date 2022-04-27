# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink


class Table:
    """Represents a spark table which may or may not currently exist.

    It's possible to pass table names that aren't valid spark table names to
    this class (for example, "@@@"). In this case, this class does not throw
    errors; it just treats the tables like any other spark tables that don't
    exist.
    """

    def __init__(self, spark, name: str, desc: str, hide: bool = False):
        self.spark = spark
        # User-facing name
        self.name = name
        # Name used to interact with spark
        self._name_lower = name.lower()
        self.desc = desc
        self.hide = hide

    def exists(self) -> bool:
        """Check whether the table currently exists in spark."""
        return self._name_lower in [
            table.name for table in self.spark.catalog.listTables()
        ]

    def drop(self):
        """Drop the table if it `exists()`.

        If the table doesn't exist, then don't do anything.
        """
        if self.exists():
            self.spark.sql(f"DROP TABLE {self._name_lower}")
        assert (
            not self.exists()
        ), f"table '{self.name}' has been dropped but still exists"

    def df(self):
        """Get the DataFrame of the table from spark.

        Returns:
            Option[DataFrame]: the DataFrame of the table, or None if the table doesn't exist
        """
        if self.exists():
            return self.spark.table(self._name_lower)
        return None

    def __str__(self):
        return f"Table '{self.name}' <- {self.desc}"
