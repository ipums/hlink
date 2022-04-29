# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import sys
from pathlib import Path

from hlink.spark.session import SparkConnection


class SparkFactory:
    """This class allows convenient creation of a spark session.

    It defines defaults for many settings that can be overwritten with the
    `set_*` functions. Each `set_*` function returns the SparkFactory for easy
    chaining of settings.

    Note that some settings values are paths. These paths should be absolute.
    This applies to derby_dir, warehouse_dir, and tmp_dir.
    """

    def __init__(self):
        spark_dir = Path("spark").resolve()
        self.derby_dir = spark_dir / "derby"
        self.warehouse_dir = spark_dir / "warehouse"
        self.tmp_dir = spark_dir / "tmp"
        self.python = sys.executable
        self.db_name = "linking"
        self.is_local = True
        self.num_cores = 4
        self.executor_memory = "10G"
        self.executor_cores = 16

    def set_derby_dir(self, derby_dir):
        self.derby_dir = derby_dir
        return self

    def set_warehouse_dir(self, warehouse_dir):
        self.warehouse_dir = warehouse_dir
        return self

    def set_tmp_dir(self, tmp_dir):
        self.tmp_dir = tmp_dir
        return self

    def set_python(self, python):
        """Set the python executable.

        Useful when you want to guarantee that remote machines are running the
        same version of python.
        """
        self.python = python
        return self

    def set_db_name(self, db_name):
        self.db_name = db_name
        return self

    def set_local(self):
        """Make a local spark connection."""
        self.is_local = True
        return self

    def set_num_cores(self, num_cores):
        self.num_cores = num_cores
        return self

    def set_executor_memory(self, executor_memory):
        self.executor_memory = executor_memory
        return self

    def set_executor_cores(self, executor_cores):
        self.executor_cores = executor_cores
        return self

    def create(self):
        spark_conn = SparkConnection(
            str(self.derby_dir),
            str(self.warehouse_dir),
            str(self.tmp_dir),
            self.python,
            self.db_name,
        )
        return spark_conn.local(self.num_cores, self.executor_memory)
