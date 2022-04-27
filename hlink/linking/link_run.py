# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pathlib import Path
import pandas as pd

from hlink.linking.preprocessing import Preprocessing
from hlink.linking.model_exploration import ModelExploration
from hlink.linking.training import Training
from hlink.linking.matching import Matching
from hlink.linking.reporting import Reporting
from hlink.linking.hh_model_exploration import HHModelExploration
from hlink.linking.hh_training import HHTraining
from hlink.linking.hh_matching import HHMatching
from hlink.linking.table import Table

table_definitions_file = Path(__file__).parent / "table_definitions.csv"

link_task_choices = {
    "preprocessing": Preprocessing,
    "training": Training,
    "matching": Matching,
    "hh_training": HHTraining,
    "hh_matching": HHMatching,
    "model_exploration": ModelExploration,
    "hh_model_exploration": HHModelExploration,
    "reporting": Reporting,
}


class LinkRun:
    """A link run, which manages link tasks, spark tables, and related settings.

    A link run has attributes for each link task in `link_task_choices`. These can
    be accessed like normal attributes with dot notation or with the `get_task()`
    method. The link run also has a `known_tables` attribute which lists tables
    commonly generated during linking and their descriptions. See the `get_table()`
    function for a way to get access to `Table` objects by passing a string name.

    The `use_preexisting_tables` and `print_sql` attributes are boolean settings flags.

    The `trained_models` dictionary is used for communication between the matching
    and training link tasks and the household matching and household training link
    tasks.
    """

    def __init__(self, spark, config, use_preexisting_tables=True, print_sql=False):
        self.spark = spark
        self.config = config
        self.use_preexisting_tables = use_preexisting_tables
        self.print_sql = print_sql

        self.trained_models = {}

        for task_name in link_task_choices:
            link_task = link_task_choices[task_name](self)
            setattr(self, task_name, link_task)

        self._init_tables()

    def _init_tables(self):
        """Initialize `self.known_tables` from the contents of `table_definitions_file`."""
        table_defs = pd.read_csv(table_definitions_file)
        tables = []

        for table_def in table_defs.itertuples():
            hide = table_def.hide != 0
            tables.append(Table(self.spark, table_def.name, table_def.desc, hide))

        self.known_tables = {table.name: table for table in tables}

    def get_task(self, task_name: str):
        """Get a link task attribute of the link run by name.

        If you have the string name of a link task and want the task itself,
        use this method instead of something like `getattr()`.

        Args:
            task_name (str): the name of the link task

        Raises:
            AttributeError: if `task_name` is not the name of a link task on the link run

        Returns:
            LinkTask: the requested link task
        """
        if task_name in link_task_choices:
            return getattr(self, task_name)
        else:
            raise AttributeError(f"LinkRun has no task named '{task_name}'")

    def get_table(self, table_name: str):
        """Get a `Table` by name.

        If the table is in `self.known_tables`, return it. Otherwise, return a new
        table. This method is infallible, so it will always return a table. The
        returned table may or may not exist in spark.

        Args:
            table_name (str): the name of the table to retrieve

        Returns:
            Table: the requested table
        """
        if table_name in self.known_tables:
            return self.known_tables[table_name]
        return Table(self.spark, table_name, "Unknown table", hide=True)

    def drop_temp_tables(self):
        """Delete all temporary spark tables."""
        all_tables = self.spark.catalog.listTables()
        temp_tables = filter((lambda table: table.tableType == "TEMPORARY"), all_tables)

        for table in temp_tables:
            print(f"Dropping {table.name}")
            self.spark.catalog.dropTempView(table.name)
