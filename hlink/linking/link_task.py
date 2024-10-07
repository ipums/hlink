# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import logging
from typing import Iterable, Callable, Any
from timeit import default_timer as timer
from jinja2 import Environment, PackageLoader, ChoiceLoader
import pyspark

from hlink.errors import SparkError
from hlink.linking.link_step import LinkStep

logger = logging.getLogger(__name__)


class LinkTask:
    """Base class for link tasks.

    A link task consists of one or more `LinkStep`s and belongs to one `LinkRun`.
    The `get_steps()` function returns a list specifying which steps the task has
    and what order they should be run in.

    The `run_all_steps()` and `run_step()` functions are ways to run the link steps
    belonging to the task.

    `run_register_python()` and `run_register_sql()` are methods to be used by
    concrete subclasses for creating spark tables and performing step work.
    """

    def __init__(self, link_run, display_name: str | None = None):
        self.link_run = link_run
        loader = ChoiceLoader(
            [
                PackageLoader(self.__class__.__module__),
                PackageLoader("hlink.linking", "templates/shared"),
            ]
        )
        self.jinja_env = Environment(loader=loader)

        if display_name is None:
            self.display_name = self.__class__.__name__
        else:
            self.display_name = display_name

    @property
    def spark(self):
        return self.link_run.spark

    def __str__(self) -> str:
        return self.display_name

    def get_steps(self) -> list[LinkStep]:
        """Get a list of the steps that make up the link task.

        This abstract method must be implemented by concrete subclasses.

        Returns:
            a list of the link steps making up the link task
        """
        raise NotImplementedError()

    def run_all_steps(self) -> None:
        """Run all steps in order."""
        logger.info(f"Running all steps for task {self.display_name}")
        start_all = timer()
        for i, step in enumerate(self.get_steps()):
            print(f"Running step {i}: {step}")
            logger.info(f"Running step {i}: {step}")
            step.run()
        end_all = timer()
        elapsed_time_all = round(end_all - start_all, 2)
        print(f"Finished all in {elapsed_time_all}s")
        logger.info(f"Finished all steps in {elapsed_time_all}s")

    def run_step(self, step_num: int) -> None:
        """Run a particular step.

        Note that running steps out of order may cause errors when later steps
        depend on the work done by earlier steps and that work is not available.

        Args:
            step_num (int): the step number, used as an index into the `get_steps()` list
        """
        steps = self.get_steps()

        if step_num < 0 or step_num >= len(steps):
            steps_string = "\n\t".join(
                [f"step {i}: {step.desc}" for (i, step) in enumerate(steps)]
            )
            print(
                f"Error! Couldn't find step {step_num}. Valid steps are: \n\t{steps_string}"
            )
            return

        step = steps[step_num]
        step_string = f"step {step_num}: {step}"
        print(f"Running {step_string}")
        logger.info(f"Starting {step.task.display_name} - {step_string}")

        start = timer()
        step.run()
        end = timer()

        elapsed_time = round(end - start, 2)
        print(f"Finished {step_string} in {elapsed_time}s")
        logger.info(
            f"Finished {step.task.display_name} - {step_string} in {elapsed_time}s"
        )

    def run_register_python(
        self,
        name: str,
        func: Callable[..., pyspark.sql.dataframe.DataFrame],
        args: Iterable[Any] = [],
        persist: bool = False,
        overwrite_preexisting_tables: bool = False,
    ) -> pyspark.sql.dataframe.DataFrame:
        """Run the given Python function `func` on `args` and register the returned data
        frame with the given table `name`.
        """
        if name is not None:
            if overwrite_preexisting_tables is False:
                df = self._check_preexisting_table(name)
            else:
                df = None
            if df is None:
                df = func(*args)
                if persist:
                    df.write.mode("overwrite").saveAsTable(name)
                else:
                    df.createOrReplaceTempView(name)
                self.spark.sql(f"REFRESH TABLE {name}")
            return self.spark.table(name)
        else:
            try:
                return func(*args)
            except Exception as err:
                logger.error(err)
                raise SparkError(str(err))

    def run_register_sql(
        self,
        name: str,
        sql: str | None = None,
        template: str | None = None,
        t_ctx: dict[str, Any] = {},
        persist: bool = False,
        overwrite_preexisting_tables: bool = False,
    ) -> pyspark.sql.dataframe.DataFrame:
        """Run the given sql or template (with context) and register the returned
        data frame with the given table `name`.

        Read the table from disk instead of running sql if `use_preexisting_tables`
        is set to True on the link task's `LinkRun`.

        Persist the created table if `persist` is True.
        """

        def run_sql() -> pyspark.sql.dataframe.DataFrame:
            sql_to_run = self._get_sql(name, sql, template, t_ctx)
            if self.link_run.print_sql:
                print(sql_to_run)
            try:
                return self.spark.sql(sql_to_run)
            except Exception as err:
                print(f"Exception in Spark SQL: {sql_to_run}")

                logger.error(str(err))
                raise SparkError(str(err))

        return self.run_register_python(
            name=name,
            func=run_sql,
            persist=persist,
            overwrite_preexisting_tables=overwrite_preexisting_tables,
        )

    def _check_preexisting_table(
        self, name: str
    ) -> pyspark.sql.dataframe.DataFrame | None:
        table = self.link_run.get_table(name)
        if self.link_run.use_preexisting_tables and table.exists():
            print(f"Preexisting table: {name}")
            return table.df()
        return None

    def _get_sql(
        self, name: str, sql: str | None, template: str | None, t_ctx: dict[str, Any]
    ) -> str:
        if sql is None:
            template_file_name = template if template is not None else name
            template_path = f"{template_file_name}.sql"
            sql = self.jinja_env.get_template(f"{template_path}").render(t_ctx)
            print(f"{name} -- {template_path}")
        else:
            print(name)
        return sql
