# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
from typing import List


class LinkStep:
    def __init__(
        self,
        task,
        desc: str,
        *,
        input_table_names: List[str] = [],
        output_table_names: List[str] = [],
        input_model_names: List[str] = [],
        output_model_names: List[str] = [],
    ):
        self.task = task
        self.desc = desc
        self.input_table_names = input_table_names
        self.output_table_names = output_table_names
        self.input_model_names = input_model_names
        self.output_model_names = output_model_names

    def find_missing_input_table_names(self):
        tables = map(self.task.link_run.get_table, self.input_table_names)
        missing_tables = filter((lambda table: not table.exists()), tables)
        return [table.name for table in missing_tables]

    def run(self):
        missing_table_names = self.find_missing_input_table_names()
        if len(missing_table_names) > 0:
            missing_names_str = ", ".join(missing_table_names)
            raise RuntimeError(
                f"Missing input tables required for link step '{self}': {missing_names_str}"
            )

        self._run()

    def _run(self):
        """Run the link step.

        This abstract method must be implemented by concrete subclasses. It is
        wrapped by the `run()` method, which makes some additional quick checks.
        """
        raise NotImplementedError()

    def __str__(self):
        return self.desc
