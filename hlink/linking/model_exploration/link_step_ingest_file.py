# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from hlink.linking.link_step import LinkStep


class LinkStepIngestFile(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "ingest file",
            input_table_names=[],
            output_table_names=[f"{task.table_prefix}training_data"],
        )

    def _run(self):
        self.task.run_register_python(
            f"{self.task.table_prefix}training_data",
            lambda: self.task.spark.read.csv(
                self.task.link_run.config[f"{self.task.training_conf}"]["dataset"],
                header=True,
                inferSchema=True,
            ),
            persist=True,
        )
