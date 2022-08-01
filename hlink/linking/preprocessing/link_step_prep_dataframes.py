# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import logging
from pyspark.sql.functions import col

import hlink.linking.core.column_mapping as column_mapping_core
import hlink.linking.core.substitutions as substitutions_core
import hlink.linking.core.transforms as transforms_core
from hlink.linking.util import spark_shuffle_partitions_heuristic

from hlink.linking.link_step import LinkStep


class LinkStepPrepDataframes(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "prepare dataframes",
            input_table_names=["raw_df_a", "raw_df_b"],
            output_table_names=["prepped_df_a", "prepped_df_b"],
        )

    def _run(self):
        config = self.task.link_run.config

        dataset_size_a = self.task.spark.table("raw_df_a").count()
        dataset_size_b = self.task.spark.table("raw_df_b").count()
        dataset_size_max = max(dataset_size_a, dataset_size_b)
        num_partitions = spark_shuffle_partitions_heuristic(dataset_size_max)
        self.task.spark.sql(f"set spark.sql.shuffle.partitions={num_partitions}")
        logging.info(
            f"Dataset sizes are A={dataset_size_a}, B={dataset_size_b}, so set Spark partitions to {num_partitions} for this step"
        )

        substitution_columns = config.get("substitution_columns", [])
        feature_selections = config.get("feature_selections", [])

        self.task.run_register_python(
            name="prepped_df_a",
            func=lambda: self._prep_dataframe(
                self.task.spark.table("raw_df_a"),
                config["column_mappings"],
                substitution_columns,
                feature_selections,
                True,
                config["id_column"],
            ),
            persist=True,
        )
        self.task.run_register_python(
            name="prepped_df_b",
            func=lambda: self._prep_dataframe(
                self.task.spark.table("raw_df_b"),
                config["column_mappings"],
                substitution_columns,
                feature_selections,
                False,
                config["id_column"],
            ),
            persist=True,
        )

        self.task.spark.sql("set spark.sql.shuffle.partitions=200")

    # Create a function to correctly map and select the columns from the data frames
    def _prep_dataframe(
        self,
        df,
        column_definitions,
        substitution_columns,
        feature_selections,
        is_a,
        id_column,
    ):
        """
        Returns a new dataframe after having selected the given columns out with appropriate
        transformations and substitutions.

        Parameters
        ----------
        df: dataframe to operate on
        column_definitions: config array of columns to pull out and transforms to apply
        substitution_columns: config array of substitutions to apply to the selected columns
        id_column: unique id column for a record

        Returns
        ----------
        New dataframe with the operations having been applied.
        """
        df_selected = df
        spark = self.task.spark
        column_selects = [col(id_column)]
        if column_definitions and isinstance(column_definitions[0], list):
            print(
                "DEPRECATION WARNING: The config value 'column_mappings' is no longer a nested (double) array and is now an array of objects. Please change your config for future releases."
            )
            flat_column_mappings = [
                item for sublist in column_definitions for item in sublist
            ]
        else:
            flat_column_mappings = column_definitions

        for column_mapping in flat_column_mappings:
            df_selected, column_selects = column_mapping_core.select_column_mapping(
                column_mapping, df_selected, is_a, column_selects
            )

        df_selected = df_selected.select(column_selects)

        df_selected = substitutions_core.generate_substitutions(
            spark, df_selected, substitution_columns
        )

        df_selected = transforms_core.generate_transforms(
            spark, df_selected, feature_selections, self.task, is_a, id_column
        )
        return df_selected
