# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from collections import defaultdict
import logging
from typing import Any

import hlink.linking.core.comparison_feature as comparison_feature_core
import hlink.linking.core.dist_table as dist_table_core
import hlink.linking.core.comparison as comparison_core
from hlink.linking.util import spark_shuffle_partitions_heuristic

from hlink.linking.link_step import LinkStep

logger = logging.getLogger(__name__)


def extract_or_groups_from_blocking(blocking: list[dict[str, Any]]) -> list[list[str]]:
    """
    Extract a list of "or_groups" from the blocking section of the config. Each
    blocking table may have an or_group attribute. When two or more tables have
    the same value for or_group, they belong to the same or_group and will be
    connected by ORs in the potential_matches SQL query instead of by ANDs.
    Tables without an explicit or_group belong to their own or_group.

    For example, the blocking section

    ```
    [[blocking]]
    column_name = "AGE1"
    or_group = "AGE"

    [[blocking]]
    column_name = "AGE2"
    or_group = "AGE"

    [[blocking]]
    column_name = "BPL"
    ```

    Would give the SQL condition

    ```
    (a.AGE1 = b.AGE1 OR a.AGE2 = b.AGE2) AND (a.BPL = b.BPL)
    ```

    This function returns a list of or_groups, each of which is a list of
    column names. It maintains the input order except that the implicit
    or_groups are all placed after the explicit or_groups.
    """
    or_groups: defaultdict[str | None, list[str]] = defaultdict(list)

    for blocking_table in blocking:
        column_name = blocking_table["column_name"]
        or_group = blocking_table.get("or_group")
        or_groups[or_group].append(column_name)

    implicit_or_groups = [[column_name] for column_name in or_groups.pop(None, [])]
    return list(or_groups.values()) + implicit_or_groups


class LinkStepMatch(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "match",
            input_table_names=["exploded_df_a", "exploded_df_b"],
            output_table_names=["potential_matches"],
        )

    def _run(self):
        config = self.task.link_run.config

        dataset_size_a = self.task.spark.table("exploded_df_a").count()
        dataset_size_b = self.task.spark.table("exploded_df_b").count()
        dataset_size_max = max(dataset_size_a, dataset_size_b)
        num_partitions = spark_shuffle_partitions_heuristic(dataset_size_max)
        self.task.spark.sql(f"set spark.sql.shuffle.partitions={num_partitions}")
        logger.info(
            f"Dataset sizes are A={dataset_size_a}, B={dataset_size_b}, so set Spark partitions to {num_partitions} for this step"
        )

        blocking = config["blocking"]

        t_ctx = {}
        if config.get("comparisons", False):
            if config["comparisons"] != {}:
                t_ctx["matching_clause"] = comparison_core.generate_comparisons(
                    config["comparisons"],
                    config["comparison_features"],
                    config["id_column"],
                )

        t_ctx["blocking_columns"] = extract_or_groups_from_blocking(blocking)

        blocking_exploded_columns = [
            bc["column_name"] for bc in blocking if "explode" in bc and bc["explode"]
        ]
        t_ctx["dataset_columns"] = [
            c
            for c in self.task.spark.table("exploded_df_a").columns
            if c not in blocking_exploded_columns
        ]

        # comp_feature_names, dist_features_to_run, features_to_run = comparison_core.get_feature_specs_from_comp(
        #     config["comparisons"], config["comparison_features"]
        # )
        if config.get("comparisons", {}):
            comps = comparison_core.get_comparison_leaves(config["comparisons"])
            comp_feature_names = [c["feature_name"] for c in comps]

            t_ctx["feature_columns"] = [
                comparison_feature_core.generate_comparison_feature(
                    f, config["id_column"], include_as=True
                )
                for f in config["comparison_features"]
                if f["alias"] in comp_feature_names
            ]

            dist_feature_names = [
                c["alias"]
                for c in config["comparison_features"]
                if c["comparison_type"] in ["geo_distance"]
            ]
            dist_features_to_run = [
                c["feature_name"]
                for c in comps
                if c["feature_name"] in dist_feature_names
            ]

            if dist_features_to_run:
                dist_comps = [
                    c
                    for c in config["comparison_features"]
                    if c["alias"] in dist_features_to_run
                ]
                (
                    t_ctx["distance_table"],
                    dist_tables,
                ) = dist_table_core.register_dist_tables_and_create_sql(
                    self.task, dist_comps
                )
                if dist_tables:
                    t_ctx["broadcast_hints"] = dist_table_core.get_broadcast_hint(
                        dist_tables
                    )

        if config.get("streamline_potential_match_generation", False):
            t_ctx["dataset_columns"] = [config["id_column"]]
        try:
            self.task.run_register_sql("potential_matches", t_ctx=t_ctx, persist=True)
        finally:
            self.task.spark.sql("set spark.sql.shuffle.partitions=200")
