# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.sql.functions import array, explode, col

import hlink.linking.core.comparison as comparison_core
from . import _helpers as matching_helpers

from hlink.linking.link_step import LinkStep


class LinkStepExplode(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "explode",
            input_table_names=["prepped_df_a", "prepped_df_b"],
            output_table_names=["exploded_df_a", "exploded_df_b"],
        )

    def _run(self):
        config = self.task.link_run.config
        # filter the universe of potential matches before exploding
        t_ctx = {}
        universe_conf = config.get("potential_matches_universe", [])
        t_ctx["universe_exprs"] = [
            conf_entry["expression"] for conf_entry in universe_conf
        ]
        for suffix in ("a", "b"):
            t_ctx["prepped_df"] = f"prepped_df_{suffix}"
            output_table_name = f"match_universe_df_{suffix}"
            self.task.run_register_sql(
                output_table_name,
                template="potential_matches_universe",
                t_ctx=t_ctx,
                persist=True,
            )

        # self.spark.sql("set spark.sql.shuffle.partitions=4000")
        blocking = matching_helpers.get_blocking(config)

        self.task.run_register_python(
            name="exploded_df_a",
            func=lambda: self._explode(
                df=self.task.spark.table("match_universe_df_a"),
                comparisons=config["comparisons"],
                comparison_features=config["comparison_features"],
                blocking=blocking,
                id_column=config["id_column"],
                is_a=True,
            ),
        )
        self.task.run_register_python(
            name="exploded_df_b",
            func=lambda: self._explode(
                df=self.task.spark.table("match_universe_df_b"),
                comparisons=config["comparisons"],
                comparison_features=config["comparison_features"],
                blocking=blocking,
                id_column=config["id_column"],
                is_a=False,
            ),
        )

    def _explode(self, df, comparisons, comparison_features, blocking, id_column, is_a):

        # comp_feature_names, dist_features_to_run, feature_columns = comparison_core.get_feature_specs_from_comp(
        #     comparisons, comparison_features
        # )
        feature_columns = []
        if comparisons:
            comps = comparison_core.get_comparison_leaves(comparisons)
            comparison_feature_names = [c["feature_name"] for c in comps]
            comparison_features_to_run = [
                c for c in comparison_features if c["alias"] in comparison_feature_names
            ]
            for c in comparison_features_to_run:
                if c.get("column_name", False):
                    feature_columns.append(c["column_name"])
                elif c.get("column_names", False):
                    feature_columns += c["column_names"]

        exploded_df = df

        blocking_columns = [bc["column_name"] for bc in blocking]

        all_column_names = set(blocking_columns + feature_columns + [id_column])

        all_exploding_columns = [bc for bc in blocking if bc.get("explode", False)]

        for exploding_column in all_exploding_columns:
            exploding_column_name = exploding_column["column_name"]
            if exploding_column.get("expand_length", False):
                expand_length = exploding_column["expand_length"]
                derived_from_column = exploding_column["derived_from"]
                explode_selects = [
                    explode(self._expand(derived_from_column, expand_length)).alias(
                        exploding_column_name
                    )
                    if exploding_column_name == column
                    else column
                    for column in all_column_names
                ]
            else:
                explode_selects = [
                    explode(col(exploding_column_name)).alias(exploding_column_name)
                    if exploding_column_name == c
                    else c
                    for c in all_column_names
                ]
            if "dataset" in exploding_column:
                derived_from_column = exploding_column["derived_from"]
                explode_selects_with_derived_column = [
                    col(derived_from_column).alias(exploding_column_name)
                    if exploding_column_name == column
                    else column
                    for column in all_column_names
                ]
                if exploding_column["dataset"] == "a":
                    exploded_df = (
                        exploded_df.select(explode_selects)
                        if is_a
                        else exploded_df.select(explode_selects_with_derived_column)
                    )
                elif exploding_column["dataset"] == "b":
                    exploded_df = (
                        exploded_df.select(explode_selects)
                        if not (is_a)
                        else exploded_df.select(explode_selects_with_derived_column)
                    )
            else:
                exploded_df = exploded_df.select(explode_selects)
        return exploded_df

    def _expand(self, column_name, expand_length):
        return array(
            [
                col(column_name).cast("int") + i
                for i in range(-expand_length, expand_length + 1)
            ]
        )
