# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import array, explode, col

import hlink.linking.core.comparison as comparison_core
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
        blocking = config["blocking"]

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

    def _explode(
        self,
        df: DataFrame,
        comparisons: dict[str, Any],
        comparison_features: list[dict[str, Any]],
        blocking: list[dict[str, Any]],
        id_column: str,
        is_a: bool,
    ) -> DataFrame:
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
                # A special case for multi_jaro_winkler_search because it supports
                # templating. It doesn't store the column names it's going to use
                # in a column_name or column_names attribute...
                elif c.get("comparison_type") == "multi_jaro_winkler_search":
                    num_cols = c["num_cols"]
                    jw_col_template = c["jw_col_template"]
                    equal_templates = c.get("equal_and_not_null_templates", [])

                    # The comparison feature will iterate over the Cartesian product
                    # of this range with itself. But this single loop gets us all
                    # of the integers that will appear in the Cartesian product.
                    for i in range(1, num_cols + 1):
                        realized_jw_template = jw_col_template.replace("{n}", str(i))
                        realized_equal_templates = [
                            equal_template.replace("{n}", str(i))
                            for equal_template in equal_templates
                        ]

                        feature_columns.append(realized_jw_template)
                        feature_columns.extend(realized_equal_templates)

        exploded_df = df

        blocking_columns = [bc["column_name"] for bc in blocking]

        all_column_names = set(blocking_columns + feature_columns + [id_column])

        all_exploding_columns = [bc for bc in blocking if bc.get("explode", False)]

        for exploding_column in all_exploding_columns:
            exploding_column_name = exploding_column["column_name"]
            if exploding_column.get("expand_length", False):
                expand_length = exploding_column["expand_length"]
                derived_from_column = exploding_column["derived_from"]

                explode_col_expr = explode(
                    self._expand(derived_from_column, expand_length)
                )
            else:
                explode_col_expr = explode(col(exploding_column_name))

            if "dataset" in exploding_column:
                derived_from_column = exploding_column["derived_from"]
                no_explode_col_expr = col(derived_from_column)

                if exploding_column["dataset"] == "a":
                    expr = explode_col_expr if is_a else no_explode_col_expr
                    exploded_df = exploded_df.withColumn(exploding_column_name, expr)
                elif exploding_column["dataset"] == "b":
                    expr = explode_col_expr if not is_a else no_explode_col_expr
                    exploded_df = exploded_df.withColumn(exploding_column_name, expr)
            else:
                exploded_df = exploded_df.withColumn(
                    exploding_column_name, explode_col_expr
                )

        # If there are exploding columns, then select out "all_column_names".
        # Otherwise, just let all of the columns through without selecting
        # specific ones. I believe this is an artifact of a previous
        # implementation, but the tests currently enforce it. It may or may not
        # be a breaking change to remove this. We'd have to look into the
        # ramifications.
        if len(all_exploding_columns) > 0:
            exploded_df = exploded_df.select(sorted(all_column_names))

        return exploded_df

    def _expand(self, column_name: str, expand_length: int) -> Column:
        return array(
            [
                col(column_name).cast("int") + i
                for i in range(-expand_length, expand_length + 1)
            ]
        )
