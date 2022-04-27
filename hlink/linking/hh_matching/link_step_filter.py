# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import hlink.linking.core.comparison_feature as comparison_feature_core
import hlink.linking.core.comparison as comparison_core

from hlink.linking.link_step import LinkStep


class LinkStepFilter(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "filter",
            input_table_names=["hh_blocked_matches"],
            output_table_names=["hh_potential_matches"],
        )

    def _run(self):
        # self.task.spark.sql("set spark.sql.shuffle.partitions=4000")
        config = self.task.link_run.config

        # establish empty table context dict to pass to SQL template
        t_ctx = {}
        t_ctx["id_col"] = config["id_column"]
        # get comparison_features
        if config.get("hh_comparisons", False):
            t_ctx["matching_clause"] = comparison_core.generate_comparisons(
                config["hh_comparisons"],
                config["comparison_features"],
                config["id_column"],
            )

            comps = comparison_core.get_comparison_leaves(config["hh_comparisons"])
            comp_feature_names = [c["feature_name"] for c in comps]

            t_ctx["feature_columns"] = [
                comparison_feature_core.generate_comparison_feature(
                    f, config["id_column"], include_as=True
                )
                for f in config["comparison_features"]
                if f["alias"] in comp_feature_names
            ]

            self.task.run_register_sql(
                "hh_potential_matches", t_ctx=t_ctx, persist=True
            )

        else:
            self.task.run_register_python(
                "hh_potential_matches",
                lambda: self.task.spark.table("hh_blocked_matches"),
                persist=True,
            )

        self.task.spark.sql("set spark.sql.shuffle.partitions=200")

        print(
            "Potential matches from households which meet hh_comparsions thresholds have been saved to table 'hh_potential_matches'."
        )
