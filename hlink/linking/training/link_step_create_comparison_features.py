# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import hlink.linking.core.comparison_feature as comparison_feature_core
import hlink.linking.core.dist_table as dist_table_core

from hlink.linking.link_step import LinkStep


class LinkStepCreateComparisonFeatures(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "create comparison features",
            input_table_names=[
                "prepped_df_a",
                "prepped_df_b",
                f"{task.table_prefix}training_data",
            ],
            output_table_names=[f"{task.table_prefix}training_features"],
        )

    def _run(self):
        self.task.spark.sql("set spark.sql.shuffle.partitions=200")
        self.__create_training_features()

    def __create_training_features(self):
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        if config[training_conf].get("use_training_data_features", False):
            return self.task.run_register_python(
                f"{table_prefix}training_features",
                lambda: self.task.spark.table(f"{table_prefix}training_data"),
                persist=True,
            )
        id_col = config["id_column"]
        dep_var = config[training_conf]["dependent_var"]
        if training_conf == "hh_training":
            hh_col = config[training_conf].get("hh_col", "serialp")
            tdl = self.task.spark.sql(
                f"""SELECT
                                    td.{id_col}_a,
                                    td.{id_col}_b,
                                    td.{dep_var},
                                    pdfa.{hh_col} as {hh_col}_a,
                                    pdfb.{hh_col} as {hh_col}_b
                                    from
                                    {table_prefix}training_data td
                                    left join
                                    prepped_df_a pdfa
                                    on pdfa.{id_col} = td.{id_col}_a
                                    left join
                                    prepped_df_b pdfb
                                    on pdfb.{id_col} = td.{id_col}_b
                                """
            )
        else:
            tdl = self.task.spark.table(f"{table_prefix}training_data").select(
                f"{id_col}_a", f"{id_col}_b", dep_var
            )
        self.task.run_register_python(f"{table_prefix}training_data_ids", lambda: tdl)
        (
            comp_features,
            advanced_comp_features,
            hh_comp_features,
            dist_features,
        ) = comparison_feature_core.get_features(
            config, config[training_conf]["independent_vars"]
        )
        t_ctx_def = {
            "comp_features": comp_features,
            "match_feature": config[training_conf]["dependent_var"],
            "id": config["id_column"],
            "potential_matches": f"{table_prefix}training_data_ids",
        }
        join_clauses, dist_tables = dist_table_core.register_dist_tables_and_create_sql(
            self.task, dist_features
        )
        t_ctx_def["distance_table"] = join_clauses
        if len(dist_tables) > 0:
            t_ctx_def["broadcast_hints"] = dist_table_core.get_broadcast_hint(
                dist_tables
            )

        comparison_feature_core.create_feature_tables(
            self.task,
            t_ctx_def,
            advanced_comp_features,
            hh_comp_features,
            config["id_column"],
            table_name=f"{table_prefix}training_features",
        )
