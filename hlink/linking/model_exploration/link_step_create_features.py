# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.ml import Pipeline

import hlink.linking.core.comparison_feature as comparison_feature_core
import hlink.linking.core.dist_table as dist_table_core
import hlink.linking.core.pipeline as pipeline_core

from hlink.linking.link_step import LinkStep


class LinkStepCreateFeatures(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "create features",
            input_table_names=[
                "prepped_df_a",
                "prepped_df_b",
                f"{task.table_prefix}training_data",
            ],
            output_table_names=[
                f"{task.table_prefix}training_features",
                f"{task.table_prefix}training_vectorized",
            ],
        )

    def _run(self):
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        self.task.spark.sql("set spark.sql.shuffle.partitions=200")
        id_col = config["id_column"]
        dep_var = config[training_conf]["dependent_var"]

        if training_conf == "hh_training":
            self.task.run_register_python(
                f"{table_prefix}training_data_ids",
                lambda: self.task.spark.table(f"{table_prefix}training_data").select(
                    f"{id_col}_a", f"{id_col}_b", "serialp_a", "serialp_b", dep_var
                ),
                persist=True,
            )
        else:
            self.task.run_register_python(
                f"{table_prefix}training_data_ids",
                lambda: self.task.spark.table(f"{table_prefix}training_data").select(
                    f"{id_col}_a", f"{id_col}_b", dep_var
                ),
                persist=True,
            )
        self._create_training_features(dep_var)

        training_features = self.task.spark.table(f"{table_prefix}training_features")
        pipeline = self._create_pipeline(training_features)
        model = pipeline.fit(training_features)
        prepped_data = model.transform(training_features)
        prepped_data.write.mode("overwrite").saveAsTable(
            f"{table_prefix}training_vectorized"
        )
        self.task.link_run.drop_temp_tables()
        self.task.spark.sql("set spark.sql.shuffle.partitions=200")

    def _create_training_features(self, dep_var):
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        if config[training_conf].get("use_training_data_features", False):
            self.task.run_register_python(
                f"{table_prefix}training_features",
                lambda: self.task.spark.table(f"{table_prefix}training_data"),
                persist=True,
            )
        else:
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
                "match_feature": dep_var,
                "advanced_comp_features": advanced_comp_features,
                "id": config["id_column"],
                "potential_matches": f"{table_prefix}training_data_ids",
            }

            (
                join_clauses,
                dist_tables,
            ) = dist_table_core.register_dist_tables_and_create_sql(
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

    def _create_pipeline(self, training_features):
        training_conf = str(self.task.training_conf)
        config = self.task.link_run.config
        ind_vars = list(config[training_conf]["independent_vars"])

        pipeline_stages = pipeline_core.generate_pipeline_stages(
            config, ind_vars, training_features, training_conf
        )
        return Pipeline(stages=pipeline_stages)
