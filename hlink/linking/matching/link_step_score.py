# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import logging
from pyspark.sql import Row, Window
from pyspark.sql import functions as f

import hlink.linking.core.comparison_feature as comparison_feature_core
import hlink.linking.core.threshold as threshold_core
import hlink.linking.core.dist_table as dist_table_core
from hlink.linking.util import spark_shuffle_partitions_heuristic

from hlink.linking.link_step import LinkStep


class LinkStepScore(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "score",
            input_table_names=[f"{task.table_prefix}potential_matches"],
            output_table_names=[
                f"{task.table_prefix}potential_matches_prepped",
                f"{task.table_prefix}scored_potential_matches",
                f"{task.table_prefix}predicted_matches",
            ],
            input_model_names=[f"{task.table_prefix}trained_model"],
        )

    def _run(self):
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        if training_conf not in config or "chosen_model" not in config[training_conf]:
            print(
                f"WARNING: Skipping step '{self.desc}'. Your config file either does not contain a '{training_conf}' section or a 'chosen_model' section within the '{training_conf}' section."
            )
            return

        dataset_size = self.task.spark.table(f"{table_prefix}potential_matches").count()
        num_partitions = spark_shuffle_partitions_heuristic(dataset_size)
        self.task.spark.sql(f"set spark.sql.shuffle.partitions={num_partitions}")
        logging.info(
            f"Dataset size is {dataset_size}, so set Spark partitions to {num_partitions} for this step"
        )

        id_a = config["id_column"] + "_a"
        id_b = config["id_column"] + "_b"
        chosen_model_params = config[training_conf]["chosen_model"].copy()
        self._create_features(config)
        pm = self.task.spark.table(f"{table_prefix}potential_matches_prepped")

        ind_var_columns = config[training_conf]["independent_vars"]
        flatten = lambda l: [item for sublist in l for item in sublist]
        if config.get("pipeline_features", False):
            pipeline_columns = flatten(
                [
                    f["input_columns"] if "input_columns" in f else [f["input_column"]]
                    for f in config["pipeline_features"]
                ]
            )
        else:
            pipeline_columns = []
        required_columns = set(
            ind_var_columns
            + pipeline_columns
            + ["exact", id_a, id_b, "serialp_a", "serialp_b"]
        ) & set(pm.columns)

        pre_pipeline = self.task.link_run.trained_models.get(
            f"{table_prefix}pre_pipeline"
        )
        if pre_pipeline is None:
            raise ValueError(
                "Missing a temporary table from the training task. This table will not be persisted between sessions of hlink for technical reasons. Please run training before running this step."
            )

        self.task.run_register_python(
            f"{table_prefix}potential_matches_pipeline",
            lambda: pre_pipeline.transform(pm.select(*required_columns)),
            persist=True,
        )
        plm = self.task.link_run.trained_models[f"{table_prefix}trained_model"]
        pp_required_cols = set(plm.stages[0].getInputCols() + [id_a, id_b])
        pre_pipeline = self.task.spark.table(
            f"{table_prefix}potential_matches_pipeline"
        ).select(*pp_required_cols)
        score_tmp = plm.transform(pre_pipeline)
        # TODO: Move save_feature_importances to training or model evaluation step
        # _save_feature_importances(self.spark, score_tmp)

        alpha_threshold = chosen_model_params.get("threshold", 0.5)
        threshold_ratio = threshold_core.get_threshold_ratio(
            config[training_conf], chosen_model_params, default=1.3
        )
        predictions = threshold_core.predict_using_thresholds(
            score_tmp,
            alpha_threshold,
            threshold_ratio,
            config[training_conf],
            config["id_column"],
        )
        predictions.write.mode("overwrite").saveAsTable(f"{table_prefix}predictions")
        pmp = self.task.spark.table(f"{table_prefix}potential_matches_pipeline")
        self._save_table_with_requested_columns(pm, pmp, predictions, id_a, id_b)
        self._save_predicted_matches(config, id_a, id_b)
        self.task.spark.sql("set spark.sql.shuffle.partitions=200")

    def _save_feature_importances(self, spark, score_tmp):
        config = self.task.link_run.config
        if not (config[f"{self.task.training_conf}"].get("feature_importances", False)):
            return
        cols = (
            score_tmp.select("*").schema["features_vector"].metadata["ml_attr"]["attrs"]
        )
        list_extract = []
        for i in cols:
            list_extract += cols[i]

        varlist = spark.createDataFrame(Row(**x) for x in list_extract)
        varlist.write.mode("overwrite").saveAsTable(
            f"{self.task.table_prefix}features_list"
        )

    def _save_table_with_requested_columns(self, pm, pmp, predictions, id_a, id_b):
        # merge back in original data for feature verification
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        if config.get("drop_data_from_scored_matches", False):
            output_columns = [
                f"{id_a}",
                f"{id_b}",
                "probability_array",
                "probability",
                "prediction",
            ]
            columns_to_select = sorted(
                list(set(predictions.columns) & set(output_columns))
            )
            self.task.run_register_python(
                f"{table_prefix}scored_potential_matches",
                lambda: self.task.spark.table(f"{table_prefix}predictions").select(
                    columns_to_select
                ),
                persist=True,
            )
        else:
            pm_source_cols = list(set(pmp.columns) - set(predictions.columns))
            self.task.run_register_sql(
                f"{table_prefix}scored_potential_matches",
                template="scored_potential_matches",
                t_ctx={
                    "pm_source_cols": pm_source_cols,
                    "id_a": id_a,
                    "id_b": id_b,
                    "predictions": f"{table_prefix}predictions",
                    "potential_matches": f"{table_prefix}potential_matches_pipeline",
                },
                persist=True,
            )
        print(
            f"Scored potential matches have been saved to the Spark table '{table_prefix}scored_potential_matches'."
        )

    def _save_predicted_matches(self, conf, id_a, id_b):
        table_prefix = self.task.table_prefix

        spms = self.task.spark.table(f"{table_prefix}scored_potential_matches").filter(
            "prediction == 1"
        )
        w = Window.partitionBy(f"{id_b}")
        spms = spms.select("*", f.count(f"{id_b}").over(w).alias(f"{id_b}_count"))
        spms = spms.filter(f"{id_b}_count == 1")
        spms = spms.drop(f"{id_b}_count")
        spms.write.mode("overwrite").saveAsTable(f"{table_prefix}predicted_matches")
        print(
            f"Predicted matches with duplicate histid_b removed have been saved to the Spark table '{table_prefix}predicted_matches'."
        )

    def _create_features(self, conf):
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix

        dep_var = conf[training_conf]["dependent_var"]
        potential_matches = f"{table_prefix}potential_matches"
        table_name = f"{table_prefix}potential_matches_prepped"
        pm_columns = self.task.spark.table(potential_matches).columns
        (
            comp_features,
            advanced_comp_features,
            hh_comp_features,
            dist_features,
        ) = comparison_feature_core.get_features(
            conf,
            conf[f"{training_conf}"]["independent_vars"],
            pregen_features=pm_columns,
        )
        t_ctx_def = {
            "comp_features": comp_features,
            "match_feature": dep_var,
            "advanced_comp_features": advanced_comp_features,
            "id": conf["id_column"],
            "potential_matches": potential_matches,
        }
        join_clauses, dist_tables = dist_table_core.register_dist_tables_and_create_sql(
            self.task, dist_features
        )
        t_ctx_def["distance_table"] = join_clauses
        if len(dist_tables):
            t_ctx_def["broadcast_hints"] = dist_table_core.get_broadcast_hint(
                dist_tables
            )

        comparison_feature_core.create_feature_tables(
            self.task,
            t_ctx_def,
            advanced_comp_features,
            hh_comp_features,
            conf["id_column"],
            table_name=table_name,
        )
