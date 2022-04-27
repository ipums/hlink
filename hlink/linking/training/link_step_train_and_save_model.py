# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.ml import Pipeline

import hlink.linking.core.classifier as classifier_core
import hlink.linking.core.pipeline as pipeline_core
import hlink.linking.core.threshold as threshold_core

from hlink.linking.link_step import LinkStep


class LinkStepTrainAndSaveModel(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "train and save the model",
            input_table_names=[f"{task.table_prefix}training_features"],
            output_table_names=[],
            output_model_names=[f"{task.table_prefix}trained_model"],
        )

    def _run(self):
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        if not (config[training_conf].get("score_with_model", False)):
            raise ValueError(
                f"'score_with_model' not included or set to true in '{training_conf}' section of config!  Initiation of a model and scoring not completed!"
            )

        chosen_model_params = config[training_conf]["chosen_model"].copy()
        chosen_model_type = chosen_model_params.pop("type")
        chosen_model_params.pop(
            "threshold", config[training_conf].get("threshold", 0.8)
        )
        chosen_model_params.pop(
            "threshold_ratio",
            threshold_core.get_threshold_ratio(
                config[training_conf], chosen_model_params
            ),
        )

        ind_vars = config[training_conf]["independent_vars"]
        dep_var = config[training_conf]["dependent_var"]

        tf = self.task.spark.table(f"{table_prefix}training_features")

        # Create pipeline
        pipeline_stages = pipeline_core.generate_pipeline_stages(
            config, ind_vars, tf, training_conf
        )
        # TODO: Test if this will break if the scaler is used
        vector_assembler = pipeline_stages[-1]

        pre_pipeline = Pipeline(stages=pipeline_stages[:-1]).fit(tf)
        self.task.link_run.trained_models[f"{table_prefix}pre_pipeline"] = pre_pipeline
        tf_prepped = pre_pipeline.transform(tf)

        classifier, post_transformer = classifier_core.choose_classifier(
            chosen_model_type, chosen_model_params, dep_var
        )

        # Train and save pipeline
        pipeline = Pipeline(stages=[vector_assembler, classifier, post_transformer])

        model = pipeline.fit(tf_prepped)
        # model_path = config["spark_tmp_dir"] + "/chosen_model"
        self.task.link_run.trained_models[f"{table_prefix}trained_model"] = model
        # model.write().overwrite().save(model_path)
        # model.transform(pre_pipeline.transform(tf)).write.mode("overwrite").saveAsTable("training_features_scored")

        # model.transform(tf).write.mode("overwrite").saveAsTable("training_features_pipelined")
