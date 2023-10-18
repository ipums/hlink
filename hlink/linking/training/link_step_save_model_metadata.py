# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.ml import PipelineModel
from hlink.linking.link_step import LinkStep
from pathlib import Path


class LinkStepSaveModelMetadata(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "save metadata about the model",
            output_table_names=[f"{task.table_prefix}training_model_metadata"],
            input_model_names=[f"{task.table_prefix}trained_model"],
        )

    def _run(self):
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        do_get_feature_importances = config[training_conf].get("feature_importances")

        if do_get_feature_importances is None or not do_get_feature_importances:
            print(
                "Skipping the save model metadata training step. "
                "To run this step and save model metadata like feature importances, "
                "set feature_importances = true in the [training] section of your "
                "config file."
            )
            return

        # retrieve the saved chosen model
        print("Loading chosen ML model...")

        try:
            pipeline_model = self.task.link_run.trained_models[
                f"{table_prefix}trained_model"
            ]
        except KeyError as e:
            new_error = RuntimeError(
                "Model not found!  Please run training step 2 to generate and "
                "train the chosen model. The model does not persist between runs "
                "of hlink."
            )

            raise new_error from e

        # make look at the features and their importances
        print("Retrieving model feature importances or coefficients...")
        try:
            feature_imp = pipeline_model.stages[-2].coefficients
        except:
            try:
                feature_imp = pipeline_model.stages[-2].featureImportances
            except:
                print(
                    "This model doesn't contain a coefficient or feature importances parameter -- check chosen model type."
                )
            else:
                label = "Feature importances"
        else:
            feature_imp = feature_imp.round(4)
            label = "Coefficients"

        varlist = self.task.spark.table(f"{table_prefix}features_list").toPandas()
        for i in varlist["idx"]:
            varlist.at[i, "score"] = feature_imp[i]
        varlist.sort_values("score", ascending=False, inplace=True)
        vl = self.spark.createDataFrame(varlist)
        vl.write.mode("overwrite").saveAsTable(f"{table_prefix}feature_importances")

        print(
            f"{label} have been saved to the Spark table '{table_prefix}feature_importances'."
        )
        print(varlist)
