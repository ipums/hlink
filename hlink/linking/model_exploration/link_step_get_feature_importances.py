# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.ml import PipelineModel
from hlink.linking.link_step import LinkStep
from pathlib import Path


class LinkStepGetFeatureImportances(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "get feature importances",
            input_table_names=[
                f"{task.table_prefix}training_features",
                f"{task.table_prefix}training_vectorized",
            ],
            output_table_names=[f"{task.table_prefix}training_results"],
        )

    def _run(self):
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        self.task.spark.sql("set spark.sql.shuffle.partitions=1")

        if "feature_importances" in config[training_conf]:
            if config[training_conf]["feature_importances"]:

                # retrieve the saved chosen model
                print("Loading chosen ML model...")
                model_path = Path(config["spark_tmp_dir"]) / "chosen_model"
                try:
                    plm = PipelineModel.load(str(model_path))
                except:
                    print(
                        "Model not found!  You might need to run step_2 to generate and train the chosen model if you haven't already done so."
                    )

                # make look at the features and their importances
                print("Retrieving model feature importances or coefficients...")
                try:
                    feature_imp = plm.stages[-2].coefficients
                except:
                    try:
                        feature_imp = plm.stages[-2].featureImportances
                    except:
                        print(
                            "This model doesn't contain a coefficient or feature importances parameter -- check chosen model type."
                        )
                    else:
                        label = "Feature importances"
                else:
                    feature_imp = feature_imp.round(4)
                    label = "Coefficients"

                varlist = self.task.spark.table(
                    f"{table_prefix}features_list"
                ).toPandas()
                for i in varlist["idx"]:
                    varlist.at[i, "score"] = feature_imp[i]
                varlist.sort_values("score", ascending=False, inplace=True)
                vl = self.spark.createDataFrame(varlist)
                vl.write.mode("overwrite").saveAsTable(
                    f"{table_prefix}feature_importances"
                )

                print(
                    f"{label} have been saved to the Spark table '{table_prefix}feature_importances'."
                )
                print(varlist)

            else:
                print(
                    f"'feature_importances' not set == true in '{training_conf}' section of config! Calculation of feature importances or coefficients not completed!"
                )
        else:
            print(
                f"'feature_importances' not included and set == true in '{training_conf}' section of config! Calculation of feature importances or coefficients not completed!"
            )

        self.task.spark.sql("set spark.sql.shuffle.partitions=200")
