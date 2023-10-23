# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from hlink.linking.link_step import LinkStep


class LinkStepSaveModelMetadata(LinkStep):
    """Save metadata about the trained machine learning model.

    By default this step is skipped. The training.feature_importances config
    attribute enables it. The step saves either coefficients or feature
    importances for the trained model, depending on the type of model. It saves
    these data to the training_feature_importances Spark table.

    This step is primarily helpful for debugging issues with machine learning
    features and understanding how those features affect scoring of potential
    matches.
    """

    def __init__(self, task):
        super().__init__(
            task,
            "save metadata about the model",
            output_table_names=[f"{task.table_prefix}training_feature_importances"],
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

        # The pipeline model has three stages: vector assembler, classifier, post
        # transformer.
        vector_assembler = pipeline_model.stages[0]
        classifier = pipeline_model.stages[1]

        print("Retrieving model feature importances or coefficients...")
        try:
            feature_imp = classifier.coefficients
        except:
            try:
                feature_imp = classifier.featureImportances
            except:
                print(
                    "This model doesn't contain a coefficient or feature importances parameter -- check chosen model type."
                )
                return
            else:
                label = "Feature importances"
        else:
            label = "Coefficients"

        column_names = vector_assembler.getInputCols()
        # We need to convert from numpy float64s to Python floats to avoid type
        # issues when creating the DataFrame below.
        feature_importances = [
            float(importance) for importance in feature_imp.toArray()
        ]

        features_df = self.task.spark.createDataFrame(
            zip(column_names, feature_importances),
            "feature_name: string, coefficient_or_importance: double",
        ).sort("coefficient_or_importance", ascending=False)

        feature_importances_table = (
            f"{self.task.table_prefix}training_feature_importances"
        )
        features_df.write.mode("overwrite").saveAsTable(feature_importances_table)

        print(f"{label} have been saved to the {feature_importances_table} table")
