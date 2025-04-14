# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import logging

from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from hlink.linking.link_step import LinkStep

logger = logging.getLogger(__name__)


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
            input_table_names=[f"{task.table_prefix}training_features_prepped"],
            output_table_names=[f"{task.table_prefix}training_feature_importances"],
            input_model_names=[f"{task.table_prefix}trained_model"],
        )

    def _run(self):
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        do_get_feature_importances = config[training_conf].get("feature_importances")

        if do_get_feature_importances is None or not do_get_feature_importances:
            logger.info(
                f"Skipping {table_prefix}training step 3 - save model metadata since {table_prefix}training.feature_importances is not set"
            )
            print(
                f"Skipping the save model metadata {table_prefix}training step. "
                "To run this step and save model metadata like feature importances, "
                f"set feature_importances = true in the [{table_prefix}training] section of your "
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
                f"Model not found!  Please run {table_prefix}training step 2 to generate and "
                "train the chosen model. The model does not persist between runs "
                "of hlink."
            )

            raise new_error from e

        # The pipeline model has three stages: vector assembler, model, and post transformer.
        vector_assembler = pipeline_model.stages[0]
        model = pipeline_model.stages[1]

        feature_names = vector_assembler.getInputCols()
        logger.debug(f"Feature names are {feature_names}")
        tf_prepped = self.task.spark.table(f"{table_prefix}training_features_prepped")
        tf_prepped_schema = dict(tf_prepped.dtypes)
        tf_prepped_row = tf_prepped.head()

        # Expand categorical features into multiple rows for display with their
        # respective coefficients / feature importances per category. Non-categorical
        # features just get one entry that looks like ("feature_name", None).
        expanded_features: list[(str, int | None)] = []
        for feature_name in feature_names:
            # Columns with type "vector" are categorical and may have more than one coefficient.
            # Many of these columns end with "_onehotencoded", and we remove that
            # suffix to clean up the column names. Categorical columns created through
            # feature interaction will probably not have the "_onehotencoded" suffix,
            # so we can't just check for that to find the categorical features.
            data_type = tf_prepped_schema[feature_name]
            if data_type == "vector":
                base_name = feature_name.removesuffix("_onehotencoded")
                num_categories = len(tf_prepped_row[feature_name])
                # Categories are numeric, starting at 0.
                expanded_features.extend(
                    (base_name, category) for category in range(num_categories)
                )
            else:
                base_name = feature_name.removesuffix("_imp")
                expanded_features.append((base_name, None))

        model_type = config[training_conf]["chosen_model"]["type"]

        logger.debug(f"Expanded features with categories are {expanded_features}")
        logger.debug(f"The model type is '{model_type}'")

        print("Retrieving model feature importances or coefficients...")

        if model_type == "xgboost":
            raw_weights = model.get_feature_importances("weight")
            raw_gains = model.get_feature_importances("total_gain")
            keys = [f"f{index}" for index in range(len(expanded_features))]

            weights = [raw_weights.get(key, 0.0) for key in keys]
            gains = [raw_gains.get(key, 0.0) for key in keys]
            label = "Feature importances (weights and gains)"

            importance_columns = [
                (StructField("weight", FloatType(), nullable=False), weights),
                (StructField("gain", FloatType(), nullable=False), gains),
            ]
        elif model_type == "lightgbm":
            # The "weight" of a feature is the number of splits it causes.
            weights = model.getFeatureImportances("split")
            gains = model.getFeatureImportances("gain")
            label = "Feature importances (weights and gains)"

            importance_columns = [
                (StructField("weight", FloatType(), nullable=False), weights),
                (StructField("gain", FloatType(), nullable=False), gains),
            ]
        else:
            try:
                feature_imp = model.coefficients
            except:
                try:
                    feature_imp = model.featureImportances
                except:
                    logger.warning(
                        f"Cannot compute feature importances for model of type '{model_type}'"
                    )
                    print(
                        "This model doesn't contain a coefficient or feature importances parameter -- check chosen model type."
                    )
                    return
                else:
                    label = "Feature importances"
            else:
                label = "Coefficients"

            # We need to convert from numpy float64s to Python floats to avoid type
            # issues when creating the DataFrame below.
            feature_importances = [
                float(importance) for importance in feature_imp.toArray()
            ]

            importance_columns = [
                (
                    StructField(
                        "coefficient_or_importance", FloatType(), nullable=False
                    ),
                    feature_importances,
                ),
            ]

        logger.debug("Creating the DataFrame and saving it as a table")
        feature_names, categories = zip(*expanded_features)
        importance_schema, importance_data = zip(*importance_columns)
        features_df = self.task.spark.createDataFrame(
            zip(feature_names, categories, *importance_data, strict=True),
            StructType(
                [
                    StructField("feature_name", StringType(), nullable=False),
                    StructField("category", IntegerType(), nullable=True),
                    *importance_schema,
                ]
            ),
        ).sort("feature_name", "category")
        feature_importances_table = (
            f"{self.task.table_prefix}training_feature_importances"
        )
        features_df.write.mode("overwrite").saveAsTable(feature_importances_table)

        print(f"{label} have been saved to the {feature_importances_table} table")
