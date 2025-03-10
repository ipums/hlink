# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from typing import Any

from pyspark.ml.feature import SQLTransformer
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import (
    RandomForestClassifier,
    LogisticRegression,
    DecisionTreeClassifier,
    GBTClassifier,
)
import hlink.linking.transformers.rename_prob_column

try:
    import synapse.ml.lightgbm
except ModuleNotFoundError:
    _lightgbm_available = False
else:
    _lightgbm_available = True

try:
    import xgboost.spark
except ModuleNotFoundError:
    _xgboost_available = False
else:
    _xgboost_available = True


def choose_classifier(model_type: str, params: dict[str, Any], dep_var: str):
    """Given a model type and hyper-parameters for the model, return a
    classifier of that type with those hyper-parameters, along with a
    post-classification transformer to run after classification.

    The post-classification transformer standardizes the output of the
    classifier for further processing. For example, some classifiers create
    models that output a probability array of [P(dep_var=0), P(dep_var=1)], and
    the post-classification transformer extracts the single float P(dep_var=1)
    as the probability for these models.

    Parameters
    ----------
    model_type
        the type of model, which may be random_forest, probit,
        logistic_regression, decision_tree, gradient_boosted_trees, lightgbm
        (requires the 'lightgbm' extra), or xgboost (requires the 'xgboost'
        extra)
    params
        a dictionary of hyper-parameters for the model
    dep_var
        the dependent variable for the model, sometimes also called the "label"

    Returns
    -------
    The classifier and a transformer to be used after classification, as a tuple.
    """
    post_transformer = SQLTransformer(statement="SELECT * FROM __THIS__")
    features_vector = "features_vector"
    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            **params,
            labelCol=dep_var,
            featuresCol=features_vector,
            seed=2133,
            probabilityCol="probability_array",
        )
        post_transformer = SQLTransformer(
            statement="SELECT *, parseProbVector(probability_array, 1) as probability FROM __THIS__"
        )

    elif model_type == "probit":
        classifier = GeneralizedLinearRegression(
            family="binomial",
            link="probit",
            labelCol=dep_var,
            featuresCol=features_vector,
            predictionCol="probability",
        )

    elif model_type == "logistic_regression":
        classifier = LogisticRegression(
            **params,
            featuresCol=features_vector,
            labelCol=dep_var,
            predictionCol="prediction",
            probabilityCol="probability_array",
        )
        post_transformer = SQLTransformer(
            statement="SELECT *, parseProbVector(probability_array, 1) as probability FROM __THIS__"
        )

    elif model_type == "decision_tree":
        classifier = DecisionTreeClassifier(
            **params,
            featuresCol=features_vector,
            labelCol=dep_var,
            probabilityCol="probability_array",
            seed=2133,
        )
        post_transformer = SQLTransformer(
            statement="SELECT *, parseProbVector(probability_array, 1) as probability FROM __THIS__"
        )

    elif model_type == "gradient_boosted_trees":
        classifier = GBTClassifier(
            **params,
            featuresCol=features_vector,
            labelCol=dep_var,
            seed=2133,
        )
        post_transformer = (
            hlink.linking.transformers.rename_prob_column.RenameProbColumn()
        )
    elif model_type == "lightgbm":
        if not _lightgbm_available:
            raise ModuleNotFoundError(
                "To use the 'lightgbm' model type, you need to install the synapseml "
                "Python package, which provides LightGBM-Spark integration, and "
                "its dependencies. Try installing hlink with the lightgbm extra: "
                "\n\n    pip install hlink[lightgbm]"
            )
        classifier = synapse.ml.lightgbm.LightGBMClassifier(
            **params,
            featuresCol=features_vector,
            labelCol=dep_var,
            probabilityCol="probability_array",
        )
        post_transformer = SQLTransformer(
            statement="SELECT *, parseProbVector(probability_array, 1) as probability FROM __THIS__"
        )
    elif model_type == "xgboost":
        if not _xgboost_available:
            raise ModuleNotFoundError(
                "To use the experimental 'xgboost' model type, you need to install "
                "the xgboost library and its dependencies. Try installing hlink with "
                "the xgboost extra:\n\n    pip install hlink[xgboost]"
            )
        classifier = xgboost.spark.SparkXGBClassifier(
            **params,
            features_col=features_vector,
            label_col=dep_var,
            probability_col="probability_array",
        )
        post_transformer = SQLTransformer(
            statement="SELECT *, parseProbVector(probability_array, 1) as probability FROM __THIS__"
        )
    else:
        raise ValueError(
            "Model type not recognized! Please check your config, reload, and try again."
        )
    return classifier, post_transformer
