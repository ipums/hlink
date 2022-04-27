# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.ml.feature import SQLTransformer
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import (
    RandomForestClassifier,
    LogisticRegression,
    DecisionTreeClassifier,
    GBTClassifier,
)
import hlink.linking.transformers.rename_prob_column


def choose_classifier(model_type, params, dep_var):
    """Returns a classifier and a post_classification transformer given model type and params.

    Parameters
    ----------
    model_type: string
        name of model
    params: dictionary
        dictionary of parameters for model
    dep_var: string
        the dependent variable for the model

    Returns
    -------
    The classifer and a transformer to be used after classification.

    """
    post_transformer = SQLTransformer(statement="SELECT * FROM __THIS__")
    features_vector = "features_vector"
    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            **{
                key: val
                for key, val in params.items()
                if key not in ["threshold", "threshold_ratio"]
            },
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
            **{
                key: val
                for key, val in params.items()
                if key not in ["threshold", "threshold_ratio"]
            },
            featuresCol=features_vector,
            labelCol=dep_var,
            seed=2133,
        )
        post_transformer = (
            hlink.linking.transformers.rename_prob_column.RenameProbColumn()
        )

    else:
        raise ValueError(
            "Model type not recognized! Please check your config, reload, and try again."
        )
    return classifier, post_transformer
