# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import logging
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lead, rank, when

logger = logging.getLogger(__name__)


def get_threshold_ratio(
    training_conf: dict[str, Any], model_conf: dict[str, Any], default: float = 1.3
) -> float | Any:
    """Gets the threshold ratio or default from the config using the correct precedence.

    Parameters
    ----------
    training_conf: dictionary
        the config dictionary containing the training conf
    model_conf: dictionary
        the config dictionary for a specific model
    default: float
        the default value to use if the threshold is missing

    Returns
    -------
    The threshold ratio.
    """
    if "threshold_ratio" in model_conf:
        return model_conf["threshold_ratio"]
    elif "threshold_ratio" in training_conf:
        return training_conf["threshold_ratio"]
    else:
        return default


def predict_using_thresholds(
    pred_df: DataFrame,
    alpha_threshold: float,
    threshold_ratio: float,
    id_col: str,
    decision: str | None,
) -> DataFrame:
    """Adds a "prediction" column to the given data frame by applying
    thresholds to the "probability" column. The prediction column has either
    the value 0, indicating that the potential match does not meet the
    requirements for a match, or 1, indicating that the potential match does
    meet the requirements for a match. The requirements for a match depend on
    the decision argument, which switches between two different options.

    1. If decision is "drop_duplicate_with_threshold_ratio", then
    predict_using_thresholds() uses both the alpha_threshold and
    threshold_ratio.

    predict_using_thresholds() groups the matches by their id in data set A,
    and selects from each group the potential match with the highest
    probability. Then, if there is a second-highest probability in the group,
    predict_using_thresholds() computes the ratio of the highest probability to
    the second highest probability and stores it as the ratio column. Finally,
    predict_using_thresholds() picks out of each group the potential match with
    the highest probability and marks it with prediction = 1 if

      A. its probability is at least alpha_threshold and
      B. either there is no second-highest probability, or
      the ratio of the highest probability to the second-highest is greater
      than threshold_ratio.

    2. If decision is any other string or is None, then
    predict_using_thresholds() does not use threshold_ratio and instead just
    applies alpha_threshold. Each potential match with a probability of at
    least alpha_threshold gets prediction = 1, and each potential match with a
    probability less than alpha_threshold gets prediction = 0.

    Parameters
    ----------
    pred_df:
        a Spark DataFrame of potential matches with a probability column
    alpha_threshold:
        The alpha threshold cutoff value. No record with a probability lower
        than this value will be considered for prediction = 1.
    threshold_ratio:
        The threshold ratio cutoff value, only used with the
        "drop_duplicate_with_threshold_ratio" decision. The ratio is between
        the best probability and second-best probability for potential matches
        with the same id in data set A.
    id_col:
        the name of the id column
    decision:
        how to apply the alpha_threshold and threshold_ratio

    Returns
    -------
    a Spark DataFrame containing the "prediction" column, and possibly some
    additional intermediate columns generated to create the prediction
    """
    if "probability" not in pred_df.columns:
        raise ValueError(
            "the input data frame must have a 'probability' column to make predictions using thresholds"
        )

    use_threshold_ratio = (
        decision is not None and decision == "drop_duplicate_with_threshold_ratio"
    )

    if use_threshold_ratio:
        logger.debug(
            f"Making predictions with alpha threshold and threshold ratio: {alpha_threshold=}, {threshold_ratio=}"
        )
        return _apply_threshold_ratio(
            pred_df.drop("prediction"), alpha_threshold, threshold_ratio, id_col
        )
    else:
        logger.debug(
            f"Making predictions with alpha threshold but without threshold ratio: {alpha_threshold=}"
        )
        return _apply_alpha_threshold(pred_df.drop("prediction"), alpha_threshold)


def _apply_alpha_threshold(pred_df: DataFrame, alpha_threshold: float) -> DataFrame:
    prediction = when(col("probability") >= alpha_threshold, 1).otherwise(0)
    return pred_df.withColumn("prediction", prediction)


def _apply_threshold_ratio(
    df: DataFrame, alpha_threshold: float, threshold_ratio: float, id_col: str
) -> DataFrame:
    """Apply an alpha_threshold and threshold_ratio.

    After thresholding on alpha_threshold, compute the ratio of each id_a's
    highest potential match probability to its second-highest potential match
    probability and compare the ratio to threshold_ratio."""
    id_a = id_col + "_a"
    id_b = id_col + "_b"
    windowSpec = Window.partitionBy(id_a).orderBy(col("probability").desc(), id_b)
    prob_rank = rank().over(windowSpec)
    prob_lead = lead("probability", 1).over(windowSpec)

    should_compute_probability_ratio = col("second_best_prob").isNotNull() & (
        col("prob_rank") == 1
    )
    # To be a match, the row must...
    # 1. Have prob_rank 1, so that it's the most likely match,
    # 2. Have a probability of at least alpha_threshold,
    # and
    # 3. Either have no ratio (since there's no second best probability), or
    #    have a ratio of more than threshold_ratio.
    is_match = (
        (col("probability") >= alpha_threshold)
        & (col("prob_rank") == 1)
        & ((col("ratio") > threshold_ratio) | col("ratio").isNull())
    )
    return (
        df.select(
            "*",
            prob_rank.alias("prob_rank"),
            prob_lead.alias("second_best_prob"),
        )
        .withColumn(
            "ratio",
            when(
                should_compute_probability_ratio,
                col("probability") / col("second_best_prob"),
            ).otherwise(None),
        )
        .withColumn("prediction", is_match.cast("integer"))
        .drop("prob_rank")
    )
