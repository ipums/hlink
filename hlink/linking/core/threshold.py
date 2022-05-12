# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.sql.window import Window
from pyspark.sql.functions import rank, lead


def get_threshold_ratio(training_conf, model_conf, default=1.3):
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
    pred_df, alpha_threshold, threshold_ratio, training_conf, id_col
):
    """Adds a prediction column to the given pred_df by applying thresholds.

    Parameters
    ----------
    pred_df: DataFrame
        a Spark DataFrame of potential matches a probability column
    alpha_threshold: float
        the alpha threshold cutoff value. No record with a probability lower than this
        value will be considered for prediction = 1.
    threshold_ratio: float
        the threshold ratio cutoff value. Ratio's refer
        to the "a" record's next best probability value.
        Only used with the "drop_duplicate_with_threshold_ratio"
        configuration value.
    training_conf: dictionary
        the training config section
    id_col: string
        the id column

    Returns
    -------
    A Spark DataFrame containing the "prediction" column as well as other intermediate columns generated to create the prediction.
    """
    use_threshold_ratio = (
        training_conf.get("decision", "") == "drop_duplicate_with_threshold_ratio"
    )

    if use_threshold_ratio:
        return _apply_threshold_ratio(
            pred_df.drop("prediction"), alpha_threshold, threshold_ratio, id_col
        )
    else:
        return _apply_alpha_threshold(
            pred_df.drop("prediction"), alpha_threshold, threshold_ratio
        )


def _apply_alpha_threshold(pred_df, alpha_threshold, threshold_ratio):
    return pred_df.selectExpr(
        "*",
        f"case when probability >= {alpha_threshold} then 1 else 0 end as prediction",
    )


def _apply_threshold_ratio(df, alpha_threshold, threshold_ratio, id_col):
    """Apply a decision threshold using the ration of a match's probability to the next closest match's probability."""
    id_a = id_col + "_a"
    id_b = id_col + "_b"
    if "probability" not in df.columns:
        raise NameError(
            'In order to calculate the threshold ratio based on probabilities, you need to have a "probability" column in your data.'
        )
    else:
        windowSpec = Window.partitionBy(df[f"{id_a}"]).orderBy(
            df["probability"].desc(), df[f"{id_b}"]
        )
        prob_rank = rank().over(windowSpec)
        prob_lead = lead(df["probability"], 1).over(windowSpec)
        return (
            df.select(
                df["*"],
                prob_rank.alias("prob_rank"),
                prob_lead.alias("second_best_prob"),
            )
            .selectExpr(
                "*",
                f"""
            IF(
              second_best_prob IS NOT NULL
              AND second_best_prob >= {alpha_threshold}
              AND prob_rank == 1,
              probability / second_best_prob,
              NULL)
            as ratio
            """,
            )
            .selectExpr(
                "*",
                f"""
            CAST(
                probability >= {alpha_threshold}
            AND prob_rank == 1
            AND (ratio > {threshold_ratio} OR ratio is NULL)
            as INT) as prediction
            """,
            )
            .drop("prob_rank")
        )
