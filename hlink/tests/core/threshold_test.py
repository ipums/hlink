# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.sql import Row, SparkSession
import pytest

from hlink.linking.core.threshold import predict_using_thresholds


def test_predict_using_thresholds_default_decision(spark: SparkSession) -> None:
    """
    The default decision tells predict_using_thresholds() not to do
    de-duplication on the id. Instead, it just applies alpha_threshold to the
    probabilities to determine predictions.
    """
    input_rows = [
        (0, "A", 0.1),
        (0, "B", 0.7),
        (1, "C", 0.2),
        (2, "D", 0.4),
        (3, "E", 1.0),
        (4, "F", 0.0),
    ]
    df = spark.createDataFrame(input_rows, schema=["id_a", "id_b", "probability"])

    # We are using the default decision, so threshold_ratio will be ignored
    predictions = predict_using_thresholds(
        df, alpha_threshold=0.6, threshold_ratio=0.0, id_col="id", decision=None
    )

    output_rows = (
        predictions.sort("id_a", "id_b").select("id_a", "id_b", "prediction").collect()
    )

    OutputRow = Row("id_a", "id_b", "prediction")
    assert output_rows == [
        OutputRow(0, "A", 0),
        OutputRow(0, "B", 1),
        OutputRow(1, "C", 0),
        OutputRow(2, "D", 0),
        OutputRow(3, "E", 1),
        OutputRow(4, "F", 0),
    ]


def test_predict_using_thresholds_drop_duplicates_decision(spark: SparkSession) -> None:
    """
    The "drop_duplicate_with_threshold_ratio" decision tells
    predict_using_thresholds() to look at the ratio between the first- and
    second-best probabilities for each id, and to only set prediction = 1 when
    the ratio between those probabilities is at least threshold_ratio.
    """
    # id_a 0: two probable matches that will be de-duplicated so that both have prediction = 0
    # id_a 1: one probable match that will have prediction = 1
    # id_a 2: one improbable match that will have prediction = 0
    # id_a 3: one probable match that will have prediction = 1, and one improbable match that will have prediction = 0
    input_rows = [
        (0, "A", 0.8),
        (0, "B", 0.9),
        (1, "C", 0.75),
        (2, "C", 0.3),
        (3, "D", 0.1),
        (3, "E", 0.8),
    ]
    df = spark.createDataFrame(input_rows, schema=["id_a", "id_b", "probability"])
    predictions = predict_using_thresholds(
        df,
        alpha_threshold=0.5,
        threshold_ratio=2.0,
        id_col="id",
        decision="drop_duplicate_with_threshold_ratio",
    )

    output_rows = (
        predictions.sort("id_a", "id_b").select("id_a", "id_b", "prediction").collect()
    )
    OutputRow = Row("id_a", "id_b", "prediction")

    assert output_rows == [
        OutputRow(0, "A", 0),
        OutputRow(0, "B", 0),
        OutputRow(1, "C", 1),
        OutputRow(2, "C", 0),
        OutputRow(3, "D", 0),
        OutputRow(3, "E", 1),
    ]


@pytest.mark.parametrize("decision", [None, "drop_duplicate_with_threshold_ratio"])
def test_predict_using_thresholds_missing_probability_column_error(
    spark: SparkSession, decision: str | None
) -> None:
    """
    When the input DataFrame is missing the "probability" column,
    predict_using_thresholds() raises a friendly error.
    """
    df = spark.createDataFrame([(0, "A"), (1, "B")], schema=["id_a", "id_b"])
    with pytest.raises(
        ValueError, match="the input data frame must have a 'probability' column"
    ):
        predict_using_thresholds(
            df, alpha_threshold=0.5, threshold_ratio=1.5, id_col="id", decision=decision
        )
