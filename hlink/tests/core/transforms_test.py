from pyspark.sql import Row, SparkSession
import pytest

from hlink.linking.core.transforms import generate_transforms
from hlink.linking.link_task import LinkTask


@pytest.mark.parametrize("is_a", [True, False])
def test_generate_transforms_array_transform_1_col(
    spark: SparkSession, preprocessing: LinkTask, is_a: bool
) -> None:
    df = spark.createDataFrame(
        [[1, "Leto II", 3508], [2, "Hwi", 26], [3, "Siona", 25]],
        schema=["id", "name", "age"],
    )
    feature_selections = [
        {
            "transform": "array",
            "input_columns": ["name"],
            "output_column": "array_column",
        }
    ]

    df_result = generate_transforms(
        spark, df, feature_selections, preprocessing, is_a, "id"
    )
    array_column = df_result.select("array_column").collect()
    assert array_column == [
        Row(array_column=["Leto II"]),
        Row(array_column=["Hwi"]),
        Row(array_column=["Siona"]),
    ]


@pytest.mark.parametrize("is_a", [True, False])
def test_generate_transforms_array_transform_2_cols(
    spark: SparkSession, preprocessing: LinkTask, is_a: bool
) -> None:
    df = spark.createDataFrame(
        [[1, "Leto II", 3508], [2, "Hwi", 26], [3, "Siona", 25]],
        schema=["id", "name", "age"],
    )
    feature_selections = [
        {
            "transform": "array",
            "input_columns": ["name", "age"],
            "output_column": "array_column",
        }
    ]

    df_result = generate_transforms(
        spark, df, feature_selections, preprocessing, is_a, "id"
    )
    array_column = df_result.select("array_column").collect()
    assert array_column == [
        Row(array_column=["Leto II", "3508"]),
        Row(array_column=["Hwi", "26"]),
        Row(array_column=["Siona", "25"]),
    ]


@pytest.mark.parametrize("is_a", [True, False])
def test_generate_transforms_array_transform_3_cols(
    spark: SparkSession,
    preprocessing: LinkTask,
    is_a: bool,
) -> None:
    df = spark.createDataFrame(
        [
            [1, "Leto II", 3508, "Arrakis"],
            [2, "Hwi", 26, "Ix"],
            [3, "Siona", 25, "Arrakis"],
        ],
        schema=["id", "name", "age", "home"],
    )
    feature_selections = [
        {
            "transform": "array",
            "input_columns": ["home", "age", "name"],
            "output_column": "array_column",
        }
    ]

    df_result = generate_transforms(
        spark, df, feature_selections, preprocessing, is_a, "id"
    )
    array_column = df_result.select("array_column").collect()
    assert array_column == [
        Row(array_column=["Arrakis", "3508", "Leto II"]),
        Row(array_column=["Ix", "26", "Hwi"]),
        Row(array_column=["Arrakis", "25", "Siona"]),
    ]
