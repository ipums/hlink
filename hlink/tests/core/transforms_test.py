from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import col
import pytest

from hlink.linking.core.transforms import apply_transform, generate_transforms
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


def test_generate_transforms_override_column_a(
    spark: SparkSession, preprocessing: LinkTask
) -> None:
    """
    In a feature selection, you can set the `override_column_a` attribute to
    copy that column in dataset A as the feature selection instead of computing
    the feature selection like normal. This does not affect dataset B.
    """
    feature_selections = [
        {
            "output_column": "mbpl_range",
            "transform": "sql_condition",
            "input_column": "mother_nativity",
            "condition": "CASE WHEN mother_nativity = 0 THEN 0 WHEN mother_nativity > 0 and mother_nativity < 5 THEN 1 WHEN mother_nativity = 5 THEN 2 ELSE 0 END",
            "override_column_a": "test_override_column",
        }
    ]
    df_a = spark.createDataFrame(
        [[0, 2, -1], [1, 5, -1], [2, 0, -1], [3, 6, -1]],
        "id:integer, mother_nativity:integer, test_override_column:integer",
    )
    df_b = spark.createDataFrame(
        [[0, 2, -1], [1, 5, -1], [2, 0, -1], [3, 6, -1]],
        "id:integer, mother_nativity:integer, test_override_column:integer",
    )

    df_result_a = generate_transforms(
        spark, df_a, feature_selections, preprocessing, is_a=True, id_col="id"
    ).sort("id")

    result_a = df_result_a.collect()
    assert result_a == [
        Row(id=0, mother_nativity=2, test_override_column=-1, mbpl_range=-1),
        Row(id=1, mother_nativity=5, test_override_column=-1, mbpl_range=-1),
        Row(id=2, mother_nativity=0, test_override_column=-1, mbpl_range=-1),
        Row(id=3, mother_nativity=6, test_override_column=-1, mbpl_range=-1),
    ]

    # mbpl_range should be computed with the SQL condition in dataset B
    df_result_b = generate_transforms(
        spark, df_b, feature_selections, preprocessing, is_a=False, id_col="id"
    ).sort("id")

    result_b = df_result_b.collect()
    assert result_b == [
        Row(id=0, mother_nativity=2, test_override_column=-1, mbpl_range=1),
        Row(id=1, mother_nativity=5, test_override_column=-1, mbpl_range=2),
        Row(id=2, mother_nativity=0, test_override_column=-1, mbpl_range=0),
        Row(id=3, mother_nativity=6, test_override_column=-1, mbpl_range=0),
    ]


def test_generate_transforms_override_column_b(
    spark: SparkSession, preprocessing: LinkTask
) -> None:
    """
    In a feature selection, you can set the `override_column_b` attribute to
    copy that column in dataset B as the feature selection instead of computing
    the feature selection like normal. This does not affect dataset A.
    """
    feature_selections = [
        {
            "output_column": "mbpl_range",
            "transform": "sql_condition",
            "input_column": "mother_nativity",
            "condition": "CASE WHEN mother_nativity = 0 THEN 0 WHEN mother_nativity > 0 AND mother_nativity < 5 THEN 1 WHEN mother_nativity = 5 THEN 2 ELSE 0 END",
            "override_column_b": "test_override_column",
        }
    ]
    df_a = spark.createDataFrame(
        [[0, 2, -1], [1, 5, -1], [2, 0, -1], [3, 6, -1]],
        "id:integer, mother_nativity:integer, test_override_column:integer",
    )
    df_b = spark.createDataFrame(
        [[0, 2, -1], [1, 5, -1], [2, 0, -1], [3, 6, -1]],
        "id:integer, mother_nativity:integer, test_override_column:integer",
    )

    # mbpl_range should be computed with the SQL condition in dataset A
    df_result_a = generate_transforms(
        spark, df_a, feature_selections, preprocessing, is_a=True, id_col="id"
    ).sort("id")
    result_a = df_result_a.collect()
    assert result_a == [
        Row(id=0, mother_nativity=2, test_override_column=-1, mbpl_range=1),
        Row(id=1, mother_nativity=5, test_override_column=-1, mbpl_range=2),
        Row(id=2, mother_nativity=0, test_override_column=-1, mbpl_range=0),
        Row(id=3, mother_nativity=6, test_override_column=-1, mbpl_range=0),
    ]

    df_result_b = generate_transforms(
        spark, df_b, feature_selections, preprocessing, is_a=False, id_col="id"
    ).sort("id")
    result_b = df_result_b.collect()
    assert result_b == [
        Row(id=0, mother_nativity=2, test_override_column=-1, mbpl_range=-1),
        Row(id=1, mother_nativity=5, test_override_column=-1, mbpl_range=-1),
        Row(id=2, mother_nativity=0, test_override_column=-1, mbpl_range=-1),
        Row(id=3, mother_nativity=6, test_override_column=-1, mbpl_range=-1),
    ]


@pytest.mark.parametrize("is_a", [True, False])
def test_generate_transforms_error_when_unrecognized_transform(
    spark: SparkSession, preprocessing: LinkTask, is_a: bool
) -> None:
    feature_selections = [
        {"input_column": "age", "output_column": "age2", "transform": "not_supported"}
    ]
    df = spark.createDataFrame([], "id:integer, age:integer")

    with pytest.raises(ValueError, match="Invalid transform type"):
        generate_transforms(spark, df, feature_selections, preprocessing, is_a, "id")


@pytest.mark.parametrize("is_a", [True, False])
def test_apply_transform_when_value(spark: SparkSession, is_a: bool) -> None:
    transform = {"type": "when_value", "value": 6, "if_value": 0, "else_value": 1}
    column_select = col("marst")
    output_col = apply_transform(column_select, transform, is_a)

    df = spark.createDataFrame([[3], [6], [2], [6], [1]], "marst:integer")
    transformed = df.select("marst", output_col.alias("output"))
    result = transformed.collect()

    assert result == [
        Row(marst=3, output=1),
        Row(marst=6, output=0),
        Row(marst=2, output=1),
        Row(marst=6, output=0),
        Row(marst=1, output=1),
    ]
