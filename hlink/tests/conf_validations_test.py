import os
import pytest

from pyspark.sql import SparkSession

from hlink.configs.load_config import load_conf_file
from hlink.scripts.lib.conf_validations import analyze_conf, check_column_mappings
from hlink.linking.link_run import LinkRun


@pytest.mark.parametrize(
    "conf_name,error_msg",
    [
        ("missing_datasource_a", r"Section \[datasource_a\] does not exist in config"),
        ("missing_datasource_b", r"Section \[datasource_b\] does not exist in config"),
        ("no_id_column_a", "Datasource A is missing the id column 'ID'"),
        ("no_id_column_b", "Datasource B is missing the id column 'ID'"),
        ("duplicate_comp_features", "Alias names are not unique"),
        ("duplicate_feature_sel", "Output columns are not unique"),
        ("duplicate_col_maps", "Column names are not unique"),
    ],
)
def test_invalid_conf(conf_dir_path, spark, conf_name, error_msg):
    conf_file = os.path.join(conf_dir_path, conf_name)
    _path, config = load_conf_file(conf_file)
    link_run = LinkRun(spark, config)

    with pytest.raises(ValueError, match=error_msg):
        analyze_conf(link_run)


def test_check_column_mappings_mappings_missing(spark: SparkSession) -> None:
    """
    The config must have a column_mappings section.
    """
    config = {}
    df_a = spark.createDataFrame([[1], [2], [3]], ["a"])
    df_b = spark.createDataFrame([[4], [5], [6]], ["b"])

    with pytest.raises(
        ValueError, match=r"No \[\[column_mappings\]\] exist in the conf file"
    ):
        check_column_mappings(config, df_a, df_b)


def test_check_column_mappings_no_column_name(spark: SparkSession) -> None:
    """
    Each column mapping in the config must have a column_name attribute.
    """
    config = {
        "column_mappings": [{"column_name": "AGE", "alias": "age"}, {"alias": "height"}]
    }
    df_a = spark.createDataFrame([[20], [40], [60]], ["AGE"])
    df_b = spark.createDataFrame([[70], [50], [30]], ["AGE"])

    expected_err = (
        r"The following \[\[column_mappings\]\] has no 'column_name' attribute:"
    )
    with pytest.raises(ValueError, match=expected_err):
        check_column_mappings(config, df_a, df_b)


def test_check_column_mappings_column_name_not_available_datasource_a(
    spark: SparkSession,
) -> None:
    """
    Column mappings may only use column_names that appear in datasource A or a
    previous column mapping.
    """
    config = {"column_mappings": [{"column_name": "HEIGHT"}]}

    df_a = spark.createDataFrame([[20], [40], [60]], ["AGE"])
    df_b = spark.createDataFrame([[70, 123], [50, 123], [30, 123]], ["AGE", "HEIGHT"])

    expected_err = (
        r"Within a \[\[column_mappings\]\] the column_name 'HEIGHT' "
        r"does not exist in datasource_a and no previous \[\[column_mapping\]\] "
        "alias exists for it"
    )

    with pytest.raises(ValueError, match=expected_err):
        check_column_mappings(config, df_a, df_b)


def test_check_column_mappings_set_value_column_a_does_not_need_column(
    spark: SparkSession,
) -> None:
    """
    When set_value_column_a is present for a column mapping, that column does not
    need to be present in datasource A.
    """
    config = {"column_mappings": [{"column_name": "HEIGHT", "set_value_column_a": 125}]}

    df_a = spark.createDataFrame([[20], [40], [60]], ["AGE"])
    df_b = spark.createDataFrame([[70, 123], [50, 123], [30, 123]], ["AGE", "HEIGHT"])

    check_column_mappings(config, df_a, df_b)


def test_check_column_mappings_column_name_not_available_datasource_b(
    spark: SparkSession,
) -> None:
    """
    Column mappings may only use column_names that appear in datasource B or a
    previous column mapping.
    """
    config = {"column_mappings": [{"column_name": "HEIGHT"}]}

    df_a = spark.createDataFrame([[70, 123], [50, 123], [30, 123]], ["AGE", "HEIGHT"])
    df_b = spark.createDataFrame([[20], [40], [60]], ["AGE"])

    expected_err = (
        r"Within a \[\[column_mappings\]\] the column_name 'HEIGHT' "
        r"does not exist in datasource_b and no previous \[\[column_mapping\]\] "
        "alias exists for it"
    )

    with pytest.raises(ValueError, match=expected_err):
        check_column_mappings(config, df_a, df_b)


def test_check_column_mappings_set_value_column_b_does_not_need_column(
    spark: SparkSession,
) -> None:
    """
    When set_value_column_b is present for a column mapping, that column does not
    need to be present in datasource B.
    """
    config = {"column_mappings": [{"column_name": "HEIGHT", "set_value_column_b": 125}]}

    df_a = spark.createDataFrame([[70, 123], [50, 123], [30, 123]], ["AGE", "HEIGHT"])
    df_b = spark.createDataFrame([[20], [40], [60]], ["AGE"])

    check_column_mappings(config, df_a, df_b)


def test_check_column_mappings_previous_mappings_are_available(
    spark: SparkSession,
) -> None:
    """
    Columns created in a previous column mapping can be used in other column
    mappings.
    """
    config = {
        "column_mappings": [
            {"column_name": "AGE", "alias": "AGE_HLINK"},
            {"column_name": "AGE_HLINK", "alias": "AGE_HLINK2"},
        ]
    }
    df_a = spark.createDataFrame([[70], [50], [30]], ["AGE"])
    df_b = spark.createDataFrame([[20], [40], [60]], ["AGE"])

    check_column_mappings(config, df_a, df_b)


def test_check_column_mappings_override_column_a(spark: SparkSession) -> None:
    """
    The override_column_a attribute lets you control which column you read from
    in datasource A.
    """
    config = {
        "column_mappings": [{"column_name": "AGE", "override_column_a": "ageColumn"}]
    }
    df_a = spark.createDataFrame([[20], [40], [60]], ["ageColumn"])
    df_b = spark.createDataFrame([[70], [50], [30]], ["AGE"])

    check_column_mappings(config, df_a, df_b)


def test_check_column_mappings_override_column_b(spark: SparkSession) -> None:
    """
    The override_column_b attribute lets you control which column you read from
    in datasource B.
    """
    config = {
        "column_mappings": [{"column_name": "ageColumn", "override_column_b": "AGE"}]
    }
    df_a = spark.createDataFrame([[20], [40], [60]], ["ageColumn"])
    df_b = spark.createDataFrame([[70], [50], [30]], ["AGE"])

    check_column_mappings(config, df_a, df_b)


def test_check_column_mappings_override_column_a_not_present(
    spark: SparkSession,
) -> None:
    """
    The override_column_a column must be present in datasource A.
    """
    config = {
        "column_mappings": [
            {"column_name": "AGE", "override_column_a": "oops_not_there"}
        ]
    }
    df_a = spark.createDataFrame([[20], [40], [60]], ["ageColumn"])
    df_b = spark.createDataFrame([[70], [50], [30]], ["AGE"])

    expected_err = (
        r"Within a \[\[column_mappings\]\] the override_column_a column "
        "'oops_not_there' does not exist in datasource_a"
    )
    with pytest.raises(ValueError, match=expected_err):
        check_column_mappings(config, df_a, df_b)


def test_check_column_mappings_override_column_b_not_present(
    spark: SparkSession,
) -> None:
    """
    The override_column_b column must be present in datasource B.
    """
    config = {
        "column_mappings": [
            {"column_name": "AGE", "override_column_b": "oops_not_there"}
        ]
    }
    df_a = spark.createDataFrame([[20], [40], [60]], ["AGE"])
    df_b = spark.createDataFrame([[70], [50], [30]], ["AGE"])

    expected_err = (
        r"Within a \[\[column_mappings\]\] the override_column_b column "
        "'oops_not_there' does not exist in datasource_b"
    )
    with pytest.raises(ValueError, match=expected_err):
        check_column_mappings(config, df_a, df_b)
