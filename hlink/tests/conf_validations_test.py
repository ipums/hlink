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
    config = load_conf_file(conf_file)
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

    df_a.show()
    df_b.show()

    expected_err = (
        r"The following \[\[column_mappings\]\] has no 'column_name' attribute:"
    )
    with pytest.raises(ValueError, match=expected_err):
        check_column_mappings(config, df_a, df_b)
