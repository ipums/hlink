# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from hlink.spark.session import SparkConnection
import hlink.scripts.main_loop
import hlink.tests
from hlink.configs.load_config import load_conf_file
import hlink.linking.matching as match
import hlink.linking.training as train
import hlink.linking.hh_training as hh_train
import hlink.linking.hh_matching as hh_match
import hlink.linking.preprocessing as pre
import hlink.linking.reporting as rep
import hlink.linking.model_exploration as me
import hlink.linking.hh_model_exploration as hh_me
from hlink.linking.link_run import LinkRun
import json
import os
import pytest
import sys
from types import SimpleNamespace


pytest_plugins = (
    "hlink.tests.plugins.datasources",
    "hlink.tests.plugins.external_data_paths",
)


def load_table_from_csv(link_task, path, table_name):
    link_task.spark.read.csv(path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable(table_name)


@pytest.fixture(scope="session")
def spark(tmpdir_factory):
    os.environ["PYSPARK_PYTHON"] = sys.executable
    spark_connection = SparkConnection(
        tmpdir_factory.mktemp("derby"),
        tmpdir_factory.mktemp("warehouse"),
        tmpdir_factory.mktemp("spark_tmp_dir"),
        sys.executable,
        "linking",
    )
    return spark_connection.local()


@pytest.fixture(scope="function", autouse=True)
def set_shuffle(spark):
    spark.conf.set("spark.sql.shuffle.partitions", "1")
    spark.conf.set("spark.default.parallelism", "1")


@pytest.fixture(scope="function", autouse=True)
def drop_all_tables(main):
    main.do_drop_all("")


@pytest.fixture()
def package_path():
    """The path to the tests package."""
    return os.path.dirname(hlink.tests.conftest.__file__)


@pytest.fixture()
def input_data_dir_path(package_path):
    """The path to the directory containing test input data."""
    return os.path.join(package_path, "input_data")


@pytest.fixture()
def conf_dir_path(package_path):
    """The path to the directory containing test config files."""
    return os.path.join(package_path, "conf")


@pytest.fixture()
def spark_test_tmp_dir_path(spark, package_path):
    """The path to the test potential_matches csv file."""
    path = f"output_data/spark_test_tmp_dir{os.getenv('PYTEST_XDIST_WORKER', '')}"
    full_path = os.path.join(package_path, path)
    return full_path


@pytest.fixture()
def link_run(spark, conf):
    return LinkRun(spark, conf)


@pytest.fixture()
def preprocessing(link_run):
    return pre.Preprocessing(link_run)


@pytest.fixture()
def main(link_run):
    main = hlink.scripts.main_loop.Main(link_run)
    main.preloop()
    return main


@pytest.fixture()
def matching(link_run):
    return match.Matching(link_run)


@pytest.fixture()
def hh_matching(link_run):
    return hh_match.HHMatching(link_run)


@pytest.fixture()
def training(link_run):
    return train.Training(link_run)


@pytest.fixture()
def hh_training(link_run):
    return hh_train.HHTraining(link_run)


@pytest.fixture()
def reporting(link_run):
    return rep.Reporting(link_run)


@pytest.fixture()
def model_exploration(link_run):
    return me.ModelExploration(link_run)


@pytest.fixture()
def hh_model_exploration(link_run):
    return hh_me.HHModelExploration(link_run)


@pytest.fixture(scope="module")
def fake_self(spark):
    d = {"training_conf": "training"}
    n = SimpleNamespace(**d)
    return n


# Because of the way pytest fixtures work, `conf` is evaluated only once per test
# function call. The result is cached, and any subsequent requests for the fixture
# return the *same object*. Since this fixture returns a mutable dictionary, changes
# to the returned object will affect other fixtures that request and use `conf`.
#
# We use this with the `link_run` fixture and other fixtures like `preprocessing_conf`.
# Tests will request `preprocessing_conf`, which modifies `conf`. This modifies the
# `LinkRun.config` dictionary, since that is also a pointer to `conf`. Any additional
# modifications to `preprocessing_conf` in the test are also applied to `LinkRun.config`.
#
# TODO: Maybe think of a different way to do this. This way is convenient, but it can
# be hard to understand.
@pytest.fixture(scope="function")
def conf(conf_dir_path):
    return get_conf(conf_dir_path, "test.json")


@pytest.fixture(scope="function")
def integration_conf(input_data_dir_path, conf_dir_path):
    conf_file = os.path.join(conf_dir_path, "integration")
    conf = load_conf_file(conf_file)

    datasource_a = conf["datasource_a"]
    datasource_b = conf["datasource_b"]
    training = conf["training"]
    datasource_a["file"] = os.path.join(input_data_dir_path, datasource_a["file"])
    datasource_b["file"] = os.path.join(input_data_dir_path, datasource_b["file"])
    training["dataset"] = os.path.join(input_data_dir_path, training["dataset"])
    return conf


def get_conf(conf_dir_path, name):
    path_to_file = os.path.join(conf_dir_path, name)
    with open(path_to_file) as f:
        conf = json.load(f)
    return conf


@pytest.fixture(scope="function")
def preprocessing_conf(spark, conf, base_datasources):
    """Create a fixture to set the conf datasource_(a/b) values to the test data"""
    pathname_a, pathname_b = base_datasources
    conf["datasource_a"] = {"parquet_file": pathname_a}
    conf["datasource_b"] = {"parquet_file": pathname_b}
    return conf


@pytest.fixture(scope="function")
def preprocessing_conf_all_space_columns(
    spark, conf, datasource_unrestricted_blank_columns
):
    """Create a fixture to set the conf datasource_(a/b) values to the test data"""
    pathname_a, pathname_b = datasource_unrestricted_blank_columns
    conf["datasource_a"] = {"parquet_file": pathname_a}
    conf["datasource_b"] = {"parquet_file": pathname_b}
    return conf


@pytest.fixture(scope="function")
def preprocessing_conf_simple_names(spark, conf, datasource_preprocessing_simple_names):
    """Create a fixture for testing name substitution and bigrams"""
    pathname_a, pathname_b = datasource_preprocessing_simple_names
    conf["datasource_a"] = {"parquet_file": pathname_a}
    conf["datasource_b"] = {"parquet_file": pathname_b}
    return conf


@pytest.fixture(scope="function")
def preprocessing_conf_popularity(spark, conf, ext_path_preprocessing_popularity):
    """Create a fixture for testing name substitution and bigrams"""
    pathname = ext_path_preprocessing_popularity
    conf["datasource_a"] = {"file": pathname}
    conf["datasource_b"] = {"file": pathname}

    conf["column_mappings"] = [
        {"column_name": "sex"},
        {"column_name": "namefrst"},
        {"column_name": "namelast"},
        {"column_name": "birthyr"},
        {"column_name": "bpl"},
    ]

    conf["blocking"] = []
    conf["comparisons"] = {}

    return conf


@pytest.fixture(scope="function")
def preprocessing_conf_street_names(spark, conf, test_street_names_data_path):
    """Create a fixture for testing street name abbreviation substitutions"""
    conf["datasource_a"] = {"file": test_street_names_data_path}
    conf["datasource_b"] = {"file": test_street_names_data_path}
    return conf


@pytest.fixture(scope="function")
def preprocessing_conf_birthyr(spark, conf, birthyr_replace_path):
    """Create a fixture for testing name substitution and bigrams"""
    conf["datasource_a"] = {"file": birthyr_replace_path}
    conf["datasource_b"] = {"file": birthyr_replace_path}
    return conf


@pytest.fixture(scope="function")
def preprocessing_conf_synthetic_household_data(
    spark, conf, datasource_synthetic_households
):
    """Create a fixture conf for testing union transform of household/neighborhood data"""
    pathname_a, pathname_b = datasource_synthetic_households
    conf["datasource_a"] = {"parquet_file": pathname_a}
    conf["datasource_b"] = {"parquet_file": pathname_b}
    return conf


@pytest.fixture(scope="function")
def preprocessing_conf_19thc_nativity_conf(
    spark, conf, datasource_19thc_nativity_households_data
):
    """Create a fixture conf for testing nativity calculation"""
    full_path_a, full_path_b = datasource_19thc_nativity_households_data
    conf["datasource_a"] = {"file": full_path_a}
    conf["datasource_b"] = {"file": full_path_b}

    conf["column_mappings"] = [
        {"column_name": "serial"},
        {"column_name": "pernum"},
        {"column_name": "relate"},
        {"column_name": "bpl"},
        {"column_name": "momloc"},
        {"column_name": "poploc"},
        {"column_name": "namefrst"},
        {"column_name": "namelast"},
        {"column_name": "key_nativity_calc"},
        {
            "column_name": "nativity",
            "set_value_column_b": 0,
        },
        {
            "alias": "test_nativity",
            "column_name": "key_nativity_calc",
            "set_value_column_a": 0,
        },
        {"column_name": "key_mbpl"},
        {"column_name": "key_fbpl"},
        {
            "column_name": "mbpl",
            "set_value_column_a": 999,
        },
        {
            "column_name": "key_mbpl_range",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_mbpl_range_b",
            "set_value_column_a": 0,
        },
        {
            "column_name": "key_mother_nativity",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_mbpl_match",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_fbpl_match",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_mfbpl_match",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_m_caution_1870_1880",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_m_caution_1850_1860",
            "set_value_column_b": 0,
        },
    ]

    conf["blocking"] = [{"column_name": "id"}]
    conf["comparisons"] = {}

    conf["feature_selections"] = [
        {
            "output_column": "mbpl_range_b",
            "input_col": "mbpl",
            "transform": "sql_condition",
            "condition": "case when mbpl >= 997 then 0 when mbpl < 100 then 1 when (mbpl > 99 and mbpl < 997) then 2 else 0 end",
            "set_value_column_a": 0,
        },
        {
            "family_id": "serial",
            "other_col": "nativity",
            "output_col": "mother_nativity",
            "person_id": "pernum",
            "person_pointer": "momloc",
            "transform": "attach_family_col",
            "set_value_column_b": 0,
        },
        {
            "output_col": "mbpl_calc",
            "transform": "attach_family_col",
            "other_col": "bpl",
            "person_pointer": "momloc",
            "family_id": "serial",
            "person_id": "pernum",
        },
        {
            "output_col": "fbpl_calc",
            "transform": "attach_family_col",
            "other_col": "bpl",
            "person_pointer": "poploc",
            "family_id": "serial",
            "person_id": "pernum",
        },
        {
            "post_agg_feature": True,
            "output_column": "mbpl_range",
            "input_column": "mother_nativity",
            "transform": "sql_condition",
            "condition": "case when mother_nativity == 0 then 0 when mother_nativity > 0 and mother_nativity < 5 then 1 when mother_nativity == 5 then 2 else 0 end",
            "override_column_b": "mbpl_range_b",
        },
        {
            "post_agg_feature": True,
            "output_column": "nativity_calc",
            "transform": "sql_condition",
            "input_columns": ["bpl", "mbpl_calc", "fbpl_calc"],
            "condition": """case
                            when bpl >= 997 or mbpl_calc >= 997 or fbpl_calc >= 997
                            then 0
                            when bpl < 100 and fbpl_calc < 100 and mbpl_calc < 100
                            then 1
                            when bpl < 100 and mbpl_calc > 99 and fbpl_calc > 99
                            then 4
                            when bpl < 100 and mbpl_calc > 99 and fbpl_calc < 100
                            then 3
                            when bpl < 100 and fbpl_calc > 99 and mbpl_calc < 100
                            then 2
                            when bpl > 99 and bpl < 997
                            then 5
                            else 0
                            end""",
            "override_column_b": "test_nativity",
        },
    ]

    conf["comparison_features"] = [
        {
            "alias": "mbpl_match",
            "column_name": "mbpl_calc",
            "comparison_type": "present_and_matching_categorical",
        },
        {
            "alias": "fbpl_match",
            "column_name": "fbpl_calc",
            "comparison_type": "present_and_matching_categorical",
        },
        {
            "alias": "mfbpl_match",
            "column_name": "nativity_calc",
            "comparison_type": "present_and_equal_categorical_in_universe",
            "NIU": "0",
            "categorical": True,
        },
        {
            "alias": "m_caution_1870_1880",
            "categorical": True,
            "column_name": "mbpl_range",
            "comparison_type": "not_zero_and_not_equals",
        },
        {
            "alias": "m_caution_1850_1860",
            "categorical": True,
            "column_name": "mbpl_calc",
            "comparison_type": "present_and_not_equal",
        },
        {
            "column_name": "key_mbpl_match",
            "alias": "key_mbpl_match",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_fbpl_match",
            "alias": "key_fbpl_match",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_mfbpl_match",
            "alias": "key_mfbpl_match",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_m_caution_1870_1880",
            "alias": "key_m_caution_1870_1880",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_m_caution_1850_1860",
            "alias": "key_m_caution_1850_1860",
            "comparison_type": "fetch_a",
        },
    ]

    conf["training"] = {
        "dependent_var": "match",
        "independent_vars": [
            "mbpl_match",
            "fbpl_match",
            "mfbpl_match",
            "m_caution_1870_1880",
            "m_caution_1850_1860",
        ],
    }

    return conf


@pytest.fixture(scope="function")
def preprocessing_conf_19thc_caution_conf(
    spark, conf, datasource_19thc_nativity_households_data
):
    """Create a fixture conf for testing nativity calculation"""
    full_path_a, full_path_b = datasource_19thc_nativity_households_data
    conf["datasource_a"] = {"file": full_path_a}
    conf["datasource_b"] = {"file": full_path_b}

    conf["column_mappings"] = [
        {"column_name": "serial"},
        {"column_name": "pernum"},
        {"column_name": "relate"},
        {"column_name": "bpl"},
        {"column_name": "momloc"},
        {"column_name": "stepmom"},
        {"column_name": "poploc"},
        {"column_name": "namefrst"},
        {"column_name": "namelast"},
        {"column_name": "birthyr"},
        {"column_name": "key_nativity_calc"},
        {
            "column_name": "nativity",
            "set_value_column_b": 0,
        },
        {
            "alias": "test_nativity",
            "column_name": "key_nativity_calc",
            "set_value_column_a": 0,
        },
        {"column_name": "key_mbpl"},
        {"column_name": "key_fbpl"},
        {
            "column_name": "mbpl",
            "set_value_column_a": 999,
        },
        {
            "column_name": "key_mbpl_range",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_mbpl_range_b",
            "set_value_column_a": 0,
        },
        {
            "column_name": "key_mother_nativity",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_mbpl_match",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_fbpl_match",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_mfbpl_match",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_m_caution_1870_1880",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_m_caution_1850_1860",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_m_caution_cc3_012",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_m_caution_cc4_012",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_intermediate_mbpl_range_not_equals",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_intermediate_mbpl_range_not_zero_and_not_equals",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_intermediate_mother_birthyr_abs_diff_5",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_intermediate_stepmom_parent_step_change",
            "set_value_column_b": 0,
        },
        {
            "column_name": "key_intermediate_momloc_present_both_years",
            "set_value_column_b": 0,
        },
    ]

    conf["blocking"] = [{"column_name": "id"}]
    conf["comparisons"] = {}

    conf["feature_selections"] = [
        {
            "output_column": "mbpl_range_b",
            "input_col": "mbpl",
            "transform": "sql_condition",
            "condition": "case when mbpl >= 997 then 0 when mbpl < 100 then 1 when (mbpl > 99 and mbpl < 997) then 2 else 0 end",
            "set_value_column_a": 0,
        },
        {
            "family_id": "serial",
            "other_col": "nativity",
            "output_col": "mother_nativity",
            "person_id": "pernum",
            "person_pointer": "momloc",
            "transform": "attach_family_col",
            "set_value_column_b": 0,
        },
        {
            "output_col": "mbpl_calc",
            "transform": "attach_family_col",
            "other_col": "bpl",
            "person_pointer": "momloc",
            "family_id": "serial",
            "person_id": "pernum",
        },
        {
            "output_col": "fbpl_calc",
            "transform": "attach_family_col",
            "other_col": "bpl",
            "person_pointer": "poploc",
            "family_id": "serial",
            "person_id": "pernum",
        },
        {
            "output_col": "mother_birthyr",
            "transform": "attach_family_col",
            "other_col": "birthyr",
            "person_pointer": "momloc",
            "family_id": "serial",
            "person_id": "pernum",
        },
        {
            "post_agg_feature": True,
            "output_column": "mbpl_range",
            "input_column": "mother_nativity",
            "transform": "sql_condition",
            "condition": "case when mother_nativity == 0 then 0 when mother_nativity > 0 and mother_nativity < 5 then 1 when mother_nativity == 5 then 2 else 0 end",
            "override_column_b": "mbpl_range_b",
        },
        {
            "post_agg_feature": True,
            "output_column": "nativity_calc",
            "transform": "sql_condition",
            "input_columns": ["bpl", "mbpl_calc", "fbpl_calc"],
            "condition": """case
                            when bpl >= 997 or mbpl_calc >= 997 or fbpl_calc >= 997
                            then 0
                            when bpl < 100 and fbpl_calc < 100 and mbpl_calc < 100
                            then 1
                            when bpl < 100 and mbpl_calc > 99 and fbpl_calc > 99
                            then 4
                            when bpl < 100 and mbpl_calc > 99 and fbpl_calc < 100
                            then 3
                            when bpl < 100 and fbpl_calc > 99 and mbpl_calc < 100
                            then 2
                            when bpl > 99 and bpl < 997
                            then 5
                            else 0
                            end""",
            "override_column_b": "test_nativity",
        },
    ]

    conf["comparison_features"] = [
        {
            "alias": "mbpl_match",
            "column_name": "mbpl_calc",
            "comparison_type": "present_and_matching_categorical",
        },
        {
            "alias": "fbpl_match",
            "column_name": "fbpl_calc",
            "comparison_type": "present_and_matching_categorical",
        },
        {
            "alias": "mfbpl_match",
            "column_name": "nativity_calc",
            "comparison_type": "present_and_equal_categorical_in_universe",
            "NIU": "0",
            "categorical": True,
        },
        {
            "alias": "m_caution_1870_1880",
            "categorical": True,
            "column_name": "mbpl_range",
            "comparison_type": "not_zero_and_not_equals",
        },
        {
            "alias": "m_caution_1850_1860",
            "categorical": True,
            "column_name": "mbpl_calc",
            "comparison_type": "present_and_not_equal",
        },
        {
            "column_name": "key_mbpl_match",
            "alias": "key_mbpl_match",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_fbpl_match",
            "alias": "key_fbpl_match",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_mfbpl_match",
            "alias": "key_mfbpl_match",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_m_caution_1870_1880",
            "alias": "key_m_caution_1870_1880",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_m_caution_1850_1860",
            "alias": "key_m_caution_1850_1860",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_m_caution_cc3_012",
            "alias": "key_m_caution_cc3_012",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_m_caution_cc4_012",
            "alias": "key_m_caution_cc4_012",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_intermediate_mbpl_range_not_equals",
            "alias": "key_intermediate_mbpl_range_not_equals",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_intermediate_mbpl_range_not_zero_and_not_equals",
            "alias": "key_intermediate_mbpl_range_not_zero_and_not_equals",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_intermediate_mother_birthyr_abs_diff_5",
            "alias": "key_intermediate_mother_birthyr_abs_diff_5",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_intermediate_stepmom_parent_step_change",
            "alias": "key_intermediate_stepmom_parent_step_change",
            "comparison_type": "fetch_a",
        },
        {
            "column_name": "key_intermediate_momloc_present_both_years",
            "alias": "key_intermediate_momloc_present_both_years",
            "comparison_type": "fetch_a",
        },
        {
            "alias": "m_caution_cc3_012",
            "column_names": ["mbpl_range", "mother_birthyr", "momloc"],
            "comparison_type": "caution_comp_3_012",
            "categorical": True,
            "comp_a": {
                "column_name": "mbpl_range",
                "comparison_type": "not_equals",
            },
            "comp_b": {
                "column_name": "mother_birthyr",
                "comparison_type": "abs_diff",
                "gt_threshold": 5,
            },
            "comp_c": {
                "column_name": "momloc",
                "comparison_type": "present_both_years",
            },
        },
        {
            "alias": "m_caution_cc4_012",
            "column_names": ["mbpl_range", "mother_birthyr", "stepmom", "momloc"],
            "comparison_type": "caution_comp_4_012",
            "categorical": True,
            "comp_a": {
                "column_name": "mbpl_range",
                "comparison_type": "not_zero_and_not_equals",
            },
            "comp_b": {
                "column_name": "mother_birthyr",
                "comparison_type": "abs_diff",
                "gt_threshold": 5,
            },
            "comp_c": {
                "column_name": "stepmom",
                "comparison_type": "parent_step_change",
            },
            "comp_d": {
                "column_name": "momloc",
                "comparison_type": "present_both_years",
            },
        },
        {
            "alias": "intermediate_mbpl_range_not_equals",
            "column_name": "mbpl_range",
            "comparison_type": "not_equals",
        },
        {
            "alias": "intermediate_mbpl_range_not_zero_and_not_equals",
            "column_name": "mbpl_range",
            "comparison_type": "not_zero_and_not_equals",
        },
        {
            "alias": "intermediate_mother_birthyr_abs_diff_5",
            "column_name": "mother_birthyr",
            "comparison_type": "abs_diff",
            "gt_threshold": 5,
        },
        {
            "alias": "intermediate_stepmom_parent_step_change",
            "column_name": "stepmom",
            "comparison_type": "parent_step_change",
        },
        {
            "alias": "intermediate_momloc_present_both_years",
            "column_name": "momloc",
            "comparison_type": "present_both_years",
        },
    ]

    conf["training"] = {
        "dependent_var": "match",
        "independent_vars": [
            "mbpl_match",
            "fbpl_match",
            "mfbpl_match",
            "m_caution_1870_1880",
            "m_caution_1850_1860",
            "m_caution_cc3_012",
            "m_caution_cc4_012",
            "intermediate_mbpl_range_not_equals",
            "intermediate_mbpl_range_not_zero_and_not_equals",
            "intermediate_mother_birthyr_abs_diff_5",
            "intermediate_stepmom_parent_step_change",
            "intermediate_momloc_present_both_years",
        ],
    }

    return conf


@pytest.fixture(scope="function")
def preprocessing_conf_household_data(spark, conf, datasource_real_households):
    """Create a fixture conf for testing family/neighborhood transforms"""
    full_path_a, full_path_b = datasource_real_households
    conf["datasource_a"] = {"file": full_path_a}
    conf["datasource_b"] = {"file": full_path_b}

    conf["column_mappings"] = [
        {"column_name": "namefrst", "alias": "namefrst_orig"},
        {"column_name": "namelast", "alias": "namelast_orig"},
        {"column_name": "bpl"},
        {"column_name": "sex"},
        {"column_name": "enumdist"},
        {"column_name": "pernum"},
        {"column_name": "serial"},
        {"column_name": "sploc"},
        {"column_name": "poploc"},
        {"column_name": "momloc"},
        {"column_name": "relate"},
        {
            "column_name": "namefrst",
            "alias": "namefrst_clean",
            "transforms": [
                {"type": "lowercase_strip"},
                {"type": "rationalize_name_words"},
                {"type": "remove_qmark_hyphen"},
                {"type": "replace_apostrophe"},
                {
                    "type": "remove_suffixes",
                    "values": ["jr", "sr", "ii", "iii", "iv", "v", "vi", "vii", "viii"],
                },
                {"type": "remove_alternate_names"},
                {"type": "condense_strip_whitespace"},
            ],
        },
        {
            "column_name": "namelast",
            "alias": "namelast_clean",
            "transforms": [
                {"type": "lowercase_strip"},
                {"type": "rationalize_name_words"},
                {"type": "remove_qmark_hyphen"},
                {"type": "replace_apostrophe"},
                {
                    "type": "remove_suffixes",
                    "values": ["jr", "sr", "ii", "iii", "iv", "v", "vi", "vii", "viii"],
                },
                {"type": "remove_prefixes", "values": ["ah"]},
                {
                    "type": "condense_prefixes",
                    "values": ["mc", "mac", "o", "de", "van", "di"],
                },
                {"type": "remove_one_letter_names"},
                {"type": "remove_alternate_names"},
                {"type": "condense_strip_whitespace"},
            ],
        },
    ]

    conf["blocking"] = []
    conf["comparisons"] = {}

    return conf


@pytest.fixture(scope="function")
def preprocessing_conf_rel_rows(spark, conf, test_data_rel_rows_age):
    """Create a fixture conf for testing family/neighborhood transforms"""
    full_path_a, full_path_b = test_data_rel_rows_age
    conf["datasource_a"] = {"file": full_path_a}
    conf["datasource_b"] = {"file": full_path_b}
    conf["id_column"] = "histid"

    conf["column_mappings"] = [
        {"column_name": "namefrst", "alias": "namefrst_orig"},
        {"column_name": "namelast", "alias": "namelast_orig"},
        {"column_name": "birthyr"},
        {"column_name": "sex"},
        {"column_name": "pernum"},
        {"column_name": "serialp"},
        {"column_name": "relate"},
        {"column_name": "age"},
        {"column_name": "yearp", "alias": "year"},
        {
            "column_name": "namefrst",
            "alias": "namefrst_clean",
            "transforms": [
                {"type": "lowercase_strip"},
                {"type": "rationalize_name_words"},
                {"type": "remove_qmark_hyphen"},
                {"type": "replace_apostrophe"},
                {"type": "remove_suffixes", "values": ["jr", "sr", "ii", "iii"]},
                {"type": "remove_alternate_names"},
                {"type": "condense_strip_whitespace"},
            ],
        },
        {
            "column_name": "namefrst_clean",
            "alias": "namefrst_split",
            "transforms": [{"type": "split"}],
        },
        {
            "column_name": "namefrst_split",
            "alias": "namefrst_unstd",
            "transforms": [{"type": "array_index", "value": 0}],
        },
        {
            "column_name": "birthyr",
            "alias": "clean_birthyr",
            "transforms": [
                {
                    "type": "mapping",
                    "mappings": {9999: "", 1999: ""},
                    "output_type": "int",
                }
            ],
        },
    ]

    conf["feature_selections"] = [
        {
            "input_column": "clean_birthyr",
            "output_column": "replaced_birthyr",
            "condition": "case when clean_birthyr is null or clean_birthyr == '' then year - age else clean_birthyr end",
            "transform": "sql_condition",
        },
        {
            "family_id": "serialp",
            "input_cols": [
                "histid",
                "namefrst_unstd",
                "replaced_birthyr",
                "sex",
                "relate",
            ],
            "output_col": "namefrst_related_rows",
            "transform": "related_individual_rows",
            "filters": [
                {"column": "relate", "min": 300, "max": 1099},
                {"column": "age", "min": 0, "max": 999},
            ],
        },
        {
            "family_id": "serialp",
            "input_cols": [
                "histid",
                "namefrst_unstd",
                "replaced_birthyr",
                "sex",
                "relate",
            ],
            "output_col": "namefrst_related_rows_age_min_5",
            "transform": "related_individual_rows",
            "filters": [
                {"column": "relate", "min": 300, "max": 1099},
                {"column": "age", "min": 5, "max": 999},
            ],
        },
        {
            "family_id": "serialp",
            "input_cols": [
                "histid",
                "namefrst_unstd",
                "replaced_birthyr",
                "sex",
                "relate",
            ],
            "output_col": "namefrst_related_rows_age_b_min_5",
            "transform": "related_individual_rows",
            "filters": [
                {"column": "relate", "min": 300, "max": 1099},
                {"column": "age", "min": 5, "max": 999, "dataset": "b"},
            ],
        },
    ]

    conf["blocking"] = []
    conf["comparisons"] = {}

    return conf


@pytest.fixture(scope="function")
def matching_conf(spark, conf, datasource_matching, matching):
    """Create conf fixture for testing matching steps using the prepped_df_(a/b) dataframes and populate basic config values"""
    matching.link_run.print_sql = True
    conf["column_mappings"] = [
        {"column_name": "serialp"},
        {"column_name": "namefrst"},
        {"column_name": "namelast"},
        {"column_name": "bpl"},
        {"column_name": "sex"},
        {"column_name": "street"},
        {"column_name": "enum_dist"},
    ]
    conf["blocking"] = [{"column_name": "sex"}]
    conf["comparisons"] = {
        "comp_a": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "comp_b": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "operator": "AND",
    }
    conf["training"] = {}

    conf["training"]["dependent_var"] = "match"
    conf["training"]["independent_vars"] = ["namefrst_jw", "namelast_jw", "ssex"]
    return conf


@pytest.fixture(scope="function")
def matching_conf_namefrst_std_and_unstd(
    spark, conf, matching, test_data_blocking_double_comparison
):
    """Create conf fixture for testing matching steps using the prepped_df_(a/b) dataframes and populate basic config values"""

    conf["id_column"] = "histid"
    full_path_a, full_path_b = test_data_blocking_double_comparison
    conf["datasource_a"] = {"file": full_path_a}
    conf["datasource_b"] = {"file": full_path_b}

    conf["column_mappings"] = [
        {"column_name": "namefrst_unstd"},
        {"column_name": "namefrst_std"},
        {"column_name": "namelast_clean"},
        {"column_name": "bpl_clean"},
        {"column_name": "sex"},
        {"column_name": "birthyr"},
    ]
    conf["blocking"] = [
        {"column_name": "sex"},
        {"column_name": "bpl_clean"},
        {"column_name": "birthyr"},
    ]

    conf["comparisons"] = {
        "operator": "AND",
        "comp_a": {
            "operator": "OR",
            "comp_a": {
                "feature_name": "namefrst_unstd_jw",
                "threshold": 0.70,
                "comparison_type": "threshold",
            },
            "comp_b": {
                "feature_name": "namefrst_std_jw",
                "threshold": 0.70,
                "comparison_type": "threshold",
            },
        },
        "comp_b": {
            "feature_name": "namelast_jw",
            "threshold": 0.70,
            "comparison_type": "threshold",
        },
    }

    conf["comparison_features"] = [
        {
            "alias": "namefrst_unstd_jw",
            "column_name": "namefrst_unstd",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "namefrst_std_jw",
            "column_name": "namefrst_std",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
    ]

    conf["training"] = {}
    return conf


@pytest.fixture(scope="function")
def blocking_explode_conf(spark, conf):
    """Create conf fixture for testing matching steps using the family/neighborhood transforms and populate basic config values"""

    conf["column_mappings"] = [
        {"column_name": "namefrst"},
        {"column_name": "namelast"},
        {"column_name": "birthyr"},
        {"column_name": "sex"},
    ]

    conf["blocking"] = [
        {
            "column_name": "birthyr_3",
            "dataset": "a",
            "derived_from": "birthyr",
            "expand_length": 3,
            "explode": True,
        },
        {"column_name": "sex"},
    ]

    conf["comparison_features"] = [
        {
            "alias": "namefrst_jw",
            "column_name": "namefrst",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "ssex",
            "column_name": "sex",
            "comparison_type": "equals",
            "categorical": True,
        },
    ]
    conf["comparisons"] = {
        "comp_a": {
            "feature_name": "namefrst_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "comp_b": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "operator": "AND",
    }

    conf["training"] = {}
    conf["training"]["dependent_var"] = "match"
    conf["training"]["independent_vars"] = ["namefrst_jw", "namelast_jw", "ssex"]
    return conf


@pytest.fixture(scope="function")
def matching_household_conf(
    spark, conf, datasource_real_households, preprocessing, matching
):
    """Create conf fixture for testing matching steps using the family/neighborhood transforms and populate basic config values"""

    full_path_a, full_path_b = datasource_real_households
    conf["datasource_a"] = {"file": full_path_a}
    conf["datasource_b"] = {"file": full_path_b}

    conf["column_mappings"] = [
        {"column_name": "namefrst", "alias": "namefrst_orig"},
        {"column_name": "namelast", "alias": "namelast_orig"},
        {"column_name": "bpl"},
        {"column_name": "birthyr"},
        {"column_name": "age"},
        {"column_name": "sex"},
        {"column_name": "enumdist"},
        {"column_name": "pernum"},
        {"column_name": "serial", "alias": "serialp"},
        {"column_name": "sploc"},
        {"column_name": "poploc"},
        {"column_name": "momloc"},
        {"column_name": "relate"},
        {
            "column_name": "namefrst",
            "alias": "namefrst_std",
            "transforms": [
                {"type": "lowercase_strip"},
                {"type": "rationalize_name_words"},
                {"type": "remove_qmark_hyphen"},
                {"type": "replace_apostrophe"},
                {
                    "type": "remove_suffixes",
                    "values": ["jr", "sr", "ii", "iii", "iv", "v", "vi", "vii", "viii"],
                },
                {"type": "remove_alternate_names"},
                {"type": "condense_strip_whitespace"},
            ],
        },
        {
            "column_name": "namelast",
            "alias": "namelast_clean",
            "transforms": [
                {"type": "lowercase_strip"},
                {"type": "rationalize_name_words"},
                {"type": "remove_qmark_hyphen"},
                {"type": "replace_apostrophe"},
                {
                    "type": "remove_suffixes",
                    "values": ["jr", "sr", "ii", "iii", "iv", "v", "vi", "vii", "viii"],
                },
                {"type": "remove_prefixes", "values": ["ah"]},
                {
                    "type": "condense_prefixes",
                    "values": ["mc", "mac", "o", "de", "van", "di"],
                },
                {"type": "remove_one_letter_names"},
                {"type": "remove_alternate_names"},
                {"type": "condense_strip_whitespace"},
            ],
        },
    ]

    conf["blocking"] = [{"column_name": "sex"}]

    conf["comparisons"] = {
        "comp_a": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "comp_b": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "operator": "AND",
    }

    conf["training"] = {}
    conf["training"]["dependent_var"] = "match"

    return conf


@pytest.fixture(scope="function")
def matching_comparison_conf(spark, conf, datasource_matching_comparisons, matching):
    """Create conf fixture for testing matching steps using the prepped_df_(a/b) dataframes and populate basic config values"""
    conf["column_mappings"] = [
        {"column_name": "id"},
        {"column_name": "namelast"},
        {"column_name": "sex"},
        {"column_name": "mpbl"},
        {"column_name": "mother_birthyr"},
        {"column_name": "stepmom"},
        {"column_name": "spouse_bpl"},
        {"column_name": "spouse_birthyr"},
        {"column_name": "durmarr"},
        {"column_name": "mother_namefrst"},
        {"column_name": "spouse_namefrst"},
        {"column_name": "momloc"},
        {"column_name": "sploc"},
    ]
    conf["blocking"] = [{"column_name": "sex"}]
    conf["comparisons"] = {
        "comp_a": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "comp_b": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "operator": "AND",
    }

    return conf


@pytest.fixture(scope="function")
def matching_conf_counties(spark, conf, county_dist_datasources):
    """Create a fixture for testing name substitution and bigrams"""
    pathname_a, pathname_b = county_dist_datasources
    conf["datasource_a"] = {"parquet_file": pathname_a}
    conf["datasource_b"] = {"parquet_file": pathname_b}

    return conf


@pytest.fixture(scope="function")
def matching_conf_nativity(spark, conf, nativity_datasources):
    """Create a fixture for testing name substitution and bigrams"""
    pathname_a, pathname_b = nativity_datasources
    conf["datasource_a"] = {"file": pathname_a}
    conf["datasource_b"] = {"file": pathname_b}
    conf["training"] = {}

    conf["training"]["dependent_var"] = "match"

    return conf


@pytest.fixture(scope="function")
def training_conf(spark, conf, training_data_path, datasource_training):
    """Create the prepped_df_(a/b) dataframes and populate basic config values"""
    conf["training"] = {
        "dataset": training_data_path,
        "dependent_var": "match",
        "n_training_iterations": 10,
    }
    conf["column_mappings"] = [
        {"column_name": "serialp"},
        {"column_name": "namelast"},
        {"column_name": "bpl"},
        {"column_name": "sex"},
        {"column_name": "region"},
    ]
    conf["blocking"] = [{"column_name": "sex"}]
    conf["comparisons"] = {
        "comp_a": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "comp_b": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "operator": "AND",
    }

    return conf


@pytest.fixture(scope="function")
def hh_training_conf(spark, conf, hh_training_data_path):
    """Create the prepped_df_(a/b) dataframes and populate basic config values"""
    conf["id_column"] = "histid"
    conf["drop_data_from_scored_matches"] = False
    conf["hh_training"] = {
        "dataset": hh_training_data_path,
        "dependent_var": "match",
        "prediction_col": "match",
        "n_training_iterations": 4,
        "seed": 120,
        "independent_vars": [
            "namelast_jw",
            "namefrst_jw",
            "byrdiff",
            "ssex",
            "srelate",
        ],
        "score_with_model": True,
        "use_training_data_features": False,
        "decision": "drop_duplicate_with_threshold_ratio",
        "get_precision_recall_curve": True,
        "chosen_model": {
            "type": "logistic_regression",
            "threshold": 0.5,
            "threshold_ratio": 1.2,
        },
        "model_parameters": [
            {"type": "logistic_regression", "threshold": 0.5, "threshold_ratio": 1.2},
            {
                "type": "random_forest",
                "maxDepth": 5.0,
                "numTrees": 75.0,
                "threshold": 0.5,
                "threshold_ratio": 1.2,
            },
        ],
    }
    conf["column_mappings"] = [
        {"column_name": "serialp"},
        {"column_name": "relate"},
        {"column_name": "namelast_clean"},
        {"column_name": "namefrst_unstd"},
        {"column_name": "clean_birthyr"},
        {"column_name": "sex"},
    ]
    conf["blocking"] = []
    conf["comparison_features"] = [
        {
            "alias": "byrdiff",
            "column_name": "clean_birthyr",
            "comparison_type": "abs_diff",
        },
        {
            "alias": "ssex",
            "column_name": "sex",
            "comparison_type": "equals",
            "categorical": True,
        },
        {
            "alias": "srelate",
            "column_name": "relate",
            "comparison_type": "equals",
            "categorical": True,
        },
        {
            "alias": "namefrst_jw",
            "column_name": "namefrst_unstd",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
    ]

    return conf


@pytest.fixture(scope="function")
def hh_agg_feat_conf(spark, conf, hh_training_data_path):
    """Create the prepped_df_(a/b) dataframes and populate basic config values"""
    conf["id_column"] = "histid"
    conf["drop_data_from_scored_matches"] = False
    conf["hh_training"] = {
        "dataset": hh_training_data_path,
        "dependent_var": "match",
        "independent_vars": [
            "jw_max_a",
            "jw_max_b",
            "f1_match",
            "f2_match",
            "byrdiff",
            "sexmatch",
            "mardurmatch",
        ],
        "score_with_model": True,
        "chosen_model": {
            "type": "logistic_regression",
            "threshold": 0.5,
            "threshold_ratio": 1.2,
        },
    }
    conf["column_mappings"] = [
        {"column_name": "serialp"},
        {"column_name": "relate"},
        {"column_name": "pernum"},
        {"column_name": "namelast_clean"},
        {"column_name": "namefrst_unstd"},
        {"column_name": "clean_birthyr"},
        {"column_name": "sex"},
        {"column_name": "namefrst_mid_init"},
        {"column_name": "namefrst_init"},
        {"column_name": "namefrst_mid_init_2"},
    ]
    conf["blocking"] = []
    conf["comparison_features"] = [
        {
            "alias": "byrdiff",
            "column_name": "clean_birthyr",
            "comparison_type": "abs_diff",
        },
        {
            "alias": "sexmatch",
            "column_name": "sex",
            "comparison_type": "equals",
            "categorical": True,
        },
        {
            "alias": "namefrst_jw",
            "column_name": "namefrst_unstd",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
        {"alias": "durmarr_a", "column_name": "durmarr", "comparison_type": "fetch_a"},
        {"alias": "durmarr_b", "column_name": "durmarr", "comparison_type": "fetch_b"},
        {
            "alias": "mardurmatch",
            "column_name": "durmarr",
            "not_equals": 99,
            "comparison_type": "abs_diff",
            "btwn_threshold": [9, 14],
            "categorical": True,
        },
        {
            "alias": "f1_match",
            "first_init_col": "namefrst_init",
            "mid_init_cols": ["namefrst_mid_init", "namefrst_mid_init_2"],
            "comparison_type": "f1_match",
            "categorical": True,
        },
        {
            "alias": "f2_match",
            "first_init_col": "namefrst_init",
            "mid_init_cols": ["namefrst_mid_init", "namefrst_mid_init_2"],
            "comparison_type": "f2_match",
            "categorical": True,
        },
        {
            "alias": "fn_a",
            "column_name": "namefrst_unstd",
            "comparison_type": "fetch_a",
        },
        {"alias": "fi_a", "column_name": "namefrst_init", "comparison_type": "fetch_a"},
        {
            "alias": "fn_b",
            "column_name": "namefrst_unstd",
            "comparison_type": "fetch_b",
        },
        {"alias": "fi_b", "column_name": "namefrst_init", "comparison_type": "fetch_b"},
        {
            "alias": "mi_a",
            "column_name": "namefrst_mid_init",
            "comparison_type": "fetch_a",
        },
        {
            "alias": "mi_b",
            "column_name": "namefrst_mid_init",
            "comparison_type": "fetch_b",
        },
    ]

    return conf
