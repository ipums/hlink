# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import os
import pandas as pd
import pytest
from pyspark.sql.types import StructType, StructField, LongType
from hlink.errors import DataError


@pytest.mark.quickcheck
def test_step_0(preprocessing, spark, preprocessing_conf):
    """Test preprocessing step 0 to ensure that temporary raw_df_unpartitioned_(a/b) tables are created (exact copies of datasources from config). Also test that the presistent raw_df_(a/b) tables are created. Should be same as raw datasources with filters applied"""

    # Run the preprocessing step (this will use the test data)
    preprocessing.run_step(0)

    # Create pandas DFs of the step_0 preprocessed test data
    up_pdf_a = spark.table("raw_df_unpartitioned_a").toPandas()
    up_pdf_b = spark.table("raw_df_unpartitioned_b").toPandas()

    pdf_a = spark.table("raw_df_a").toPandas()
    pdf_b = spark.table("raw_df_b").toPandas()

    # Make assertions on the data
    assert pdf_a.query("id == 10")["serialp"].iloc[0] == "A"
    assert pdf_a.query("id == 20")["serialp"].iloc[0] == "B"

    assert pdf_b.query("id == 10")["serialp"].iloc[0] == "C"
    assert pdf_b.query("id == 30")["serialp"].iloc[0] == "D"

    assert up_pdf_a.query("id == 10")["serialp"].iloc[0] == "A"
    assert up_pdf_a.query("id == 30")["serialp"].iloc[0] == "B"

    assert up_pdf_b.query("id == 30")["serialp"].iloc[0] == "D"
    assert up_pdf_b.query("id == 50")["serialp"].iloc[0] == "E"


def test_step_0_datasource_parquet_file(
    preprocessing, spark, preprocessing_conf, input_data_dir_path
):
    """Test preprocessing step 0 to ensure that temporary raw_df_unpartitioned_(a/b) tables are created (exact copies of datasources from config). Also test that the presistent raw_df_(a/b) tables are created. Should be same as raw datasources with filters applied"""

    preprocessing_conf["datasource_a"] = {
        "parquet_file": os.path.join(
            input_data_dir_path, "test_parquet_data_a.parquet"
        ),
        "alias": "parquet_file_test_a",
    }

    # Run the preprocessing step (this will use the test data)
    preprocessing.run_step(0)

    # Create pandas DFs of the step_0 preprocessed test data
    up_pdf_a = spark.table("raw_df_unpartitioned_a").toPandas()

    pdf_a = spark.table("raw_df_a").toPandas()

    # Make assertions on the data
    assert pdf_a.query("id == 10")["bpl"].iloc[0] == 120
    assert pdf_a.query("id == 20")["bpl"].iloc[0] == 240

    assert up_pdf_a.query("id == 30")["bpl"].iloc[0] == 360
    assert up_pdf_a.query("id == 20")["bpl"].iloc[0] == 240


def test_step_0_datasource_file_parquet(
    preprocessing, spark, preprocessing_conf, input_data_dir_path
):
    """Test preprocessing step 0 to ensure that temporary raw_df_unpartitioned_(a/b) tables are created (exact copies of datasources from config). Also test that the presistent raw_df_(a/b) tables are created. Should be same as raw datasources with filters applied"""

    preprocessing_conf["datasource_a"] = {
        "file": os.path.join(input_data_dir_path, "test_parquet_data_b.parquet"),
        "alias": "parquet_file_test_b",
    }

    # Run the preprocessing step (this will use the test data)
    preprocessing.run_step(0)

    # Create pandas DFs of the step_0 preprocessed test data
    up_pdf_a = spark.table("raw_df_unpartitioned_a").toPandas()

    pdf_a = spark.table("raw_df_a").toPandas()

    # Make assertions on the data
    assert pdf_a.query("id == 10")["bpl"].iloc[0] == 460
    assert pdf_a.query("id == 30")["bpl"].iloc[0] == 540

    assert up_pdf_a.query("id == 10")["bpl"].iloc[0] == 460
    assert up_pdf_a.query("id == 50")["bpl"].iloc[0] == 710


def test_step_0_datasource_file_csv(
    preprocessing, spark, preprocessing_conf, input_data_dir_path
):
    """Test preprocessing step 0 to ensure that temporary raw_df_unpartitioned_(a/b) tables are created (exact copies of datasources from config). Also test that the presistent raw_df_(a/b) tables are created. Should be same as raw datasources with filters applied"""

    preprocessing_conf["datasource_a"] = {
        "file": os.path.join(input_data_dir_path, "test_csv_data_a.csv"),
        "alias": "csv_file_test_a",
    }
    preprocessing_conf["datasource_b"] = {
        "file": os.path.join(input_data_dir_path, "test_csv_data_b.csv"),
        "alias": "csv_file_test_b",
    }

    # Run the preprocessing step (this will use the test data)
    preprocessing.run_step(0)

    # Create pandas DFs of the step_0 preprocessed test data
    up_pdf_a = spark.table("raw_df_unpartitioned_a").toPandas()
    up_pdf_b = spark.table("raw_df_unpartitioned_b").toPandas()

    pdf_a = spark.table("raw_df_a").toPandas()
    pdf_b = spark.table("raw_df_b").toPandas()

    # Make assertions on the data
    assert pdf_a.query("id == 10")["bpl"].iloc[0] == 120
    assert pdf_a.query("id == 30")["bpl"].iloc[0] == 360

    assert pdf_b.query("id == 30")["bpl"].iloc[0] == 540
    assert pdf_b.query("id == 50")["bpl"].iloc[0] == 710

    assert up_pdf_a.query("id == 20")["bpl"].iloc[0] == 240
    assert up_pdf_a.query("id == 10")["bpl"].iloc[0] == 120

    assert up_pdf_b.query("id == 10")["bpl"].iloc[0] == 460
    assert up_pdf_b.query("id == 30")["bpl"].iloc[0] == 540


def test_step_0_filters_training_data(preprocessing, spark, preprocessing_conf):
    """Test filter run in preprocessing step 0 which selects any person in a household which has a person who is also represented in the test data"""

    # create some training data we can use for testing
    td_schema = StructType(
        [StructField("id_a", LongType(), True), StructField("id_b", LongType(), True)]
    )
    data_training = [{"id_a": 10, "id_b": 30}, {"id_a": 20, "id_b": 40}]
    td = spark.createDataFrame(data_training, schema=td_schema)

    # create the training_data table
    preprocessing.run_register_python(
        name="training_data", func=lambda: td, persist=True
    )

    # add a filter to the config to filter only for households in the training data set
    preprocessing_conf["filter"] = [{"training_data_subset": True}]
    assert preprocessing_conf["filter"] != []

    # run the preprocessing step which includes filtering as a function
    preprocessing.run_step(0)

    # Create pandas DFs of the step_0 preprocessed test data
    pdf_a = spark.table("raw_df_a").toPandas()
    pdf_b = spark.table("raw_df_b").toPandas()

    # Make assertions on the data
    assert len(pdf_a.id) == 3
    assert len(pdf_b.id) == 1
    assert pdf_a.query("id == 20")["namelast"].iloc[0] == "Mc Last"
    assert pd.isnull(pdf_b.query("id == 30")["namelast"].iloc[0])


def test_step_0_filters_expression(preprocessing, spark, preprocessing_conf):
    """Test a filter run in preprocessing step 0 which selects rows from the raw data according to an expression"""

    # overwrite the config filter value to include only an expression type filter
    preprocessing_conf["filter"] = [
        {"expression": "namelast is not null and namelast != ''"}
    ]
    assert preprocessing_conf["filter"] != []

    # run the preprocessing step which includes filtering as a function
    preprocessing.run_step(0)

    # create pandas DFs of the step_0 preprocessed test data
    pdf_a = spark.table("raw_df_a").toPandas()
    pdf_b = spark.table("raw_df_b").toPandas()

    # make assertions on the data
    assert len(pdf_a.id) == 2
    assert len(pdf_b.id) == 1
    assert not pdf_a.namelast.isnull().values.any()
    assert not pdf_b.namelast.isnull().values.any()


def test_step_0_filters_expression_and_household(
    preprocessing, spark, preprocessing_conf
):
    """Test a filter run in preprocessing step 0 which selects rows from the raw data according to an expression AND includes any other entries with the same household ID (serialp)"""

    # overwrite the config filter value to include an expression filter which includes a household:true argument.
    # Note: in a household filter, the variables for household ID in each input file must be specified in the filter.  (This is the "serial_a": "serialp" bit below.)

    preprocessing_conf["filter"] = [
        {
            "expression": "namelast == 'Mc Last' or namelast == 'Name'",
            "household": True,
            "serial_a": "serialp",
            "serial_b": "serialp",
        }
    ]

    # run the preprocessing step which includes filtering as a function
    preprocessing.run_step(0)

    # create pandas DFs of the step_0 preprocessed test data
    pdf_a = spark.table("raw_df_a").toPandas()
    pdf_b = spark.table("raw_df_b").toPandas()

    # make assertions on the data
    assert len(pdf_a.id) == 2
    assert len(pdf_b.id) == 1
    assert not pdf_a.namelast.isnull().values.any()
    assert not pdf_b.namelast.isnull().values.any()


def test_step_0_filters_datasource(preprocessing, spark, preprocessing_conf):
    """Test a filter run in preprocessing step 0 which selects rows from the raw data according
    to an expression AND only applies the expression to a specified datasource (a or b)"""

    # overwrite the config filter value to include an expression filter which includes a datasource argument.
    preprocessing_conf["filter"] = [
        {"expression": "id == 30", "datasource": "a"},
        {"expression": "id == 10", "datasource": "b"},
    ]
    assert preprocessing_conf["filter"] != []

    # run the preprocessing step which includes filtering as a function
    preprocessing.run_step(0)

    # create pandas DFs of the step_0 preprocessed test data
    pdf_a = spark.table("raw_df_a").toPandas()
    pdf_b = spark.table("raw_df_b").toPandas()

    # make assertions on the data
    assert len(pdf_a.id) == 1
    assert len(pdf_b.id) == 1
    assert pdf_a.id[0] == 30
    assert pdf_b.id[0] == 10


def test_step_0_check_for_all_spaces_unrestricted_data(
    preprocessing, spark, preprocessing_conf_all_space_columns, capsys
):
    """Tests the check in preprocessing that looks for all-space columns, as found in unrestricted data files."""
    with pytest.raises(DataError, match=r"\: namelast, street\."):
        preprocessing.run_step(0)


def test_step_1_transform_attach_variable(
    preprocessing, spark, preprocessing_conf, region_code_path
):
    """Test the transform "attach_variable" -- used to add a feature column from CSV data"""
    preprocessing_conf["column_mappings"] = [{"column_name": "bpl"}]
    preprocessing_conf["feature_selections"] = [
        {
            "input_column": "bpl",
            "output_column": "region",
            "transform": "attach_variable",
            "region_dict": region_code_path,
            "col_to_join_on": "bpl",
            "col_to_add": "region",
            "null_filler": 99,
            "col_type": "int",
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert len(pdf_a.region) == 3
    assert len(pdf_b.region) == 3
    assert pdf_a.query("id == 10")["region"].iloc[0] == 6
    assert pdf_a.query("id == 30")["region"].iloc[0] == 99
    assert pdf_b.query("id == 10")["region"].iloc[0] == 8
    assert pdf_b.query("id == 50")["region"].iloc[0] == 99


def test_step_1_transform_hash(preprocessing, spark, preprocessing_conf):
    """Test the transform "attach_variable" -- used to add a feature column from CSV data"""
    preprocessing_conf["column_mappings"].append(
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
        }
    )

    preprocessing_conf["feature_selections"] = [
        {
            "input_column": "namefrst_clean",
            "output_column": "namefrst_bigrams",
            "transform": "bigrams",
            "no_first_pad": True,
        },
        {
            "input_column": "namefrst_bigrams",
            "output_column": "namefrst_bigrams_hash",
            "transform": "hash",
            "number": 5,
        },
    ]

    preprocessing_conf["filter"] = [
        {"expression": "namefrst is not null and namefrst != ''"}
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert len(pdf_a.namefrst_bigrams_hash_count) == 3
    assert len(pdf_a.namefrst_bigrams_hash) == 3

    assert len(pdf_b.namefrst_bigrams_hash_count) == 2
    assert len(pdf_b.namefrst_bigrams_hash) == 2

    assert len(pdf_a.query("id == 10")["namefrst_bigrams_hash_count"].iloc[0]) == 17
    assert len(pdf_a.query("id == 10")["namefrst_bigrams_hash"].iloc[0]) == 5

    assert len(pdf_b.query("id == 10")["namefrst_bigrams_hash_count"].iloc[0]) == 6
    assert len(pdf_b.query("id == 10")["namefrst_bigrams_hash"].iloc[0]) == 5


def test_step_1_transform_override(
    preprocessing, spark, preprocessing_conf, region_code_path
):
    """Test a column mapping with transform and OVERRIDE column for a specified datasource -- used to generate a feature from a column from one datasource and use a pre-existing column from the other datasource"""
    preprocessing_conf["column_mappings"] = [
        {
            "column_name": "namefrst",
            "alias": "namefrst_mid_init",
            "override_column_b": "namemiddle",
            "transforms": [
                {"type": "lowercase_strip"},
                {"type": "rationalize_name_words"},
                {"type": "remove_qmark_hyphen"},
                {"type": "replace_apostrophe"},
                {
                    "type": "remove_suffixes",
                    "values": ["jr", "sr", "ii", "iii", "iv", "v", "vi", "vii", "viii"],
                },
                {"type": "remove_prefixes", "values": ["mr"]},
                {"type": "remove_alternate_names"},
                {"type": "condense_strip_whitespace"},
                {"type": "split"},
                {"type": "array_index", "value": 1},
            ],
            "override_transforms": [
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
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == 10")["namefrst_mid_init"].iloc[0] == "m"
    assert pdf_a.query("id == 20")["namefrst_mid_init"].iloc[0] == "marc"
    assert pd.isnull(pdf_a.query("id == 30")["namefrst_mid_init"].iloc[0])
    assert pdf_b.query("id == 10")["namefrst_mid_init"].iloc[0] == "m"
    assert pdf_b.query("id == 50")["namefrst_mid_init"].iloc[0] == "marc"
    assert pd.isnull(pdf_b.query("id == 30")["namefrst_mid_init"].iloc[0])


def test_step_1_transforms_namefrst_soundex(
    preprocessing, spark, preprocessing_conf, region_code_path
):
    """Test a column mapping with string-based transforms on firstname removing middle name"""
    preprocessing_conf["column_mappings"] = [
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
                {"type": "remove_prefixes", "values": ["mr"]},
                {"type": "remove_alternate_names"},
                {"type": "remove_one_letter_names"},
                {"type": "condense_strip_whitespace"},
                {"type": "split"},
                {"type": "array_index", "value": 0},
            ],
        }
    ]

    preprocessing_conf["feature_selections"] = [
        {
            "input_column": "namefrst_std",
            "output_column": "namefrst_soundex",
            "transform": "soundex",
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == 10")["namefrst_std"].iloc[0] == "john"
    assert pdf_a.query("id == 20")["namefrst_std"].iloc[0] == "marc"
    assert pdf_a.query("id == 30")["namefrst_std"].iloc[0] == "jon"
    assert pdf_a.query("namefrst_std == 'john'")["namefrst_soundex"].iloc[0] == "J500"
    assert pdf_a.query("namefrst_std == 'marc'")["namefrst_soundex"].iloc[0] == "M620"

    assert pdf_b.query("id == 10")["namefrst_std"].iloc[0] == "john"
    assert pd.isnull(pdf_b.query("id == 30")["namefrst_std"].iloc[0])
    assert pdf_b.query("id == 50")["namefrst_std"].iloc[0] == "jean"
    assert pdf_b.query("namefrst_std == 'jean'")["namefrst_soundex"].iloc[0] == "J500"


def test_step_1_transforms_prefix_suffix(
    preprocessing, spark, preprocessing_conf, region_code_path
):
    """Test a column mapping with different string transforms"""
    preprocessing_conf["column_mappings"] = [
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
                {"type": "remove_prefixes", "values": ["mr"]},
                {"type": "remove_alternate_names"},
                {"type": "condense_strip_whitespace"},
            ],
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == 10")["namefrst_std"].iloc[0] == "john m"
    assert pdf_a.query("id == 20")["namefrst_std"].iloc[0] == "j marc ell"
    assert pdf_a.query("id == 30")["namefrst_std"].iloc[0] == "jon"

    assert pdf_b.query("id == 10")["namefrst_std"].iloc[0] == "john"
    assert pd.isnull(pdf_b.query("id == 30")["namefrst_std"].iloc[0])
    assert pdf_b.query("id == 50")["namefrst_std"].iloc[0] == "jean"


def test_step_1_transform_adv_str(preprocessing, spark, preprocessing_conf):
    """Test a column mapping with remaining transforms"""
    preprocessing_conf["column_mappings"] = [
        {
            "column_name": "namelast",
            "alias": "namelast_std",
            "transforms": [
                {"type": "lowercase_strip"},
                {
                    "type": "condense_prefixes",
                    "values": ["mc", "mac", "o", "de", "van", "di"],
                },
            ],
        },
        {"column_name": "sex"},
        {
            "column_name": "sex",
            "alias": "sex_int",
            "transforms": [{"type": "cast_as_int"}],
        },
        {
            "column_name": "namelast_std",
            "alias": "namelast_std_len",
            "transforms": [{"type": "length"}],
        },
        {
            "alias": "namelast_init",
            "column_name": "namelast_std",
            "transforms": [{"type": "substring", "values": [0, 1]}],
        },
        {
            "column_name": "sex_int",
            "alias": "sex_mapped",
            "transforms": [{"type": "mapping", "mappings": {"1": "M", "2": "F"}}],
        },
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()

    assert pdf_a.query("id == 20")["namelast_std"].iloc[0] == "mclast"
    assert pdf_a.query("id == 10")["sex_int"].iloc[0] == 1
    assert pdf_a.query("id == 20")["namelast_std_len"].iloc[0] == 6
    assert pdf_a.query("id == 20")["namelast_init"].iloc[0] == "m"
    assert pdf_a.query("id == 30")["namelast_init"].iloc[0] == "l"
    assert pdf_a.query("id == 10")["sex_mapped"].iloc[0] == "M"
    assert pdf_a.query("id == 20")["sex_mapped"].iloc[0] == "F"


def test_step_1_transform_neighbor_agg(
    preprocessing, spark, preprocessing_conf_household_data
):
    """Test neighbor_aggregate transform on data containing households"""
    preprocessing_conf_household_data["feature_selections"] = [
        {
            "output_column": "namelast_neighbors",
            "input_column": "namelast_clean",
            "transform": "neighbor_aggregate",
            "neighborhood_column": "enumdist",
            "sort_column": "serial",
            "range": 5,
        }
    ]

    preprocessing.run_step(0)

    rda = spark.table("raw_df_a")
    rda = (
        rda.withColumn("enumdist_tmp", rda["enumdist"].cast("bigint"))
        .drop("enumdist")
        .withColumnRenamed("enumdist_tmp", "enumdist")
    )
    rda = (
        rda.withColumn("serial_tmp", rda["serial"].cast("bigint"))
        .drop("serial")
        .withColumnRenamed("serial_tmp", "serial")
    )
    rda = (
        rda.withColumn("pernum_tmp", rda["pernum"].cast("bigint"))
        .drop("pernum")
        .withColumnRenamed("pernum_tmp", "pernum")
    )
    rda.write.mode("overwrite").saveAsTable("raw_df_a_tmp")
    spark.sql("drop table raw_df_a")
    spark.sql("alter table raw_df_a_tmp rename to raw_df_a")

    rdb = spark.table("raw_df_b")
    rdb = (
        rdb.withColumn("enumdist_tmp", rdb["enumdist"].cast("bigint"))
        .drop("enumdist")
        .withColumnRenamed("enumdist_tmp", "enumdist")
    )
    rdb = (
        rdb.withColumn("serial_tmp", rdb["serial"].cast("bigint"))
        .drop("serial")
        .withColumnRenamed("serial_tmp", "serial")
    )
    rdb = (
        rdb.withColumn("pernum_tmp", rdb["pernum"].cast("bigint"))
        .drop("pernum")
        .withColumnRenamed("pernum_tmp", "pernum")
    )
    rdb.write.mode("overwrite").saveAsTable("raw_df_b_tmp")
    spark.sql("drop table raw_df_b")
    spark.sql("alter table raw_df_b_tmp rename to raw_df_b")

    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    wilson_a_nbrs = sorted(
        pdf_a.query("namelast_clean == 'wilson'")["namelast_neighbors"].iloc[0]
    )
    wilson_b_nbrs = sorted(
        pdf_b.query("namelast_clean == 'wilson'")["namelast_neighbors"].iloc[0]
    )
    lord_a_nbrs = sorted(
        pdf_a.query("namelast_clean == 'lord'")["namelast_neighbors"].iloc[0]
    )

    assert lord_a_nbrs == ["allen", "dekay", "foster", "graham", "taylor", "thorpe"]

    assert wilson_a_nbrs == [
        "bierhahn",
        "chambers",
        "cleveland",
        "collins",
        "flemming",
        "graham",
        "harvey",
        "mclean",
        "seward",
        "shields",
    ]
    assert wilson_b_nbrs == [
        "bierhahn",
        "cleveland",
        "collins",
        "dekay",
        "flemming",
        "graham",
        "harvey",
        "mclean",
        "seward",
        "shields",
    ]


def test_step_1_transform_attach_family_col(
    preprocessing, spark, preprocessing_conf_household_data
):
    """Test attach_family_col transform on data containing households"""
    preprocessing_conf_household_data["feature_selections"] = [
        {
            "output_col": "spouse_namefrst",
            "transform": "attach_family_col",
            "other_col": "namefrst_clean",
            "person_pointer": "sploc",
            "family_id": "serial",
            "person_id": "pernum",
        },
        {
            "output_col": "spouse_bpl",
            "transform": "attach_family_col",
            "other_col": "bpl",
            "person_pointer": "sploc",
            "family_id": "serial",
            "person_id": "pernum",
        },
        {
            "output_col": "father_namefrst",
            "transform": "attach_family_col",
            "other_col": "namefrst_clean",
            "person_pointer": "poploc",
            "family_id": "serial",
            "person_id": "pernum",
        },
        {
            "output_col": "mother_namefrst",
            "transform": "attach_family_col",
            "other_col": "namefrst_clean",
            "person_pointer": "momloc",
            "family_id": "serial",
            "person_id": "pernum",
        },
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)
    pdf_a = (
        spark.table("prepped_df_a")
        .toPandas()[
            [
                "serial",
                "pernum",
                "namefrst_clean",
                "namelast_clean",
                "spouse_namefrst",
                "spouse_bpl",
                "father_namefrst",
                "mother_namefrst",
            ]
        ]
        .sort_values(["serial", "pernum"])
    )
    pdf_b = (
        spark.table("prepped_df_b")
        .toPandas()[
            [
                "serial",
                "pernum",
                "namefrst_clean",
                "namelast_clean",
                "spouse_namefrst",
                "spouse_bpl",
                "father_namefrst",
                "mother_namefrst",
            ]
        ]
        .sort_values(["serial", "pernum"])
    )

    assert (
        pdf_a.query("namefrst_clean == 'jezebel'")["spouse_namefrst"].iloc[0] == "job"
    )
    assert (
        pdf_a.query("namefrst_clean == 'jezebel'")["mother_namefrst"].iloc[0] == "eliza"
    )
    assert (
        pdf_a.query("namefrst_clean == 'willie may'")["father_namefrst"].iloc[0]
        == "wm h"
    )
    assert (
        pdf_a.query("namefrst_clean == 'willie may'")["mother_namefrst"].iloc[0]
        == "martha"
    )

    assert (
        pdf_b.query("namefrst_clean == 'jezebel'")["spouse_namefrst"].iloc[0] == "job"
    )
    assert pdf_b.query("namefrst_clean == 'jezebel'")["spouse_bpl"].iloc[0] == 10
    assert (
        pdf_b.query("namefrst_clean == 'jezebel'")["mother_namefrst"].iloc[0] == "eliza"
    )
    assert pd.isnull(
        pdf_b.query("namefrst_clean == 'esther'")["spouse_namefrst"].iloc[0]
    )


def test_step_1_transform_calc_nativity(
    preprocessing, spark, preprocessing_conf_19thc_nativity_conf
):
    """Test attach_family_col transform on data containing households"""

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas().sort_values(["serial", "pernum"])

    pdf_b = spark.table("prepped_df_b").toPandas().sort_values(["serial", "pernum"])

    assert set(pdf_a["nativity"].tolist()) == {1, 2, 3, 4, 5}
    assert set(pdf_a["test_nativity"].tolist()) == {0}
    assert set(pdf_a["mbpl"].tolist()) == {999}
    assert set(pdf_a["key_mbpl_range_b"].tolist()) == {0}
    assert set(pdf_a["mbpl_range_b"].tolist()) == {0}
    assert pdf_a["mother_nativity"].equals(pdf_a["key_mother_nativity"])
    assert pdf_a["key_mbpl"].equals(pdf_a["mbpl_calc"])
    assert pdf_a["key_fbpl"].equals(pdf_a["fbpl_calc"])
    assert pdf_a["mbpl_range"].equals(pdf_a["key_mbpl_range"])
    assert set(pdf_a["key_mbpl_range"].tolist()) == {0, 1, 2}
    assert pdf_a["key_nativity_calc"].equals(pdf_a["nativity_calc"])

    assert set(pdf_b["nativity"].tolist()) == {0}
    assert set(pdf_b["test_nativity"].tolist()) == {0, 1, 2, 3, 4, 5}
    assert pdf_b["key_nativity_calc"].equals(pdf_b["test_nativity"])
    assert set(pdf_b["key_mbpl_range"].tolist()) == {0}
    assert set(pdf_b["key_mother_nativity"].tolist()) == {0}
    assert pdf_b["mbpl_range_b"].equals(pdf_b["key_mbpl_range_b"])
    assert set(pdf_b["mother_nativity"].tolist()) == {0}
    assert pdf_b["key_mbpl"].equals(pdf_b["mbpl_calc"])
    assert pdf_b["key_fbpl"].equals(pdf_b["fbpl_calc"])
    assert pdf_b["mbpl_range"].equals(pdf_b["mbpl_range_b"])
    assert pdf_b["mbpl_range"].equals(pdf_b["key_mbpl_range_b"])
    assert pdf_b["key_nativity_calc"].equals(pdf_b["nativity_calc"])

    assert pdf_a["nativity_calc"].tolist() == [
        0,
        0,
        1,
        0,
        5,
        0,
        2,
        0,
        5,
        3,
        5,
        0,
        2,
        5,
        5,
        0,
        0,
    ]
    assert pdf_b["nativity_calc"].tolist() == [
        0,
        0,
        1,
        0,
        5,
        0,
        2,
        0,
        5,
        3,
        5,
        5,
        4,
        5,
        5,
        1,
    ]


def test_step_1_transform_related_individuals(
    preprocessing, spark, preprocessing_conf_household_data
):
    """Test attach related_individuals transform"""
    preprocessing_conf_household_data["feature_selections"] = [
        {
            "output_col": "namefrst_related",
            "input_col": "namefrst_clean",
            "transform": "related_individuals",
            "family_id": "serial",
            "relate_col": "relate",
            "top_code": 10,
            "bottom_code": 3,
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = (
        spark.table("prepped_df_a")
        .toPandas()[
            [
                "serial",
                "pernum",
                "relate",
                "namefrst_clean",
                "namelast_clean",
                "namefrst_related",
            ]
        ]
        .sort_values(["serial", "pernum"])
    )
    pdf_b = (
        spark.table("prepped_df_b")
        .toPandas()[
            [
                "serial",
                "pernum",
                "relate",
                "namefrst_clean",
                "namelast_clean",
                "namefrst_related",
            ]
        ]
        .sort_values(["serial", "pernum"])
    )

    assert pdf_a.query("namefrst_clean == 'otillia'")["namefrst_related"].iloc[0] == []
    assert pdf_a.query("namefrst_clean == 'j clauson'")["namefrst_related"].iloc[0] == [
        "eugene"
    ]

    assert sorted(
        pdf_b.query("namefrst_clean == 'job'")["namefrst_related"].iloc[0]
    ) == ["eliza", "jo", "mary"]


def test_step_1_transform_related_individual_rows(
    preprocessing, spark, preprocessing_conf_household_data
):
    """Test attach related_individuals transform"""
    preprocessing_conf_household_data["feature_selections"] = [
        {
            "output_col": "spouse_namefrst",
            "transform": "attach_family_col",
            "other_col": "namefrst_clean",
            "person_pointer": "sploc",
            "family_id": "serial",
            "person_id": "pernum",
        },
        {
            "output_col": "namefrst_related_rows",
            "input_cols": ["namefrst_clean", "bpl", "sex"],
            "transform": "related_individual_rows",
            "family_id": "serial",
            "filters": [
                {"column": "relate", "max": 10, "min": 3},
                {"column": "pernum", "max": 4, "min": 0, "dataset": "b"},
            ],
        },
        {
            "output_col": "unrelated_rows",
            "input_cols": ["namefrst_clean", "bpl", "sex"],
            "transform": "related_individual_rows",
            "family_id": "serial",
            "filters": [{"column": "relate", "max": 999, "min": 11}],
        },
        {
            "output_column": "namelast_neighbors",
            "input_column": "namelast_clean",
            "transform": "neighbor_aggregate",
            "neighborhood_column": "enumdist",
            "sort_column": "serial",
            "range": 5,
        },
    ]

    preprocessing.run_step(0)

    rda = spark.table("raw_df_a")
    rda = (
        rda.withColumn("enumdist_tmp", rda["enumdist"].cast("bigint"))
        .drop("enumdist")
        .withColumnRenamed("enumdist_tmp", "enumdist")
    )
    rda = (
        rda.withColumn("serial_tmp", rda["serial"].cast("bigint"))
        .drop("serial")
        .withColumnRenamed("serial_tmp", "serial")
    )
    rda = (
        rda.withColumn("pernum_tmp", rda["pernum"].cast("bigint"))
        .drop("pernum")
        .withColumnRenamed("pernum_tmp", "pernum")
    )
    rda = (
        rda.withColumn("bpl_tmp", rda["bpl"].cast("bigint"))
        .drop("bpl")
        .withColumnRenamed("bpl_tmp", "bpl")
    )
    rda = (
        rda.withColumn("sex_tmp", rda["sex"].cast("bigint"))
        .drop("sex")
        .withColumnRenamed("sex_tmp", "sex")
    )
    rda = (
        rda.withColumn("relate_tmp", rda["relate"].cast("bigint"))
        .drop("relate")
        .withColumnRenamed("relate_tmp", "relate")
    )

    rda.write.mode("overwrite").saveAsTable("raw_df_a_tmp")
    spark.sql("drop table raw_df_a")
    spark.sql("alter table raw_df_a_tmp rename to raw_df_a")

    rdb = spark.table("raw_df_b")
    rdb = (
        rdb.withColumn("enumdist_tmp", rdb["enumdist"].cast("bigint"))
        .drop("enumdist")
        .withColumnRenamed("enumdist_tmp", "enumdist")
    )
    rdb = (
        rdb.withColumn("serial_tmp", rdb["serial"].cast("bigint"))
        .drop("serial")
        .withColumnRenamed("serial_tmp", "serial")
    )
    rdb = (
        rdb.withColumn("pernum_tmp", rdb["pernum"].cast("bigint"))
        .drop("pernum")
        .withColumnRenamed("pernum_tmp", "pernum")
    )
    rdb = (
        rdb.withColumn("bpl_tmp", rdb["bpl"].cast("bigint"))
        .drop("bpl")
        .withColumnRenamed("bpl_tmp", "bpl")
    )
    rdb = (
        rdb.withColumn("sex_tmp", rdb["sex"].cast("bigint"))
        .drop("sex")
        .withColumnRenamed("sex_tmp", "sex")
    )
    rdb = (
        rdb.withColumn("relate_tmp", rdb["relate"].cast("bigint"))
        .drop("relate")
        .withColumnRenamed("relate_tmp", "relate")
    )

    rdb.write.mode("overwrite").saveAsTable("raw_df_b_tmp")
    spark.sql("drop table raw_df_b")
    spark.sql("alter table raw_df_b_tmp rename to raw_df_b")

    preprocessing.run_step(1)
    select_cols = [
        "serial",
        "pernum",
        "relate",
        "namefrst_clean",
        "namelast_clean",
        "spouse_namefrst",
        "namefrst_related_rows",
        "unrelated_rows",
        "namelast_neighbors",
    ]

    pdf_a = (
        spark.table("prepped_df_a")
        .toPandas()[select_cols]
        .sort_values(["serial", "pernum"])
    )
    pdf_b = (
        spark.table("prepped_df_b")
        .toPandas()[select_cols]
        .sort_values(["serial", "pernum"])
    )

    wilson_a_nbrs = sorted(
        pdf_a.query("namelast_clean == 'wilson'")["namelast_neighbors"].iloc[0]
    )

    assert wilson_a_nbrs == [
        "bierhahn",
        "chambers",
        "cleveland",
        "collins",
        "flemming",
        "graham",
        "harvey",
        "mclean",
        "seward",
        "shields",
    ]

    assert (
        pdf_a.query("namefrst_clean == 'otillia'")["namefrst_related_rows"].iloc[0]
        == []
    )

    row_a = pdf_a.query("namefrst_clean == 'j clauson'")["namefrst_related_rows"].iloc[
        0
    ][0]
    assert row_a.namefrst_clean == "eugene"
    assert row_a.bpl == 10
    assert row_a.sex == 1

    assert (
        pdf_a.query("namefrst_clean == 'jezebel'")["spouse_namefrst"].iloc[0] == "job"
    )

    assert (
        len(pdf_b.query("namefrst_clean == 'job'")["namefrst_related_rows"].iloc[0])
        == 2
    )
    assert (
        pdf_a.query("serial == 2485411 and pernum == 1").unrelated_rows.iloc[0][0][0]
        == "anne"
    )
    assert (
        len(
            pdf_a.query("serial == 2492741 and pernum == 4").namefrst_related_rows.iloc[
                0
            ]
        )
        == 2
    )
    assert (
        len(
            pdf_b.query("serial == 2492741 and pernum == 4").namefrst_related_rows.iloc[
                0
            ]
        )
        == 1
    )


def test_step_1_transform_popularity(
    preprocessing, spark, preprocessing_conf_popularity
):
    """Test attach related_individuals transform"""
    preprocessing_conf_popularity["feature_selections"] = [
        {
            "checkpoint": True,
            "input_cols": ["namefrst", "namelast", "bpl", "sex"],
            "range_col": "birthyr",
            "range_val": 3,
            "output_col": "ncount",
            "transform": "popularity",
        },
        {
            "checkpoint": True,
            "input_cols": ["namefrst", "bpl", "sex", "birthyr"],
            "output_col": "fname_pop",
            "transform": "popularity",
        },
        {
            "checkpoint": True,
            "output_col": "byr_pop",
            "range_col": "birthyr",
            "range_val": 3,
            "transform": "popularity",
        },
        {
            "input_col": "ncount",
            "output_col": "ncount2",
            "transform": "power",
            "exponent": 2,
        },
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").orderBy("id").toPandas()

    assert pdf_a.query("id == 0")["ncount"].iloc[0] == 3
    assert pdf_a.query("id == 1")["ncount"].iloc[0] == 1
    assert pdf_a.query("id == 2")["ncount"].iloc[0] == 2
    assert pdf_a.query("id == 3")["ncount"].iloc[0] == 2
    assert pdf_a.query("id == 4")["ncount"].iloc[0] == 1
    assert pdf_a.query("id == 5")["ncount"].iloc[0] == 1
    assert pdf_a.query("id == 6")["ncount"].iloc[0] == 1
    assert pdf_a.query("id == 7")["ncount"].iloc[0] == 1

    assert pdf_a.query("id == 0")["ncount2"].iloc[0] == 9
    assert pdf_a.query("id == 1")["ncount2"].iloc[0] == 1
    assert pdf_a.query("id == 2")["ncount2"].iloc[0] == 4

    assert pdf_a.query("id == 0")["fname_pop"].iloc[0] == 2
    assert pdf_a.query("id == 1")["fname_pop"].iloc[0] == 1
    assert pdf_a.query("id == 2")["fname_pop"].iloc[0] == 1
    assert pdf_a.query("id == 3")["fname_pop"].iloc[0] == 1
    assert pdf_a.query("id == 4")["fname_pop"].iloc[0] == 2
    assert pdf_a.query("id == 5")["fname_pop"].iloc[0] == 1
    assert pdf_a.query("id == 6")["fname_pop"].iloc[0] == 1
    assert pdf_a.query("id == 7")["fname_pop"].iloc[0] == 1

    assert pdf_a.query("id == 0")["byr_pop"].iloc[0] == 7
    assert pdf_a.query("id == 1")["byr_pop"].iloc[0] == 1
    assert pdf_a.query("id == 2")["byr_pop"].iloc[0] == 4
    assert pdf_a.query("id == 3")["byr_pop"].iloc[0] == 6


def test_step_1_transforms_adv_calc(preprocessing, spark, preprocessing_conf):
    """Test more column mapping with transforms"""
    preprocessing_conf["column_mappings"] = [
        {
            "column_name": "serialp",
            "alias": "serialp_add_a",
            "transforms": [{"type": "concat_to_a", "value": "_a"}],
        },
        {
            "column_name": "bpl",
            "alias": "bpl_add_b",
            "transforms": [{"type": "concat_to_b", "value": 1}],
        },
        {
            "alias": "bpl",
            "column_name": "bpl",
        },
        {
            "alias": "sex",
            "column_name": "sex",
        },
        {
            "alias": "concat_bpl_sex",
            "column_name": "bpl",
            "transforms": [{"type": "concat_two_cols", "column_to_append": "sex"}],
        },
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == 10")["serialp_add_a"].iloc[0] == "A_a"
    assert pdf_b.query("id == 10")["serialp_add_a"].iloc[0] == "C"
    assert pdf_a.query("id == 20")["bpl_add_b"].iloc[0] == 200
    assert pdf_b.query("id == 30")["bpl_add_b"].iloc[0] == "5001"
    assert pdf_a.query("id == 10")["concat_bpl_sex"].iloc[0] == "1001"
    assert pdf_b.query("id == 50")["concat_bpl_sex"].iloc[0] == "7002"


def test_step_1_transforms_expand(preprocessing, spark, preprocessing_conf):
    """Test transform expand"""
    preprocessing_conf["column_mappings"] = [
        {"column_name": "age"},
        {
            "column_name": "age",
            "alias": "age_expand_3",
            "transforms": [{"type": "expand", "value": 3}],
        },
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()

    assert pdf_a.query("id == 10")["age_expand_3"].iloc[0] == [
        20,
        21,
        22,
        23,
        24,
        25,
        26,
    ]


def test_step_1_override(preprocessing, spark, preprocessing_conf):
    """Test column override"""
    preprocessing_conf["column_mappings"] = [
        {
            "column_name": "serialp",
            "override_column_a": "serialp",
            "override_transforms": [{"type": "lowercase_strip"}],
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == 30")["serialp"].iloc[0] == "b"
    assert pdf_b.query("id == 50")["serialp"].iloc[0] == "E"


def test_step_1_set_values_a_explicitly(preprocessing, spark, preprocessing_conf):
    """Test setting a column value explicitly"""
    preprocessing_conf["column_mappings"] = [
        {"column_name": "serialp", "alias": "serialp", "set_value_column_a": "c"}
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == 30")["serialp"].iloc[0] == "c"
    assert pdf_b.query("id == 50")["serialp"].iloc[0] == "E"


def test_step_1_set_values_b_explicitly(preprocessing, spark, preprocessing_conf):
    """Test setting b column value explicitly"""
    preprocessing_conf["column_mappings"] = [
        {"column_name": "serialp", "alias": "serialp", "set_value_column_b": "a"}
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == 30")["serialp"].iloc[0] == "B"
    assert pdf_b.query("id == 50")["serialp"].iloc[0] == "a"


def test_step_1_substitution(
    preprocessing,
    spark,
    preprocessing_conf_simple_names,
    substitutions_womens_names_path,
    substitutions_mens_names_path,
):
    """Test text substitution"""
    preprocessing_conf_simple_names["column_mappings"] = [
        {
            "column_name": "namefrst",
            "alias": "namefrst_std",
            "transforms": [{"type": "lowercase_strip"}],
        },
        {"column_name": "namefrst"},
        {"column_name": "sex"},
    ]

    preprocessing_conf_simple_names["substitution_columns"] = [
        {
            "column_name": "namefrst_std",
            "substitutions": [
                {
                    "join_column": "sex",
                    "join_value": "1",
                    "substitution_file": substitutions_mens_names_path,
                },
                {
                    "join_column": "sex",
                    "join_value": "2",
                    "substitution_file": substitutions_womens_names_path,
                },
            ],
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == '10ah'")["namefrst_std"].iloc[0] == "cat"
    assert pdf_a.query("id == '20bc'")["namefrst_std"].iloc[0] == "bernard"
    assert pdf_a.query("id == '34hi'")["namefrst_std"].iloc[0] == "catherine"
    assert pdf_a.query("id == '54de'")["namefrst_std"].iloc[0] == "kat"
    assert pdf_b.query("id == 'c23'")["namefrst_std"].iloc[0] == "bernard"
    assert pdf_b.query("id == 'd45'")["namefrst_std"].iloc[0] == "catherine"
    assert pdf_b.query("id == 'e77'")["namefrst_std"].iloc[0] == "bernard"


def test_step_1_street_abbrev_substitution(
    preprocessing,
    spark,
    preprocessing_conf_street_names,
    substitutions_street_abbrevs_path,
):
    """Test text substitution"""
    preprocessing_conf_street_names["id_column"] = "histid"
    preprocessing_conf_street_names["column_mappings"] = [
        {
            "column_name": "street",
            "alias": "street_unstd",
            "transforms": [
                {"type": "lowercase_strip"},
                {"type": "remove_qmark_hyphen"},
                {"type": "replace_apostrophe"},
                {"type": "condense_strip_whitespace"},
            ],
        },
        {
            "column_name": "street_unstd",
            "alias": "street_swapped",
            "transforms": [
                {
                    "type": "swap_words",
                    "values": {"bch": "beach", "ctr": "center", "rd": "road"},
                }
            ],
        },
    ]

    preprocessing_conf_street_names["substitution_columns"] = [
        {
            "column_name": "street_unstd",
            "substitutions": [
                {
                    "substitution_file": substitutions_street_abbrevs_path,
                    "regex_word_replace": True,
                }
            ],
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()

    assert pdf_a.query("histid == 'a01'")["street_unstd"].iloc[0] == "turnpike 35"
    assert (
        pdf_a.query("histid == 'b02'")["street_unstd"].iloc[0] == "4th terrace avenue"
    )
    assert pdf_a.query("histid == 'c03'")["street_unstd"].iloc[0] == "4th state"
    assert pdf_a.query("histid == 'd04'")["street_unstd"].iloc[0] == "old boulevard"
    assert pdf_a.query("histid == 'e05'")["street_unstd"].iloc[0] == "old motorway"
    assert pdf_a.query("histid == 'f06'")["street_unstd"].iloc[0] == "miami bch road"
    assert pdf_a.query("histid == 'g07'")["street_unstd"].iloc[0] == "center street"
    assert pdf_a.query("histid == 'g08'")["street_unstd"].iloc[0] == "ctr street"
    assert pdf_a.query("histid == 'i09'")["street_unstd"].iloc[0] == "strstreet"

    assert (
        pdf_a.query("histid == 'f06'")["street_swapped"].iloc[0] == "miami beach road"
    )
    assert pdf_a.query("histid == 'g08'")["street_swapped"].iloc[0] == "center street"


def test_step_1_street_remove_stop_words(
    preprocessing,
    spark,
    preprocessing_conf_street_names,
    substitutions_street_abbrevs_path,
):
    """Test text substitution"""
    preprocessing_conf_street_names["id_column"] = "histid"
    preprocessing_conf_street_names["column_mappings"] = [
        {"column_name": "street", "alias": "street_orig"},
        {
            "column_name": "street",
            "alias": "street_unstd",
            "transforms": [
                {"type": "lowercase_strip"},
                {"type": "remove_qmark_hyphen"},
                {"type": "replace_apostrophe"},
                {"type": "condense_strip_whitespace"},
            ],
        },
        {
            "column_name": "street_unstd",
            "alias": "street_removed",
            "transforms": [
                {
                    "type": "remove_stop_words",
                    "values": [
                        "avn",
                        "blvd",
                        "rd",
                        "road",
                        "street",
                        "str",
                        "ter",
                        "trnpk",
                    ],
                },
                {"type": "condense_strip_whitespace"},
            ],
        },
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()

    assert pdf_a.query("histid == 'a01'")["street_removed"].iloc[0] == "35"
    assert pdf_a.query("histid == 'b02'")["street_removed"].iloc[0] == "4th"
    assert pdf_a.query("histid == 'c03'")["street_removed"].iloc[0] == "4th state"
    assert pdf_a.query("histid == 'd04'")["street_removed"].iloc[0] == "old"
    assert pdf_a.query("histid == 'e05'")["street_removed"].iloc[0] == "old motorway"
    assert pdf_a.query("histid == 'f06'")["street_removed"].iloc[0] == "miami bch"
    assert pdf_a.query("histid == 'g07'")["street_removed"].iloc[0] == "centre"
    assert pdf_a.query("histid == 'g08'")["street_removed"].iloc[0] == "ctr"
    assert pdf_a.query("histid == 'i09'")["street_removed"].iloc[0] == "strstreet"


def test_step_1_divide_by_int_mapping_birthyr(
    preprocessing, spark, preprocessing_conf_birthyr
):
    """Test text substitution"""
    preprocessing_conf_birthyr["id_column"] = "histid"
    preprocessing_conf_birthyr["column_mappings"] = [
        {"column_name": "yearp", "alias": "year"},
        {"column_name": "age"},
        {"column_name": "birthyr", "alias": "raw_birthyr"},
        {
            "column_name": "birthyr",
            "transforms": [{"type": "mapping", "mappings": {1999: ""}}],
        },
        {"column_name": "bpl", "alias": "raw_bpl"},
        {
            "column_name": "bpl",
            "transforms": [
                {"type": "divide_by_int", "value": 100},
                {"type": "get_floor"},
            ],
        },
    ]
    preprocessing_conf_birthyr["feature_selections"] = [
        {
            "input_column": "birthyr",
            "output_column": "birthyr_filled",
            "condition": "case when birthyr is null or birthyr == '' then year - age else birthyr end",
            "transform": "sql_condition",
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()

    assert pdf_a.query("histid == 'a01'")["raw_birthyr"].iloc[0] == 1999
    assert pdf_a.query("histid == 'a01'")["birthyr"].iloc[0] == ""
    assert pdf_a.query("histid == 'a01'")["bpl"].iloc[0] == 33
    assert pdf_a.query("histid == 'b00'")["raw_bpl"].iloc[0] == 4799
    assert pdf_a.query("histid == 'b00'")["bpl"].iloc[0] == 47
    assert pdf_a.query("histid == 'b02'")["bpl"].iloc[0] == 30
    assert pdf_a.query("histid == 'd04'")["bpl"].iloc[0] == 1
    assert pdf_a.query("histid == 'a01'")["birthyr_filled"].iloc[0] == "1864"
    assert pdf_a.query("histid == 'b02'")["birthyr_filled"].iloc[0] == "1858"
    assert pd.isnull(pdf_a.query("histid == 'b00'")["birthyr_filled"].iloc[0])
    assert pdf_a.query("histid == 'c03'")["birthyr_filled"].iloc[0] == "1901"
    assert pdf_a.query("histid == 'd04'")["birthyr_filled"].iloc[0] == "1850"


def test_step_1_fix_bpl(preprocessing, spark, preprocessing_conf_birthyr):
    """Test text substitution"""
    preprocessing_conf_birthyr["id_column"] = "histid"
    preprocessing_conf_birthyr["column_mappings"] = [
        {"column_name": "state1"},
        {"column_name": "state2"},
        {"column_name": "bpl", "alias": "bpl_orig"},
        {
            "column_name": "bpl",
            "alias": "bpl_state_orig",
            "transforms": [
                {"type": "divide_by_int", "value": 100},
                {"type": "get_floor"},
            ],
        },
    ]
    preprocessing_conf_birthyr["feature_selections"] = [
        {
            "input_column": "bpl_orig",
            "output_column": "clean_bpl",
            "condition": """case
                            when state1 == "washington" and state2=="washington"
                            then 5300
                            when (state1 is null or state1 == '') and state2=="washington"
                            then 5300
                            when state1 == "washington" and (state2=='' or state2 is null)
                            then 5300
                            else bpl_orig
                            end""",
            "transform": "sql_condition",
        },
        {
            "input_column": "bpl_state_orig",
            "output_column": "bpl_state",
            "condition": """case
                        when state1 == "washington" and state2=="washington"
                        then 53
                        when (state1 is null or state1 == '') and state2=="washington"
                        then 53
                        when state1 == "washington" and (state2=='' or state2 is null)
                        then 53
                        else bpl_state_orig
                        end""",
            "transform": "sql_condition",
        },
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()

    assert pdf_a.query("histid == 'a01'")["clean_bpl"].iloc[0] == 5300
    assert pdf_a.query("histid == 'b02'")["clean_bpl"].iloc[0] == 5300
    assert pdf_a.query("histid == 'b00'")["clean_bpl"].iloc[0] == 4799
    assert pdf_a.query("histid == 'c03'")["clean_bpl"].iloc[0] == 5300
    assert pdf_a.query("histid == 'd04'")["clean_bpl"].iloc[0] == 100
    assert pdf_a.query("histid == 'a01'")["bpl_state"].iloc[0] == 53
    assert pdf_a.query("histid == 'b02'")["bpl_state"].iloc[0] == 53
    assert pdf_a.query("histid == 'b00'")["bpl_state"].iloc[0] == 47
    assert pdf_a.query("histid == 'c03'")["bpl_state"].iloc[0] == 53
    assert pdf_a.query("histid == 'd04'")["bpl_state"].iloc[0] == 1


def test_step_1_bigrams(preprocessing, spark, preprocessing_conf_simple_names):
    """Test checkpoint transform"""
    preprocessing_conf_simple_names["column_mappings"] = [
        {"column_name": "namefrst"},
        {
            "column_name": "namefrst",
            "alias": "namefrst_std",
            "transforms": [{"type": "lowercase_strip"}],
        },
    ]

    preprocessing_conf_simple_names["feature_selections"] = [
        {
            "input_column": "namefrst_std",
            "output_column": "bigrams_namefrst",
            "transform": "bigrams",
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == '10ah'")["bigrams_namefrst"].iloc[0] == [
        "  c",
        "a t",
        "c a",
    ]
    assert pdf_b.query("id == 'd45'")["bigrams_namefrst"].iloc[0] == [
        "  k",
        "a t",
        "i e",
        "k a",
        "t i",
    ]


def test_step_1_bigrams_no_space(preprocessing, spark, preprocessing_conf_simple_names):
    """Test checkpoint transform"""
    preprocessing_conf_simple_names["column_mappings"] = [
        {"column_name": "namefrst"},
        {
            "column_name": "namefrst",
            "alias": "namefrst_std",
            "transforms": [{"type": "lowercase_strip"}],
        },
    ]

    preprocessing_conf_simple_names["feature_selections"] = [
        {
            "input_column": "namefrst_std",
            "output_column": "bigrams_namefrst",
            "no_first_pad": True,
            "transform": "bigrams",
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == '10ah'")["bigrams_namefrst"].iloc[0] == ["a t", "c a"]
    assert pdf_b.query("id == 'd45'")["bigrams_namefrst"].iloc[0] == [
        "a t",
        "i e",
        "k a",
        "t i",
    ]


def test_step_1_array(preprocessing, spark, preprocessing_conf_simple_names):
    """Test array transform"""
    preprocessing_conf_simple_names["column_mappings"] = [
        {"column_name": "namefrst"},
        {"column_name": "sex"},
    ]

    preprocessing_conf_simple_names["feature_selections"] = [
        {
            "input_columns": ["namefrst", "sex"],
            "output_column": "namefrst_sex_array",
            "transform": "array",
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert pdf_a.query("id == '54de'")["namefrst_sex_array"].iloc[0] == ["Kat", "1"]
    assert pdf_b.query("id == 'e77'")["namefrst_sex_array"].iloc[0] == ["Bernard", "1"]


def test_step_1_union(
    preprocessing, spark, preprocessing_conf_synthetic_household_data
):
    """Test union transform"""
    preprocessing_conf_synthetic_household_data["column_mappings"] = [
        {"column_name": "namefrst"},
        {"column_name": "namelast"},
        {"column_name": "neighbors"},
        {"column_name": "nonfamily_household"},
    ]

    preprocessing_conf_synthetic_household_data["feature_selections"] = [
        {
            "input_columns": ["neighbors", "nonfamily_household"],
            "output_column": "names_union",
            "transform": "union",
        }
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert sorted(pdf_a.query("namefrst == 'jane'")["names_union"].iloc[0]) == [
        "edie",
        "elmer",
        "gerald",
    ]
    assert sorted(pdf_a.query("namefrst == 'janice'")["names_union"].iloc[0]) == [
        "edie"
    ]
    assert sorted(pdf_b.query("namefrst == 'gary'")["names_union"].iloc[0]) == [
        "colleen"
    ]


def test_rel_rows_real_data(spark, preprocessing, preprocessing_conf_rel_rows):
    """Test related_rows data and double threshold comparison blocking criteria"""

    preprocessing.run_step(0)

    rda = spark.table("raw_df_a")
    rda = (
        rda.withColumn("serialp_tmp", rda["serialp"].cast("bigint"))
        .drop("serialp")
        .withColumnRenamed("serialp_tmp", "serialp")
    )
    rda = (
        rda.withColumn("relate_tmp", rda["relate"].cast("bigint"))
        .drop("relate")
        .withColumnRenamed("relate_tmp", "relate")
    )
    rda = (
        rda.withColumn("sex_tmp", rda["sex"].cast("bigint"))
        .drop("sex")
        .withColumnRenamed("sex_tmp", "sex")
    )
    rda = (
        rda.withColumn("age_tmp", rda["age"].cast("bigint"))
        .drop("age")
        .withColumnRenamed("age_tmp", "age")
    )

    rda.write.mode("overwrite").saveAsTable("raw_df_a_tmp")
    spark.sql("drop table raw_df_a")
    spark.sql("alter table raw_df_a_tmp rename to raw_df_a")

    rdb = spark.table("raw_df_b")
    rdb = (
        rdb.withColumn("serialp_tmp", rdb["serialp"].cast("bigint"))
        .drop("serialp")
        .withColumnRenamed("serialp_tmp", "serialp")
    )
    rdb = (
        rdb.withColumn("relate_tmp", rdb["relate"].cast("bigint"))
        .drop("relate")
        .withColumnRenamed("relate_tmp", "relate")
    )
    rdb = (
        rdb.withColumn("sex_tmp", rdb["sex"].cast("bigint"))
        .drop("sex")
        .withColumnRenamed("sex_tmp", "sex")
    )
    rdb = (
        rdb.withColumn("age_tmp", rdb["age"].cast("bigint"))
        .drop("age")
        .withColumnRenamed("age_tmp", "age")
    )

    rdb.write.mode("overwrite").saveAsTable("raw_df_b_tmp")
    spark.sql("drop table raw_df_b")
    spark.sql("alter table raw_df_b_tmp rename to raw_df_b")

    preprocessing.run_step(1)

    pdf_a = spark.table("prepped_df_a").toPandas()
    pdf_b = spark.table("prepped_df_b").toPandas()

    assert (
        len(
            pdf_a.query("histid == 'D1DAEB8F-66F0-435C-8E45-F004D967549D'")[
                "namefrst_related_rows"
            ].iloc[0]
        )
        == 3
    )
    assert (
        len(
            pdf_a.query("histid == 'D1DAEB8F-66F0-435C-8E45-F004D967549D'")[
                "namefrst_related_rows_age_min_5"
            ].iloc[0]
        )
        == 1
    )
    assert (
        len(
            pdf_a.query("histid == 'D1DAEB8F-66F0-435C-8E45-F004D967549D'")[
                "namefrst_related_rows_age_b_min_5"
            ].iloc[0]
        )
        == 3
    )

    assert (
        len(
            pdf_b.query("histid == 'B04F6A33-9A86-4EAF-884B-0BD6107CCDEB'")[
                "namefrst_related_rows"
            ].iloc[0]
        )
        == 7
    )
    assert (
        len(
            pdf_b.query("histid == 'B04F6A33-9A86-4EAF-884B-0BD6107CCDEB'")[
                "namefrst_related_rows_age_min_5"
            ].iloc[0]
        )
        == 6
    )
    assert (
        len(
            pdf_b.query("histid == 'B04F6A33-9A86-4EAF-884B-0BD6107CCDEB'")[
                "namefrst_related_rows_age_b_min_5"
            ].iloc[0]
        )
        == 6
    )

    assert (
        len(
            pdf_a.query("histid == '8B0A8FA5-A260-4841-95D0-2C45689485C8'")[
                "namefrst_related_rows"
            ].iloc[0]
        )
        == 5
    )
    assert (
        len(
            pdf_a.query("histid == '8B0A8FA5-A260-4841-95D0-2C45689485C8'")[
                "namefrst_related_rows_age_min_5"
            ].iloc[0]
        )
        == 2
    )
    assert (
        len(
            pdf_a.query("histid == '8B0A8FA5-A260-4841-95D0-2C45689485C8'")[
                "namefrst_related_rows_age_b_min_5"
            ].iloc[0]
        )
        == 5
    )

    assert (
        len(
            pdf_b.query("histid == 'F7E0450D-ECCC-4338-92B0-ACB4F9D40D8F'")[
                "namefrst_related_rows"
            ].iloc[0]
        )
        == 7
    )
    assert (
        len(
            pdf_b.query("histid == 'F7E0450D-ECCC-4338-92B0-ACB4F9D40D8F'")[
                "namefrst_related_rows_age_min_5"
            ].iloc[0]
        )
        == 7
    )
    assert (
        len(
            pdf_b.query("histid == 'F7E0450D-ECCC-4338-92B0-ACB4F9D40D8F'")[
                "namefrst_related_rows_age_b_min_5"
            ].iloc[0]
        )
        == 7
    )
