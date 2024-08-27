# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
import pytest

import hlink.linking.core.comparison_feature as comparison_feature_core
import hlink.linking.core.pipeline as pipeline_core
from pyspark.ml import Pipeline
from pyspark.sql import Row


def test_rel_jaro_winkler_comparison(spark, conf, datasource_rel_jw_input):
    """Test the comparison feature data output"""

    table_a, table_b = datasource_rel_jw_input
    features = [
        {
            "alias": "rel_birthyr",
            "column_name": "namefrst_related_rows",
            "birthyr_col": "replaced_birthyr",
            "comparison_type": "rel_jaro_winkler",
            "jw_threshold": 0.9,
            "age_threshold": 5,
            "lower_threshold": 1,
        },
        {
            "alias": "rel_replaced_birthyr",
            "column_name": "namefrst_related_rows_birthyr",
            "comparison_type": "rel_jaro_winkler",
            "jw_threshold": 0.9,
            "age_threshold": 5,
            "lower_threshold": 1,
        },
    ]
    sql_expr_0 = comparison_feature_core.generate_comparison_feature(
        features[0], conf["id_column"], include_as=True
    )
    sql_expr_1 = comparison_feature_core.generate_comparison_feature(
        features[1], conf["id_column"], include_as=True
    )

    table_a.createOrReplaceTempView("table_a")
    table_b.createOrReplaceTempView("table_b")

    df = spark.sql(
        f"select a.id as id_a, b.id as id_b, {sql_expr_0}, {sql_expr_1} from table_a a cross join table_b b"
    ).toPandas()

    assert df.query("id_a == '0' and id_b == '0'")["rel_birthyr"].iloc[0]
    assert not df.query("id_a == '0' and id_b == '1'")["rel_birthyr"].iloc[0]
    assert df.query("id_a == '0' and id_b == '0'")["rel_replaced_birthyr"].iloc[0]
    assert not df.query("id_a == '0' and id_b == '1'")["rel_replaced_birthyr"].iloc[0]


def test_extra_children_comparison(spark, conf, datasource_extra_children_input):
    """Test the comparison feature data output"""

    table_a, table_b = datasource_extra_children_input
    conf["id_column"] = "histid"
    features = [
        {
            "alias": "extra_children",
            "year_b": 1910,
            "column_name": "namefrst_related_rows",
            "relate_col": "relate",
            "histid_col": "histid",
            "birthyr_col": "birthyr",
            "name_col": "namefrst",
            "comparison_type": "extra_children",
            "jw_threshold": 0.8,
            "age_threshold": 2,
        }
    ]
    sql_expr_0 = comparison_feature_core.generate_comparison_feature(
        features[0], conf["id_column"], include_as=True
    )

    table_a.createOrReplaceTempView("table_a")
    table_b.createOrReplaceTempView("table_b")

    df = spark.sql(
        f"select a.histid as histid_a, b.histid as histid_b, {sql_expr_0} from table_a a cross join table_b b"
    ).toPandas()

    assert df.query("histid_a == 0 and histid_b == 8")["extra_children"].iloc[0] == 0
    assert df.query("histid_a == 0 and histid_b == 11")["extra_children"].iloc[0] == 2
    assert df.query("histid_a == 0 and histid_b == 15")["extra_children"].iloc[0] == 0

    assert df.query("histid_a == 4 and histid_b == 8")["extra_children"].iloc[0] == 0
    assert df.query("histid_a == 4 and histid_b == 11")["extra_children"].iloc[0] == 2
    assert df.query("histid_a == 4 and histid_b == 15")["extra_children"].iloc[0] == 0

    assert df.query("histid_a == 7 and histid_b == 8")["extra_children"].iloc[0] == 0
    assert df.query("histid_a == 7 and histid_b == 11")["extra_children"].iloc[0] == 0
    assert df.query("histid_a == 7 and histid_b == 15")["extra_children"].iloc[0] == 0

    assert df.query("histid_a == 17 and histid_b == 8")["extra_children"].iloc[0] == 1
    assert df.query("histid_a == 17 and histid_b == 11")["extra_children"].iloc[0] == 2
    assert df.query("histid_a == 17 and histid_b == 15")["extra_children"].iloc[0] == 0


def test_comparison_and_mi(spark, conf, datasource_mi_comparison):
    """Test the comparison feature data output"""

    table_a, table_b = datasource_mi_comparison
    features = [
        {
            "alias": "mi_old",
            "column_name": "namefrst_mid_init",
            "comparison_type": "and",
            "comp_a": {"column_name": "namefrst_mid_init", "comparison_type": "equals"},
            "comp_b": {
                "column_name": "namefrst_mid_init",
                "comparison_type": "neither_are_null",
            },
        },
        {
            "alias": "mi",
            "column_name": "namefrst_mid_init",
            "comparison_type": "present_and_matching_categorical",
        },
    ]

    sql_expr_0 = comparison_feature_core.generate_comparison_feature(
        features[0], conf["id_column"], include_as=True
    )

    sql_expr_1 = comparison_feature_core.generate_comparison_feature(
        features[1], conf["id_column"], include_as=True
    )

    table_a.createOrReplaceTempView("table_a")
    table_b.createOrReplaceTempView("table_b")

    df = spark.sql(
        f"select a.id as id_a, b.id as id_b, a.namefrst_mid_init as namefrst_mid_init_a, b.namefrst_mid_init as namefrst_mid_init_b, {sql_expr_0}, {sql_expr_1} from table_a a cross join table_b b"
    ).toPandas()

    assert df.query("id_a == 10 and id_b == 40")["mi_old"].iloc[0]
    assert not df.query("id_a == 20 and id_b == 40")["mi_old"].iloc[0]
    assert not df.query("id_a == 20 and id_b == 50")["mi_old"].iloc[0]
    assert not df.query("id_a == 20 and id_b == 60")["mi_old"].iloc[0]
    assert not df.query("id_a == 30 and id_b == 50")["mi_old"].iloc[0]
    assert not df.query("id_a == 30 and id_b == 60")["mi_old"].iloc[0]

    assert df.query("id_a == 10 and id_b == 40")["mi"].iloc[0] == 0
    assert df.query("id_a == 10 and id_b == 50")["mi"].iloc[0] == 2
    assert df.query("id_a == 10 and id_b == 60")["mi"].iloc[0] == 2
    assert df.query("id_a == 20 and id_b == 40")["mi"].iloc[0] == 1
    assert df.query("id_a == 20 and id_b == 50")["mi"].iloc[0] == 2
    assert df.query("id_a == 20 and id_b == 60")["mi"].iloc[0] == 2
    assert df.query("id_a == 30 and id_b == 40")["mi"].iloc[0] == 2
    assert df.query("id_a == 30 and id_b == 50")["mi"].iloc[0] == 2
    assert df.query("id_a == 30 and id_b == 60")["mi"].iloc[0] == 2


def test_immyr_diff_w_imm_caution(spark, conf):
    """Test the comparison feature data output"""

    data_a = [
        (0, 5, 1900),
        (1, 5, 1900),
        (2, 5, 1900),
        (3, 5, 1900),
        (4, 5, 1900),
        (5, 5, 1900),
        (6, 5, 1900),
        (7, 1, 0000),
        (8, 1, 0000),
    ]
    data_b = [
        (0, 5, 1900),
        (1, 5, 1901),
        (2, 2, 1905),
        (3, 3, 1906),
        (4, 5, 1910),
        (5, 5, 1911),
        (6, 5, 1912),
        (7, 1, 0000),
        (8, 0, 0000),
    ]

    table_a = spark.createDataFrame(data_a, ["id", "nativity", "yrimmig"])
    table_b = spark.createDataFrame(data_b, ["id", "nativity", "yrimmig"])

    features = [
        {
            "alias": "imm",
            "column_name": "nativity",
            "comparison_type": "fetch_a",
            "threshold": 5,
            "categorical": True,
        },
        {
            "alias": "immyear_diff",
            "column_name": "yrimmig",
            "comparison_type": "abs_diff",
            "look_at_addl_var": True,
            "addl_var": "nativity",
            "datasource": "a",
            "check_val_expr": "= 5",
            "else_val": -1,
        },
    ]

    expr0 = comparison_feature_core.generate_comparison_feature(
        features[0], conf["id_column"], include_as=True
    )
    expr1 = comparison_feature_core.generate_comparison_feature(
        features[1], conf["id_column"], include_as=True
    )

    table_a.createOrReplaceTempView("table_a")
    table_b.createOrReplaceTempView("table_b")

    df0 = spark.sql(
        f"select a.id as id_a, b.id as id_b, {expr0}, {expr1} from table_a a join table_b b on a.id == b.id"
    )
    df = df0.toPandas()

    assert df.query("id_a == 0")["imm"].iloc[0]
    assert df.query("id_a == 1")["imm"].iloc[0]
    assert df.query("id_a == 2")["imm"].iloc[0]
    assert df.query("id_a == 3")["imm"].iloc[0]
    assert not df.query("id_a == 7")["imm"].iloc[0]
    assert not df.query("id_a == 8")["imm"].iloc[0]

    assert df.query("id_a == 0")["immyear_diff"].iloc[0] == 0
    assert df.query("id_a == 4")["immyear_diff"].iloc[0] == 10
    assert df.query("id_a == 7")["immyear_diff"].iloc[0] == -1
    assert df.query("id_a == 8")["immyear_diff"].iloc[0] == -1

    df0.createOrReplaceTempView("training_features")

    conf["pipeline_features"] = [
        {
            "input_column": "immyear_diff",
            "output_column": "immyear_caution",
            "transformer_type": "bucketizer",
            "categorical": True,
            "splits": [-1, 0, 6, 11, 9999],
        }
    ]
    conf["training"] = {
        "dependent_var": "match",
        "independent_vars": ["immyear_diff", "immyear_caution"],
    }
    conf["comparison_features"] = []

    ind_vars = conf["training"]["independent_vars"]
    tf = spark.table("training_features")
    pipeline_stages = pipeline_core.generate_pipeline_stages(
        conf, ind_vars, tf, "training"
    )
    prep_pipeline = Pipeline(stages=pipeline_stages)
    prep_model = prep_pipeline.fit(tf)
    prepped_data = prep_model.transform(tf)
    prepped_data = prepped_data.toPandas()

    assert prepped_data.query("id_a == 0")["immyear_caution"].iloc[0] == 1
    assert prepped_data.query("id_a == 1")["immyear_caution"].iloc[0] == 1
    assert prepped_data.query("id_a == 2")["immyear_caution"].iloc[0] == 1
    assert prepped_data.query("id_a == 3")["immyear_caution"].iloc[0] == 2
    assert prepped_data.query("id_a == 4")["immyear_caution"].iloc[0] == 2
    assert prepped_data.query("id_a == 5")["immyear_caution"].iloc[0] == 3
    assert prepped_data.query("id_a == 6")["immyear_caution"].iloc[0] == 3
    assert prepped_data.query("id_a == 7")["immyear_caution"].iloc[0] == 0
    assert prepped_data.query("id_a == 8")["immyear_caution"].iloc[0] == 0


def test_multi_jaro_winkler_search_comparison_feature(
    spark, datasource_multi_jaro_winkler_search_input
):
    feature = {
        "alias": "test_jw_search",
        "comparison_type": "multi_jaro_winkler_search",
        "num_cols": 2,
        "jw_col_template": "s{n}_namefrst",
        "jw_threshold": 0.7,
        "equal_and_not_null_templates": ["s{n}_bpl", "s{n}_sex"],
    }
    table_a, table_b = datasource_multi_jaro_winkler_search_input

    sql_expr = comparison_feature_core.generate_comparison_feature(
        feature, "id", include_as=True
    )

    table_a.createOrReplaceTempView("table_a")
    table_b.createOrReplaceTempView("table_b")

    df = spark.sql(
        f"SELECT a.id AS id_a, b.id AS id_b, {sql_expr} FROM table_a a CROSS JOIN table_b b"
    )

    assert df.filter((df.id_a == 1) & (df.id_b == 101)).collect()[0].test_jw_search
    assert not df.filter((df.id_a == 1) & (df.id_b == 102)).collect()[0].test_jw_search
    assert not df.filter((df.id_a == 1) & (df.id_b == 103)).collect()[0].test_jw_search
    assert not df.filter((df.id_a == 1) & (df.id_b == 104)).collect()[0].test_jw_search
    assert not df.filter((df.id_a == 2) & (df.id_b == 101)).collect()[0].test_jw_search
    assert not df.filter((df.id_a == 2) & (df.id_b == 102)).collect()[0].test_jw_search
    assert not df.filter((df.id_a == 2) & (df.id_b == 103)).collect()[0].test_jw_search
    assert not df.filter((df.id_a == 2) & (df.id_b == 104)).collect()[0].test_jw_search


def test_multi_jaro_winkler_search_equal_templates_optional(
    spark, datasource_multi_jaro_winkler_search_input
):
    """
    The equal_and_not_null_templates attribute is optional and defaults to the empty list [].
    """
    feature = {
        "alias": "test_jw_search",
        "comparison_type": "multi_jaro_winkler_search",
        "num_cols": 2,
        "jw_col_template": "s{n}_namefrst",
        "jw_threshold": 0.9,
    }

    table_a, table_b = datasource_multi_jaro_winkler_search_input

    sql_expr = comparison_feature_core.generate_comparison_feature(
        feature, "histid", include_as=True
    )

    table_a.createOrReplaceTempView("table_a")
    table_b.createOrReplaceTempView("table_b")

    # Just make sure that the query parses and runs
    spark.sql(f"SELECT {sql_expr} FROM table_a a CROSS JOIN table_b b")


def test_multi_jaro_winkler_search_column_templating():
    """Test the comparison feature when...
    - there are more than 10 columns
    - there are several different templates to fill in
    - some of the templates have multiple {n}s (or 0 {n}s!) and are strange
    """
    feature = {
        "alias": "test_jw_search",
        "comparison_type": "multi_jaro_winkler_search",
        "num_cols": 11,
        "jw_col_template": "COLUMN_NAME{n}",
        "jw_threshold": 0.1,
        "equal_and_not_null_templates": [
            "TEST{n}_TEST{n}",
            "{n}{n}{n}__COLUMN",
            "static_column",
        ],
    }

    sql_expr = comparison_feature_core.generate_comparison_feature(
        feature, "id", include_as=True
    )

    # Check for various column names in the sql_expr, which is just a str
    assert "COLUMN_NAME0" not in sql_expr
    assert "COLUMN_NAME1" in sql_expr
    assert "COLUMN_NAME9" in sql_expr
    assert "COLUMN_NAME10" in sql_expr
    assert "COLUMN_NAME11" in sql_expr
    assert "COLUMN_NAME12" not in sql_expr

    assert "TEST0_TEST0" not in sql_expr
    assert "TEST1_TEST1" in sql_expr
    assert "TEST3_TEST3" in sql_expr
    assert "TEST3_TEST4" not in sql_expr
    assert "TEST11_TEST11" in sql_expr
    assert "TEST12_TEST12" not in sql_expr

    assert "000__COLUMN" not in sql_expr
    assert "111__COLUMN" in sql_expr
    assert "111111__COLUMN" in sql_expr
    assert "121212__COLUMN" not in sql_expr

    assert "static_column" in sql_expr
    assert "static_column1" not in sql_expr
    assert "static_colum1" not in sql_expr


def test_b_minus_a_comparison(spark) -> None:
    comparison_feature = {
        "alias": "agediff",
        "column_name": "age",
        "comparison_type": "b_minus_a",
    }

    df_a = spark.createDataFrame([[0, 15], [1, 77]], "id:integer, age:integer")
    df_b = spark.createDataFrame([[100, 15], [101, 70]], "id:integer, age:integer")
    df_a.write.saveAsTable("table_a")
    df_b.write.saveAsTable("table_b")

    sql_expr = comparison_feature_core.generate_comparison_feature(
        comparison_feature, "id", include_as=True
    )

    result = (
        spark.sql(
            f"SELECT a.id AS id_a, b.id AS id_b, {sql_expr} FROM table_a a CROSS JOIN table_b b"
        )
        .sort("id_a", "id_b")
        .collect()
    )

    assert result == [
        Row(id_a=0, id_b=100, agediff=0),
        Row(id_a=0, id_b=101, agediff=55),
        Row(id_a=1, id_b=100, agediff=-62),
        Row(id_a=1, id_b=101, agediff=-7),
    ]


def test_b_minus_a_comparison_with_not_equals(spark) -> None:
    comparison_feature = {
        "alias": "agediff",
        "column_name": "age",
        "comparison_type": "b_minus_a",
        "not_equals": 99,
    }
    df_a = spark.createDataFrame([[0, 15], [1, 77], [2, 99]], "id:integer, age:integer")
    df_b = spark.createDataFrame(
        [[100, 15], [101, 70], [102, 99]], "id:integer, age:integer"
    )

    df_a.write.saveAsTable("table_a")
    df_b.write.saveAsTable("table_b")

    sql_expr = comparison_feature_core.generate_comparison_feature(
        comparison_feature, "id", include_as=True
    )

    result = (
        spark.sql(
            f"SELECT a.id AS id_a, b.id AS id_b, {sql_expr} FROM table_a a CROSS JOIN table_b b"
        )
        .sort("id_a", "id_b")
        .collect()
    )

    assert result == [
        Row(id_a=0, id_b=100, agediff=0),
        Row(id_a=0, id_b=101, agediff=55),
        Row(id_a=0, id_b=102, agediff=-1),
        Row(id_a=1, id_b=100, agediff=-62),
        Row(id_a=1, id_b=101, agediff=-7),
        Row(id_a=1, id_b=102, agediff=-1),
        Row(id_a=2, id_b=100, agediff=-1),
        Row(id_a=2, id_b=101, agediff=-1),
        Row(id_a=2, id_b=102, agediff=-1),
    ]


def test_generate_comparison_feature_error_on_unknown_comparison_type() -> None:
    comparison_feature = {"comparison_type": "not_supported"}
    with pytest.raises(ValueError, match="No comparison type"):
        comparison_feature_core.generate_comparison_feature(comparison_feature, "id")
