# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.sql import Row
import pandas as pd
from hlink.linking.matching.link_step_match import extract_or_groups_from_blocking
from hlink.linking.matching.link_step_score import LinkStepScore


def test_steps_1_2_matching(
    spark, blocking_explode_conf, matching_test_input, matching, main
):
    """Test explode step with blocking columns"""
    table_a, table_b = matching_test_input
    table_a.createOrReplaceTempView("prepped_df_a")
    table_b.createOrReplaceTempView("prepped_df_b")

    matching.run_step(0)

    expl_a = spark.table("exploded_df_a").toPandas()
    expl_b = spark.table("exploded_df_b").toPandas()

    assert all(
        elem in list(expl_a.columns)
        for elem in ["namefrst", "namelast", "sex", "birthyr_3"]
    )
    assert all(
        elem in list(expl_b.columns)
        for elem in ["namefrst", "namelast", "sex", "birthyr_3"]
    )

    matching.run_step(1)

    potential_matches_df = spark.table("potential_matches").toPandas()

    assert all(
        elem not in list(potential_matches_df.columns) for elem in ["birthyr_3", "ssex"]
    )

    blocking_explode_conf["streamline_potential_match_generation"] = True
    main.do_drop("potential_matches")
    matching.run_step(1)

    pm_small = spark.table("potential_matches").toPandas()

    assert "ssex" not in list(pm_small.columns)
    assert all(
        elem in list(pm_small.columns)
        for elem in ["id_a", "id_b", "namefrst_jw", "namelast_jw"]
    )

    LinkStepScore(matching)._create_features(matching.link_run.config)
    pmp = spark.table("potential_matches_prepped").toPandas()

    assert all(
        elem in list(pmp.columns)
        for elem in ["id_a", "id_b", "namefrst_jw", "namelast_jw", "ssex"]
    )


def test_blocking_multi_layer_comparison(
    matching_conf_namefrst_std_and_unstd, spark, preprocessing, matching
):
    """Test a blocking criteria comparison which contains an 'and' clause and a nested 'or' clause"""

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    matching.run_step(0)
    matching.run_step(1)

    pms = matching.spark.table("potential_matches").toPandas()

    assert len(pms["histid_a"]) == 4
    assert "ginny" not in list(pms["namefrst_unstd_a"])
    assert "jupiter" not in list(pms["namelast_clean_a"])
    for index, row in pms.iterrows():
        assert (row["namefrst_unstd_jw"] > 0.7) or (row["namefrst_std_jw"] > 0.7)
        assert row["namelast_jw"] > 0.7

    matching_conf_namefrst_std_and_unstd["comparisons"] = {
        "operator": "AND",
        "comp_a": {
            "operator": "OR",
            "comp_a": {
                "feature_name": "namefrst_unstd_jw",
                "threshold": 0.0,
                "comparison_type": "threshold",
            },
            "comp_b": {
                "feature_name": "namefrst_std_jw",
                "threshold": 0.0,
                "comparison_type": "threshold",
            },
        },
        "comp_b": {
            "feature_name": "namelast_jw",
            "threshold": 0.0,
            "comparison_type": "threshold",
        },
    }

    spark.sql("drop table potential_matches")
    matching.run_step(1)

    pm_no_clause = matching.spark.table("potential_matches").toPandas()

    assert len(pm_no_clause["histid_a"]) == 6
    assert "ginny" in list(pm_no_clause["namefrst_unstd_a"])
    assert "jupiter" in list(pm_no_clause["namelast_clean_a"])

    only_test_df = pd.merge(
        pm_no_clause, pms, on=["histid_a", "histid_b"], how="outer", indicator=True
    )
    only_test_df = only_test_df[only_test_df["_merge"] == "left_only"]
    assert len(only_test_df["histid_a"]) == 2
    for index, row in only_test_df.iterrows():
        assert (
            (row["namefrst_unstd_jw_x"] < 0.7) or (row["namefrst_std_jw_x"] < 0.7)
        ) or (row["namelast_jw_x"] < 0.7)


def test_blocking_multiple_exploded_columns(
    spark, blocking_explode_conf, matching_test_input, matching
):
    """
    Matching supports multiple exploded blocking columns. Each column is
    exploded independently. See GitHub issue #142.
    """
    table_a, table_b = matching_test_input
    table_a.createOrReplaceTempView("prepped_df_a")
    table_b.createOrReplaceTempView("prepped_df_b")

    blocking_explode_conf["blocking"] = [
        {
            "column_name": "birthyr_3",
            "dataset": "a",
            "derived_from": "birthyr",
            "expand_length": 3,
            "explode": True,
        },
        {
            "column_name": "birthyr_4",
            "dataset": "a",
            "derived_from": "birthyr",
            "expand_length": 4,
            "explode": True,
        },
        {"column_name": "sex"},
    ]

    matching.run_step(0)

    exploded_a = spark.table("exploded_df_a").toPandas()
    exploded_b = spark.table("exploded_df_b").toPandas()

    input_size_a = spark.table("prepped_df_a").count()
    input_size_b = spark.table("prepped_df_b").count()
    output_size_a = len(exploded_a)
    output_size_b = len(exploded_b)

    assert "sex" in exploded_a.columns
    assert "birthyr_3" in exploded_a.columns
    assert "birthyr_4" in exploded_a.columns
    assert "sex" in exploded_b.columns
    assert "birthyr_3" in exploded_b.columns
    assert "birthyr_4" in exploded_b.columns

    # birthyr_3 multiplies the number of columns by 2 * 3 + 1 = 7
    # birthyr_4 multiplies the number of columns by 2 * 4 + 1 = 9
    assert input_size_a * 63 == output_size_a
    # Both columns are only exploded in dataset A
    assert input_size_b == output_size_b


def test_blocking_or_groups(
    spark, blocking_or_groups_conf, matching_or_groups_test_input, matching
):
    """Test the blocking or_group functionality. This feature supports
    combining some or all blocking conditions with OR instead of AND."""
    table_a, table_b = matching_or_groups_test_input
    table_a.createOrReplaceTempView("prepped_df_a")
    table_b.createOrReplaceTempView("prepped_df_b")

    matching.run_step(0)
    matching.run_step(1)

    potential_matches = matching.spark.table("potential_matches")

    results = potential_matches.select("id_a", "id_b").collect()

    assert set(results) == {
        Row(
            id_a="ad6442b5-42bc-4c2e-a517-5a951d989a92 ",
            id_b="ad6442b5-42bc-4c2e-a517-5a951d989a92 ",
        ),
        Row(
            id_a="a499b0dc-7ac0-4d61-b493-91a3036c712e ",
            id_b="a499b0dc-7ac0-4d61-b493-91a3036c712e ",
        ),
        Row(
            id_a="7fb55d25-2a7d-486d-9efa-27b9d7e60c24 ",
            id_b="7fb55d25-2a7d-486d-9efa-27b9d7e60c24 ",
        ),
        Row(
            id_a="a0f33b36-cef7-4949-a031-22b90f1055d4 ",
            id_b="a0f33b36-cef7-4949-a031-22b90f1055d4 ",
        ),
    }


# TODO: test_step_2_length_b

# TODO: test_step_2_has_matching_element

# TODO: test_step_2_error_no_comp_type


def test_extract_or_groups_from_blocking_empty() -> None:
    blocking = []
    or_groups = extract_or_groups_from_blocking(blocking)
    assert or_groups == []


def test_extract_or_groups_from_blocking_no_explicit_or_groups() -> None:
    blocking = [
        {
            "column_name": "AGE_3",
            "explode": True,
            "expand_length": 3,
            "derived_from": "AGE",
            "dataset": "a",
        },
        {"column_name": "BPL"},
        {"column_name": "SEX"},
    ]
    or_groups = extract_or_groups_from_blocking(blocking)
    assert or_groups == [["AGE_3"], ["BPL"], ["SEX"]]


def test_extract_or_groups_from_blocking_explicit_or_groups() -> None:
    blocking = [
        {"column_name": "BPL1", "or_group": "BPL"},
        {
            "column_name": "AGE_3",
            "explode": True,
            "expand_length": 3,
            "derived_from": "AGE",
            "dataset": "a",
            "or_group": "AGE",
        },
        {"column_name": "BPL2", "or_group": "BPL"},
        {"column_name": "BPL3", "or_group": "BPL"},
    ]
    or_groups = extract_or_groups_from_blocking(blocking)
    assert or_groups == [["BPL1", "BPL2", "BPL3"], ["AGE_3"]]


def test_extract_or_groups_from_blocking_or_groups_with_explode() -> None:
    blocking = [
        {
            "column_name": "AGE1_3",
            "explode": True,
            "expand_length": 3,
            "derived_from": "AGE1",
            "dataset": "a",
            "or_group": "AGE",
        },
        {
            "column_name": "AGE2_3",
            "explode": True,
            "expand_length": 3,
            "derived_from": "AGE2",
            "dataset": "a",
            "or_group": "AGE",
        },
        {
            "column_name": "BPL",
        },
    ]
    or_groups = extract_or_groups_from_blocking(blocking)
    assert or_groups == [["AGE1_3", "AGE2_3"], ["BPL"]]
