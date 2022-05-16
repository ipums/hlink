# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pytest
import pandas as pd
from hlink.linking.matching.link_step_score import LinkStepScore


@pytest.mark.quickcheck
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


# TODO: test_step_2_length_b

# TODO: test_step_2_has_matching_element

# TODO: test_step_2_error_no_comp_type
