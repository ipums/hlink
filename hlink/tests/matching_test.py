# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pandas as pd
from hlink.linking.matching.link_step_score import LinkStepScore


def test_removal_of_duplicate_histid_b(
    spark, matching, matching_conf, scored_matches_test_data
):
    """Test all hh matching and training steps to ensure they work as a workflow"""
    path_pms = scored_matches_test_data
    matching.spark.read.csv(path_pms, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("scored_potential_matches")

    LinkStepScore(matching)._save_predicted_matches(
        matching_conf, "histid_a", "histid_b"
    )

    pred_matches = matching.spark.table("predicted_matches").toPandas()

    assert pred_matches.shape == (1, 3)
    assert (
        pred_matches.query("histid_a == 'A004' and histid_b == 'B003'")[
            "prediction"
        ].iloc[0]
        == 1
    )


def test_step_2_any_equals(spark, matching_household_conf, matching, preprocessing):

    matching_household_conf["column_mappings"].append(
        {
            "column_name": "namefrst_std",
            "alias": "namefrst_split",
            "transforms": [{"type": "split"}],
        }
    )

    matching_household_conf["column_mappings"].append(
        {
            "column_name": "namefrst_split",
            "alias": "namefrst_mid_init",
            "transforms": [
                {"type": "array_index", "value": 1},
                {"type": "substring", "values": [0, 1]},
            ],
        }
    )

    matching_household_conf["column_mappings"].append(
        {
            "column_name": "namefrst_split",
            "alias": "namefrst_unstd",
            "transforms": [{"type": "array_index", "value": 0}],
        }
    )

    matching_household_conf["comparison_features"] = [
        {
            "alias": "mid_init_match",
            "column_names": ["namefrst_mid_init", "namefrst_unstd"],
            "comparison_type": "any_equals",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
    ]

    matching_household_conf["training"]["dependent_var"] = "match"
    matching_household_conf["training"]["independent_vars"] = [
        "neighbor_namelast_jw_rate",
        "neighbor_namelast_jw_rate_threshold",
        "namelast_jw",
    ]
    preprocessing.run_step(0)
    preprocessing.run_step(1)
    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(matching_household_conf)

    # Create pandas DFs of the step_2 potential matches table
    potential_matches_df = spark.table("potential_matches_prepped").toPandas()

    # Make assertions on the data
    assert len(potential_matches_df.id_a) == 37
    assert len(potential_matches_df.mid_init_match) == 37
    assert len(potential_matches_df.namefrst_mid_init_a == 37)
    assert (
        potential_matches_df.query(
            "id_a == '50b33ef6-259d-43af-8cdc-56a61f881169 ' and id_b == '50b33ef6-259d-43af-8cdc-56a61f881169 '"
        )["mid_init_match"].iloc[0]
        == 1
    )
    assert pd.isna(
        potential_matches_df.query(
            "id_a == '7fb55d25-2a7d-486d-9efa-27b9d7e60c24 ' and id_b == '7fb55d25-2a7d-486d-9efa-27b9d7e60c24 '"
        )["mid_init_match"].iloc[0]
    )
    assert (
        potential_matches_df.query(
            "id_a == '7fb55d25-2a7d-486d-9efa-27b9d7e60c24 ' and id_b == '7fb55d25-2a7d-486d-9efa-27b9d7e60c24 '"
        )["namefrst_mid_init_a"].iloc[0]
        is None
    )
    assert (
        potential_matches_df.query(
            "id_a == '7fb55d25-2a7d-486d-9efa-27b9d7e60c24 ' and id_b == '7fb55d25-2a7d-486d-9efa-27b9d7e60c24 '"
        )["namefrst_unstd_a"].iloc[0]
        == "phineas"
    )


def test_step_2_sum(spark, matching_household_conf, matching, preprocessing):

    matching_household_conf["feature_selections"] = [
        {
            "output_col": "namelast_popularity",
            "input_cols": ["sex", "bpl", "namelast_clean"],
            "transform": "popularity",
        }
    ]

    matching_household_conf["comparison_features"] = [
        {
            "alias": "namelast_popularity_sum",
            "column_name": "namelast_popularity",
            "comparison_type": "sum",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
    ]

    matching_household_conf["training"]["dependent_var"] = "match"
    matching_household_conf["training"]["independent_vars"] = [
        "neighbor_namelast_jw_rate",
        "neighbor_namelast_jw_rate_threshold",
        "namelast_jw",
    ]
    preprocessing.run_step(0)
    preprocessing.run_step(1)
    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(matching_household_conf)

    # Create pandas DFs of the step_2 potential matches table
    potential_matches_df = spark.table("potential_matches_prepped").toPandas()
    assert len(potential_matches_df.namelast_popularity_a) == 37
    assert len(potential_matches_df.namelast_popularity_b) == 37
    assert len(potential_matches_df.namelast_popularity_sum) == 37
    assert (
        potential_matches_df.query(
            "id_a == 'ae7261c3-7d71-4ea1-997f-5d1a68c18777 ' and id_b == 'ad6442b5-42bc-4c2e-a517-5a951d989a92 '"
        )["namelast_popularity_a"].iloc[0]
        == 3
    )
    assert (
        potential_matches_df.query(
            "id_a == 'ae7261c3-7d71-4ea1-997f-5d1a68c18777 ' and id_b == 'ad6442b5-42bc-4c2e-a517-5a951d989a92 '"
        )["namelast_popularity_b"].iloc[0]
        == 2
    )
    assert (
        potential_matches_df.query(
            "id_a == 'ae7261c3-7d71-4ea1-997f-5d1a68c18777 ' and id_b == 'ad6442b5-42bc-4c2e-a517-5a951d989a92 '"
        )["namelast_popularity_sum"].iloc[0]
        == 5
    )


#
# TODO: fix hh compare rate java function
# def test_step_2_hh_compare_rate(spark, matching_household_conf, matching, preprocessing):
# matching_household_conf['feature_selections'] = [
#  {
#   "output_col": "namefrst_related_rows",
#  "input_cols": ["namefrst_std", "bpl", "sex"],
# "transform": "related_individual_rows",
# "family_id": "serialp",
# "relate_col": "relate",
# "top_code": 10,
# "bottom_code": 3
# }
# ]
#
#  matching_household_conf['comparison_features'] = [
#   {
#    "alias": "namelast_jw",
#   "column_name": "namelast_clean",
#  "comparison_type": "jaro_winkler"
# },
# {
# "alias": "related_match_rate",
# "column_name": "namefrst_related_rows",
# "comparison_type": "hh_compare_rate"
# }
# ]
#
# preprocessing.step_0_register_raw_dfs()
# preprocessing.step_1_prep_dataframe()
# matching.step_0_explode()
# matching.step_1_match()
#
# Create pandas DFs of the step_2 potential matches table
# potential_matches_df = spark.table("potential_matches").toPandas()
# assert (len(potential_matches_df.namelast_popularity_a) == 41)
