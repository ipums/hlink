# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pandas as pd
from hlink.linking.matching.link_step_score import LinkStepScore


# TODO: add documentation
def test_step_2_equals_and_equals_as_int(
    spark, matching_household_conf, matching, preprocessing
):
    matching_household_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "namelast_equal",
            "column_name": "namelast_clean",
            "comparison_type": "equals",
        },
        {
            "alias": "namelast_equal_as_int",
            "column_name": "namelast_clean",
            "comparison_type": "equals_as_int",
        },
    ]
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

    # Create pandas DFs of the step_1 potential matches table
    potential_matches_df = spark.table("potential_matches_prepped").toPandas()

    # Make assertions on the data
    assert len(potential_matches_df.id_a) == 37
    assert len(potential_matches_df.namelast_equal) == 37
    assert len(potential_matches_df.namelast_equal_as_int == 37)
    assert not potential_matches_df.query(
        "id_a == '92277f0b-1476-41f5-9dc8-bf83672616d0' and id_b == '9e807937-de09-414c-bfb2-ac821e112929 '"
    )["namelast_equal"].iloc[0]
    assert (
        potential_matches_df.query(
            "id_a == '92277f0b-1476-41f5-9dc8-bf83672616d0' and id_b == '9e807937-de09-414c-bfb2-ac821e112929 '"
        )["namelast_equal_as_int"].iloc[0]
        == 0
    )
    assert potential_matches_df.query(
        "id_a == '49e53dbc-fe8e-4e55-8cb9-a1d93c284d98 ' and id_b == '3575c9ba-1527-4ca2-aff0-d7c2d1efb421 '"
    )["namelast_equal"].iloc[0]
    assert (
        potential_matches_df.query(
            "id_a == '49e53dbc-fe8e-4e55-8cb9-a1d93c284d98 ' and id_b == '3575c9ba-1527-4ca2-aff0-d7c2d1efb421 '"
        )["namelast_equal_as_int"].iloc[0]
        == 1
    )


# TODO: add documentation
def test_step_2_all_equals(spark, matching_household_conf, matching, preprocessing):
    matching_household_conf["comparison_features"] = [
        {
            "alias": "exact_all",
            "column_names": ["namefrst_std", "namelast_clean", "age"],
            "comparison_type": "all_equals",
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
    assert len(potential_matches_df.exact_all) == 37
    assert (
        potential_matches_df.query(
            "id_a == 'a499b0dc-7ac0-4d61-b493-91a3036c712e ' and id_b == '426f2cbe-32e1-45eb-9f86-89a2b9116b7e '"
        )["exact_all"].iloc[0]
        == 0
    )
    assert (
        potential_matches_df.query(
            "id_a == 'bcc0988e-2397-4f1b-8e76-4bfe1b05dbc6 ' and id_b == 'bcc0988e-2397-4f1b-8e76-4bfe1b05dbc6 '"
        )["exact_all"].iloc[0]
        == 1
    )


def test_step_2_fetch_either_length(
    spark, preprocessing, matching, matching_conf_nativity
):
    """Test nativity, imm, sgen (second generation immigrant) code transforms as well as nested comps and fetch_a"""
    matching_conf_nativity["id_column"] = "histid"
    matching_conf_nativity["column_mappings"] = [
        {"column_name": "pair_no"},
        {"column_name": "nativity"},
        {"column_name": "county"},
        {"column_name": "state"},
        {"column_name": "street"},
    ]
    matching_conf_nativity["blocking"] = [{"column_name": "pair_no"}]
    matching_conf_nativity["comparisons"] = {}
    matching_conf_nativity["comparison_features"] = [
        {
            "alias": "imm",
            "comparison_type": "fetch_b",
            "column_name": "nativity",
            "threshold": 5,
            "categorical": True,
        },
        {
            "alias": "either_1",
            "column_name": "nativity",
            "comparison_type": "either_are_1",
            "categorical": True,
        },
        {
            "alias": "either_0",
            "column_name": "nativity",
            "comparison_type": "either_are_0",
            "categorical": True,
        },
    ]

    matching_conf_nativity["training"]["independent_vars"] = [
        "imm",
        "sgen",
        "street_jw",
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(matching_conf_nativity)

    # Create pandas DFs of the step_1 potential matches table
    matches = spark.table("potential_matches_prepped").toPandas()

    assert matches.query("pair_no_a == 5")["either_1"].iloc[0]
    assert not matches.query("pair_no_a == 5")["either_0"].iloc[0]
    assert matches.query("pair_no_a == 2")["imm"].iloc[0]
    assert matches.query("pair_no_a == 4")["imm"].iloc[0]


def test_step_2_nativity(spark, preprocessing, matching, matching_conf_nativity):
    """Test nativity, imm, sgen (second generation immigrant) code transforms as well as nested comps and fetch_a"""
    matching_conf_nativity["id_column"] = "histid"
    matching_conf_nativity["column_mappings"] = [
        {"column_name": "pair_no"},
        {"column_name": "nativity"},
        {"column_name": "county"},
        {"column_name": "state"},
        {"column_name": "street"},
    ]
    matching_conf_nativity["blocking"] = [{"column_name": "pair_no"}]
    matching_conf_nativity["comparisons"] = {}
    matching_conf_nativity["comparison_features"] = [
        {
            "alias": "imm",
            "comparison_type": "fetch_a",
            "column_name": "nativity",
            "threshold": 5,
            "categorical": True,
        },
        {
            "alias": "sgen",
            "column_name": "nativity",
            "comparison_type": "second_gen_imm",
            "categorical": True,
        },
        {
            "alias": "street_jw",
            "column_names": ["street", "county", "state"],
            "comparison_type": "times",
            "comp_a": {
                "column_name": "street",
                "comparison_type": "jaro_winkler",
                "lower_threshold": 0.9,
            },
            "comp_b": {
                "comparison_type": "and",
                "comp_a": {"column_name": "county", "comparison_type": "equals"},
                "comp_b": {"column_name": "state", "comparison_type": "equals"},
            },
        },
    ]

    matching_conf_nativity["training"]["independent_vars"] = [
        "imm",
        "sgen",
        "street_jw",
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(matching_conf_nativity)

    matches = spark.table("potential_matches_prepped").toPandas()

    assert matches.query("pair_no_a == 4")["imm"].iloc[0]
    assert matches.query("pair_no_a == 2")["sgen"].iloc[0]
    assert not matches.query("pair_no_a == 5")["sgen"].iloc[0]
    assert matches.query("pair_no_a == 4")["street_jw"].iloc[0] == 1
    assert matches.query("pair_no_a == 1")["street_jw"].iloc[0] == 0


def test_step_2_JW_only(spark, matching_conf, matching):
    """Test matching step 2 to ensure that comparison features are generated (can a regular comparison (as represented by J/W) still run if there's NOT a distance lookup feature)"""

    matching_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        }
    ]

    matching.run_step(0)
    matching.run_step(1)

    # Create pandas DFs of the step_2 potential matches table
    potential_matches_df = spark.table("potential_matches").toPandas()

    # Make assertions on the data
    assert len(potential_matches_df.id_a) == 5
    assert len(potential_matches_df.namelast_jw) == 5
    assert (
        potential_matches_df.query("id_a == 20 and id_b == 30")["namelast_jw"].iloc[0]
        == 1
    )
    assert (
        potential_matches_df.query("id_a == 10 and id_b == 10")["namelast_jw"].iloc[0]
        > 0.87
    )


def test_step_2_JW_street(spark, matching_conf, matching):
    """Test creation of comparison feature with an IF requirement (jw_street) as well as a regular comparison feature (represented by J/W)"""

    matching_conf["comparison_features"] = [
        {
            "alias": "jw_street",
            "column_name": "street",
            "boundary": "enum_dist",
            "comparison_type": "jaro_winkler_street",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
    ]

    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(matching_conf)

    # Create pandas DFs of the step_1 potential matches table
    potential_matches_df = spark.table("potential_matches_prepped").toPandas()

    # Make assertions on the data
    assert len(potential_matches_df.id_a) == 5
    assert len(potential_matches_df.jw_street) == 5
    assert (
        potential_matches_df.query("id_a == 20 and id_b == 50")["jw_street"].iloc[0]
        == 0
    )
    assert (
        potential_matches_df.query("id_a == 10 and id_b == 10")["jw_street"].iloc[0]
        == 0.95
    )
    assert (
        potential_matches_df.query("id_a == 20 and id_b == 30")["namelast_jw"].iloc[0]
        == 1
    )
    assert (
        0.48
        < potential_matches_df.query("id_a == 30 and id_b == 50")["jw_street"].iloc[0]
        < 0.5
    )


def test_step_2_maximum_jaro_winkler(spark, matching_conf, matching):
    """Test creation of maximum_jaro_winkler comparison feature"""

    matching_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "namefrst_jw",
            "column_name": "namefrst",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "maximum_jw",
            "column_names": ["namelast", "namefrst"],
            "comparison_type": "maximum_jaro_winkler",
        },
    ]

    matching_conf["training"]["dependent_var"] = "matching"
    matching_conf["training"]["independent_vars"] = [
        "neighbor_namelast_jw_rate",
        "neighbor_namelast_jw_rate_threshold",
        "namelast_jw",
    ]

    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(matching_conf)

    # Create pandas DFs of the step_1 potential matches table
    potential_matches_prepped_df = spark.table("potential_matches_prepped").toPandas()

    # Create pandas DFs of the step_2 potential matches table
    # potential_matches_prepped_df = spark.table("potential_matches").toPandas()

    # Make assertions on the data
    assert len(potential_matches_prepped_df.id_a) == 5
    assert len(potential_matches_prepped_df.maximum_jw) == 5
    assert (
        potential_matches_prepped_df.query("id_a == 20 and id_b == 30")[
            "maximum_jw"
        ].iloc[0]
        == 1
    )
    assert (
        0.97
        > potential_matches_prepped_df.query("id_a == 10 and id_b == 10")[
            "maximum_jw"
        ].iloc[0]
        > 0.96
    )
    assert (
        0.855
        > potential_matches_prepped_df.query("id_a == 30 and id_b == 30")[
            "maximum_jw"
        ].iloc[0]
        > 0.84
    )
    assert (
        0.80
        < potential_matches_prepped_df.query("id_a == 30 and id_b == 30")[
            "namefrst_jw"
        ].iloc[0]
        < 0.81
    )


def test_step_2_max_jaro_winkler(
    spark, matching_household_conf, matching, preprocessing
):
    """Test creation of max_jaro_winkler comparison feature"""

    matching_household_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "related_individual_max_jw",
            "column_name": "namefrst_related",
            "comparison_type": "max_jaro_winkler",
        },
    ]

    matching_household_conf["feature_selections"] = [
        {
            "output_col": "namefrst_related",
            "input_col": "namefrst_std",
            "transform": "related_individuals",
            "family_id": "serialp",
            "relate_col": "relate",
            "top_code": 10,
            "bottom_code": 3,
        }
    ]
    matching_household_conf["training"]["dependent_var"] = "matching"
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

    # Create pandas DFs of the step_1 potential matches table
    potential_matches_prepped_df = spark.table("potential_matches_prepped").toPandas()

    # Make assertions on the data
    assert len(potential_matches_prepped_df.id_a) == 37
    assert len(potential_matches_prepped_df.related_individual_max_jw) == 37
    assert (
        potential_matches_prepped_df.query(
            "id_a == '3575c9ba-1527-4ca2-aff0-d7c2d1efb421 ' and id_b == '49e53dbc-fe8e-4e55-8cb9-a1d93c284d98 '"
        )["related_individual_max_jw"].iloc[0]
        == 1
    )
    assert potential_matches_prepped_df.query(
        "id_a == '3575c9ba-1527-4ca2-aff0-d7c2d1efb421 ' and id_b == '49e53dbc-fe8e-4e55-8cb9-a1d93c284d98 '"
    )["namefrst_related_a"].iloc[0] == ["john", "mary"]
    assert potential_matches_prepped_df.query(
        "id_a == '3575c9ba-1527-4ca2-aff0-d7c2d1efb421 ' and id_b == '49e53dbc-fe8e-4e55-8cb9-a1d93c284d98 '"
    )["namefrst_related_b"].iloc[0] == ["john", "maggie"]
    assert (
        round(
            potential_matches_prepped_df.query(
                "id_a == 'ad6442b5-42bc-4c2e-a517-5a951d989a92 ' and id_b == 'ae7261c3-7d71-4ea1-997f-5d1a68c18777 '"
            )["related_individual_max_jw"].iloc[0],
            2,
        )
        == 0.63
    )
    assert potential_matches_prepped_df.query(
        "id_a == 'ad6442b5-42bc-4c2e-a517-5a951d989a92 ' and id_b == 'ae7261c3-7d71-4ea1-997f-5d1a68c18777 '"
    )["namefrst_related_a"].iloc[0] == ["sally"]
    assert potential_matches_prepped_df.query(
        "id_a == 'ad6442b5-42bc-4c2e-a517-5a951d989a92 ' and id_b == 'ae7261c3-7d71-4ea1-997f-5d1a68c18777 '"
    )["namefrst_related_b"].iloc[0] == ["mary"]


def test_step_2_rel_jaro_winkler(
    spark, matching_household_conf, matching, preprocessing
):
    """Test creation of max_jaro_winkler comparison feature"""

    matching_household_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "rel",
            "column_name": "namefrst_related_rows",
            "comparison_type": "rel_jaro_winkler",
            "jw_threshold": 0.9,
            "age_threshold": 5,
        },
        {
            "alias": "rel_threshold",
            "column_name": "namefrst_related_rows",
            "comparison_type": "rel_jaro_winkler",
            "jw_threshold": 0.9,
            "age_threshold": 5,
            "lower_threshold": 1,
        },
    ]

    matching_household_conf["feature_selections"] = [
        {
            "family_id": "serialp",
            "input_cols": ["namefrst_std", "birthyr", "sex"],
            "output_col": "namefrst_related_rows",
            "transform": "related_individual_rows",
            "filters": [
                {"column": "relate", "max": 10, "min": 3},
                {"column": "age", "max": 99, "min": 8, "dataset": "b"},
            ],
        }
    ]

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

    # Create pandas DFs of the step_1 potential matches table
    potential_matches_df = spark.table("potential_matches_prepped").toPandas()

    # Make assertions on the data
    assert len(potential_matches_df.id_a) == 37
    assert len(potential_matches_df.namefrst_related_rows_a) == 37
    assert len(potential_matches_df.rel) == 37
    assert len(potential_matches_df.rel_threshold) == 37

    assert (
        len(
            potential_matches_df.query(
                "id_a == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 ' and id_b == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 '"
            )["namefrst_related_rows_a"].iloc[0]
        )
        == 3
    )
    assert (
        len(
            potential_matches_df.query(
                "id_a == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 ' and id_b == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 '"
            )["namefrst_related_rows_b"].iloc[0]
        )
        == 1
    )
    assert (
        potential_matches_df.query(
            "id_a == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 ' and id_b == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 '"
        )["rel"].iloc[0]
        == 1
    )
    assert potential_matches_df.query(
        "id_a == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 ' and id_b == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 '"
    )["rel_threshold"].iloc[0]


def test_step_2_jaro_winkler_rate(
    spark, matching_household_conf, matching, preprocessing
):
    """Test creation of jaro_winkler_rate comparison feature"""

    matching_household_conf["comparison_features"] = [
        {
            "alias": "neighbor_namelast_jw_rate",
            "column_name": "namelast_neighbors",
            "comparison_type": "jaro_winkler_rate",
            "jw_threshold": 0.95,
        },
        {
            "alias": "neighbor_namelast_jw_rate_threshold",
            "column_name": "namelast_neighbors",
            "comparison_type": "jaro_winkler_rate",
            "jw_threshold": 0.95,
            "threshold": "0.8",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
    ]

    matching_household_conf["feature_selections"] = [
        {
            "output_column": "namelast_neighbors",
            "input_column": "namelast_clean",
            "transform": "neighbor_aggregate",
            "neighborhood_column": "enumdist",
            "sort_column": "serialp",
            "range": 10,
        }
    ]
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

    # Create pandas DFs of the step_1 potential matches table
    potential_matches_df = spark.table("potential_matches_prepped").toPandas()

    # Make assertions on the data
    assert len(potential_matches_df.id_a) == 37
    assert len(potential_matches_df.neighbor_namelast_jw_rate) == 37
    assert len(potential_matches_df.neighbor_namelast_jw_rate_threshold) == 37
    assert (
        round(
            potential_matches_df.query(
                "id_a == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 ' and id_b == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 '"
            )["neighbor_namelast_jw_rate"].iloc[0],
            2,
        )
        == 0.92
    )
    assert potential_matches_df.query(
        "id_a == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 ' and id_b == 'd3217545-3453-4d96-86c0-d6a3e60fb2f8 '"
    )["neighbor_namelast_jw_rate_threshold"].iloc[0]
    assert (
        potential_matches_df.query(
            "id_a == '426f2cbe-32e1-45eb-9f86-89a2b9116b7e ' and id_b == 'a499b0dc-7ac0-4d61-b493-91a3036c712e '"
        )["neighbor_namelast_jw_rate"].iloc[0]
        == 0.75
    )
    assert not potential_matches_df.query(
        "id_a == '426f2cbe-32e1-45eb-9f86-89a2b9116b7e ' and id_b == 'a499b0dc-7ac0-4d61-b493-91a3036c712e '"
    )["neighbor_namelast_jw_rate_threshold"].iloc[0]
    assert (
        potential_matches_df.query(
            "id_a == '92277f0b-1476-41f5-9dc8-bf83672616d0' and id_b == '9e807937-de09-414c-bfb2-ac821e112929 '"
        )["neighbor_namelast_jw_rate"].iloc[0]
        == 0
    )
    assert not potential_matches_df.query(
        "id_a == '92277f0b-1476-41f5-9dc8-bf83672616d0' and id_b == '9e807937-de09-414c-bfb2-ac821e112929 '"
    )["neighbor_namelast_jw_rate_threshold"].iloc[0]
    assert (
        potential_matches_df.query(
            "id_a == 'a499b0dc-7ac0-4d61-b493-91a3036c712e ' and id_b == '426f2cbe-32e1-45eb-9f86-89a2b9116b7e '"
        )["neighbor_namelast_jw_rate"].iloc[0]
        == 0.9
    )
    assert potential_matches_df.query(
        "id_a == 'a499b0dc-7ac0-4d61-b493-91a3036c712e ' and id_b == '426f2cbe-32e1-45eb-9f86-89a2b9116b7e '"
    )["neighbor_namelast_jw_rate_threshold"].iloc[0]


def test_step_2_JW_double_array_blocking_conf(spark, matching_conf, matching, capsys):
    """Test matching step 2 to ensure that comparison features are generated (can a regular comparison (as represented by J/W) still run if there's NOT a distance lookup feature)"""
    matching_conf["blocking_steps"] = [[{"column_name": "sex"}]]
    matching_conf.pop("blocking")

    matching_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        }
    ]

    matching.run_step(0)
    matching.run_step(1)

    # Create pandas DFs of the step_2 potential matches table
    potential_matches_df = spark.table("potential_matches").toPandas()

    # Make assertions on the data
    assert len(potential_matches_df.id_a) == 5
    assert len(potential_matches_df.namelast_jw) == 5
    assert (
        potential_matches_df.query("id_a == 20 and id_b == 30")["namelast_jw"].iloc[0]
        == 1
    )
    assert (
        potential_matches_df.query("id_a == 10 and id_b == 10")["namelast_jw"].iloc[0]
        > 0.87
    )

    captured = capsys.readouterr()
    assert (
        "DEPRECATION WARNING: The config value 'blocking_steps' has been renamed to 'blocking' and is now just a single array of objects."
        in captured.out
    )


def test_step_2_comparison_features_comp_c_and_caution(
    spark, matching_comparison_conf, matching
):
    """Test a comparison feature with comp_a, comp_b, and comp_c using spouse caution feature as example"""
    matching_comparison_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "sp_caution",
            "column_names": ["spouse_bpl", "spouse_birthyr", "durmarr"],
            "comparison_type": "or",
            "comp_a": {"column_name": "spouse_bpl", "comparison_type": "not_equals"},
            "comp_b": {
                "column_name": "spouse_birthyr",
                "comparison_type": "abs_diff",
                "lower_threshold": 5,
            },
            "comp_c": {
                "column_name": "durmarr",
                "comparison_type": "new_marr",
                "upper_threshold": 7,
            },
        },
        {
            "alias": "m_caution",
            "column_names": ["mbpl", "mother_birthyr", "stepmom"],
            "comparison_type": "or",
            "comp_a": {"column_name": "mbpl", "comparison_type": "not_equals"},
            "comp_b": {
                "column_name": "mother_birthyr",
                "comparison_type": "abs_diff",
                "lower_threshold": 5,
            },
            "comp_c": {
                "column_name": "stepmom",
                "comparison_type": "parent_step_change",
            },
        },
        {
            "alias": "new_marr",
            "column_name": "durmarr",
            "comparison_type": "new_marr",
            "upper_threshold": 7,
        },
        {
            "alias": "existing_marr",
            "column_name": "durmarr",
            "comparison_type": "existing_marr",
            "lower_threshold": 8,
        },
        {
            "alias": "mom_step_change",
            "column_name": "stepmom",
            "comparison_type": "parent_step_change",
        },
    ]
    matching_comparison_conf["streamline_potential_match_generation"] = True

    matching_comparison_conf["training"] = {
        "independent_vars": [
            "namelast_jw",
            "sp_caution",
            "m_caution",
            "new_marr",
            "existing_marr",
            "mom_step_change",
        ],
        "dependent_var": "match",
        "use_training_data_features": False,
        # "use_potential_matches_features": False,
        # "check_for_null_columns": False,
    }

    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(matching.link_run.config)

    # Create pandas DFs of the step_2 potential matches table
    assert sorted(spark.table("potential_matches").columns) == [
        "id_a",
        "id_b",
        "namelast_jw",
    ]

    potential_matches_prepped = spark.table("potential_matches_prepped").toPandas()
    assert not potential_matches_prepped.query("id_a == 10 and id_b == 10")[
        "sp_caution"
    ].iloc[0]
    assert not potential_matches_prepped.query("id_a == 10 and id_b == 10")[
        "m_caution"
    ].iloc[0]
    assert potential_matches_prepped.query("id_a == 20 and id_b == 20")[
        "sp_caution"
    ].iloc[0]
    assert potential_matches_prepped.query("id_a == 20 and id_b == 20")[
        "m_caution"
    ].iloc[0]
    assert potential_matches_prepped.query("id_a == 30 and id_b == 30")[
        "sp_caution"
    ].iloc[0]
    assert potential_matches_prepped.query("id_a == 30 and id_b == 30")[
        "m_caution"
    ].iloc[0]
    assert potential_matches_prepped.query("id_a == 30 and id_b == 40")[
        "sp_caution"
    ].iloc[0]
    assert potential_matches_prepped.query("id_a == 30 and id_b == 40")[
        "m_caution"
    ].iloc[0]
    assert not potential_matches_prepped.query("id_a == 10 and id_b == 10")[
        "new_marr"
    ].iloc[0]
    assert potential_matches_prepped.query("id_a == 10 and id_b == 10")[
        "existing_marr"
    ].iloc[0]
    assert not potential_matches_prepped.query("id_a == 10 and id_b == 10")[
        "mom_step_change"
    ].iloc[0]
    assert potential_matches_prepped.query("id_a == 10 and id_b == 40")[
        "new_marr"
    ].iloc[0]
    assert not potential_matches_prepped.query("id_a == 10 and id_b == 40")[
        "existing_marr"
    ].iloc[0]
    assert potential_matches_prepped.query("id_a == 10 and id_b == 40")[
        "mom_step_change"
    ].iloc[0]


def test_step_2_comparison_features_comp_d_and_caution(
    spark, matching_comparison_conf, matching
):
    """Test a comparison feature with comp_a, comp_b, comp_c, and comp_d using mixed booleans and caution features as example"""
    matching.link_run.print_sql = True
    matching_comparison_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "new_marr",
            "column_name": "durmarr",
            "comparison_type": "new_marr",
            "upper_threshold": 7,
        },
        {
            "alias": "existing_marr",
            "column_name": "durmarr",
            "comparison_type": "existing_marr",
            "lower_threshold": 8,
        },
        {
            "alias": "mom_present_both_years",
            "column_name": "momloc",
            "comparison_type": "present_both_years",
        },
        {
            "alias": "spouse_present_both_years",
            "column_name": "sploc",
            "comparison_type": "present_both_years",
        },
        {
            "alias": "mom_step_change",
            "column_name": "stepmom",
            "comparison_type": "parent_step_change",
        },
        {
            "alias": "m_caution",
            "column_names": ["mbpl", "mother_birthyr", "stepmom", "momloc"],
            "comparison_type": "caution_comp_4",
            "comp_a": {"column_name": "mbpl", "comparison_type": "not_equals"},
            "comp_b": {
                "column_name": "mother_birthyr",
                "comparison_type": "abs_diff",
                "lower_threshold": 5,
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
            "alias": "sp_caution",
            "column_names": ["spouse_bpl", "spouse_birthyr", "durmarr", "sploc"],
            "comparison_type": "caution_comp_4",
            "comp_a": {"column_name": "spouse_bpl", "comparison_type": "not_equals"},
            "comp_b": {
                "column_name": "spouse_birthyr",
                "comparison_type": "abs_diff",
                "lower_threshold": 5,
            },
            "comp_c": {
                "column_name": "durmarr",
                "comparison_type": "new_marr",
                "upper_threshold": 7,
            },
            "comp_d": {"column_name": "sploc", "comparison_type": "present_both_years"},
        },
    ]

    matching_comparison_conf["training"] = {}

    matching_comparison_conf["training"]["dependent_var"] = "matching"
    matching_comparison_conf["training"]["independent_vars"] = [
        "neighbor_namelast_jw_rate",
        "neighbor_namelast_jw_rate_threshold",
        "namelast_jw",
    ]

    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(matching_comparison_conf)

    # Create pandas DFs of the step_2 potential matches table
    potential_matches_df = spark.table("potential_matches_prepped").toPandas()

    assert potential_matches_df.shape == (12, 34)
    assert not potential_matches_df.query("id_a == 10 and id_b == 10")[
        "sp_caution"
    ].iloc[0]
    assert not potential_matches_df.query("id_a == 10 and id_b == 10")[
        "m_caution"
    ].iloc[0]
    assert not potential_matches_df.query("id_a == 20 and id_b == 20")[
        "sp_caution"
    ].iloc[0]
    assert potential_matches_df.query("id_a == 20 and id_b == 20")["m_caution"].iloc[0]
    assert potential_matches_df.query("id_a == 10 and id_b == 30")["sp_caution"].iloc[0]
    assert not potential_matches_df.query("id_a == 10 and id_b == 30")[
        "m_caution"
    ].iloc[0]
    assert not potential_matches_df.query("id_a == 30 and id_b == 40")[
        "sp_caution"
    ].iloc[0]
    assert not potential_matches_df.query("id_a == 30 and id_b == 40")[
        "m_caution"
    ].iloc[0]
    assert not potential_matches_df.query("id_a == 10 and id_b == 10")["new_marr"].iloc[
        0
    ]
    assert potential_matches_df.query("id_a == 10 and id_b == 10")[
        "existing_marr"
    ].iloc[0]
    assert not potential_matches_df.query("id_a == 10 and id_b == 10")[
        "mom_step_change"
    ].iloc[0]
    assert potential_matches_df.query("id_a == 10 and id_b == 40")["new_marr"].iloc[0]
    assert not potential_matches_df.query("id_a == 10 and id_b == 40")[
        "existing_marr"
    ].iloc[0]
    assert potential_matches_df.query("id_a == 10 and id_b == 40")[
        "mom_step_change"
    ].iloc[0]


def test_step_2_neither_are_null(
    spark, matching_household_conf, matching, preprocessing
):
    """Test a comparison feature with comp_a, comp_b, and comp_c using spouse caution feature as example"""
    matching_household_conf["feature_selections"] = [
        {
            "output_col": "spouse_namefrst",
            "transform": "attach_family_col",
            "other_col": "namefrst_std",
            "person_pointer": "sploc",
            "family_id": "serialp",
            "person_id": "pernum",
        },
        {
            "output_col": "father_namefrst",
            "transform": "attach_family_col",
            "other_col": "namefrst_std",
            "person_pointer": "poploc",
            "family_id": "serialp",
            "person_id": "pernum",
        },
        {
            "output_col": "mother_namefrst",
            "transform": "attach_family_col",
            "other_col": "namefrst_std",
            "person_pointer": "momloc",
            "family_id": "serialp",
            "person_id": "pernum",
        },
    ]

    matching_household_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "sp_present",
            "column_name": "spouse_namefrst",
            "comparison_type": "neither_are_null",
        },
        {
            "alias": "m_present",
            "column_name": "mother_namefrst",
            "comparison_type": "neither_are_null",
        },
        {
            "alias": "f_present",
            "column_name": "father_namefrst",
            "comparison_type": "neither_are_null",
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
    assert len(potential_matches_df.sp_present) == 37
    assert len(potential_matches_df.mother_namefrst_a == 37)
    assert pd.isnull(
        potential_matches_df.query(
            "id_a == '92277f0b-1476-41f5-9dc8-bf83672616d0' and id_b == '9e807937-de09-414c-bfb2-ac821e112929 '"
        )["father_namefrst_a"].iloc[0]
    )
    assert pd.isnull(
        potential_matches_df.query(
            "id_a == '92277f0b-1476-41f5-9dc8-bf83672616d0' and id_b == '9e807937-de09-414c-bfb2-ac821e112929 '"
        )["father_namefrst_b"].iloc[0]
    )
    assert (
        potential_matches_df.query(
            "id_a == '92277f0b-1476-41f5-9dc8-bf83672616d0' and id_b == '9e807937-de09-414c-bfb2-ac821e112929 '"
        )["f_present"].iloc[0]
        == 0
    )
    assert pd.notnull(
        potential_matches_df.query(
            "id_a == 'bfe1080e-2e67-4a8c-a6e1-ed94ea103712 ' and id_b == 'bfe1080e-2e67-4a8c-a6e1-ed94ea103712 '"
        )["mother_namefrst_a"].iloc[0]
    )
    assert pd.notnull(
        potential_matches_df.query(
            "id_a == 'bfe1080e-2e67-4a8c-a6e1-ed94ea103712 ' and id_b == 'bfe1080e-2e67-4a8c-a6e1-ed94ea103712 '"
        )["mother_namefrst_b"].iloc[0]
    )
    assert (
        potential_matches_df.query(
            "id_a == 'bfe1080e-2e67-4a8c-a6e1-ed94ea103712 ' and id_b == 'bfe1080e-2e67-4a8c-a6e1-ed94ea103712 '"
        )["m_present"].iloc[0]
        == 1
    )


def test_step_2_create_features_sql_condition(
    spark, conf, matching, datasource_sql_condition_input
):
    """Test a comparison feature with comp_a, comp_b, and comp_c using spouse caution feature as example"""
    conf["comparison_features"] = [
        {
            "alias": "marst_warn",
            "column_name": "marst",
            "comparison_type": "sql_condition",
            "condition": """case
                            when ((a.marst == 6) AND (b.marst > 0) AND (b.marst < 6)) then 1
                            when ((a.marst > 0) and (a.marst < 6) AND (b.marst == 6)) then 1
                            when (((a.marst == 4) OR (a.marst == 5)) AND ((b.marst == 1) OR (b.marst == 2))) then 1
                            else 0 end""",
        },
        {
            "alias": "key_marst_warn",
            "column_name": "key_marst_warn",
            "comparison_type": "fetch_a",
        },
    ]
    conf["training"] = {
        "dependent_var": "match",
        "independent_vars": ["marst_warn", "key_marst_warn"],
    }

    pa_path, pb_path, pm_path = datasource_sql_condition_input

    matching.spark.read.csv(pa_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_a")
    matching.spark.read.csv(pb_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_b")
    matching.spark.read.csv(pm_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("potential_matches")

    LinkStepScore(matching)._create_features(conf)

    pmp = spark.table("potential_matches_prepped").toPandas()
    assert pmp["key_marst_warn"].equals(pmp["marst_warn"])


def test_step_1_transform_calc_nativity(
    preprocessing, spark, preprocessing_conf_19thc_nativity_conf, matching
):
    """Test attach_family_col transform on data containing households"""

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(preprocessing_conf_19thc_nativity_conf)

    pmp = spark.table("potential_matches_prepped").toPandas().sort_values(["id_a"])

    assert pmp["key_mbpl_match"].equals(pmp["mbpl_match"])
    assert pmp["key_fbpl_match"].equals(pmp["fbpl_match"])
    assert pmp["key_mfbpl_match"].equals(pmp["mfbpl_match"])
    assert pmp["key_m_caution_1870_1880"].equals(pmp["m_caution_1870_1880"])
    assert pmp["key_m_caution_1850_1860"].equals(pmp["m_caution_1850_1860"])


def test_step_1_transform_calc_mfbpl_match(
    preprocessing,
    spark,
    preprocessing_conf_19thc_nativity_conf,
    matching,
    datasource_calc_mfbpl_pm_data,
):
    """Test attach_family_col transform on data containing households"""
    path_a, path_b = datasource_calc_mfbpl_pm_data

    spark.read.csv(path_a, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_a")
    spark.read.csv(path_b, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_b")

    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(preprocessing_conf_19thc_nativity_conf)

    pmp = spark.table("potential_matches_prepped").toPandas().sort_values(["id_a"])

    assert pmp["key_mbpl_match"].equals(pmp["mbpl_match"])
    assert pmp["key_fbpl_match"].equals(pmp["fbpl_match"])
    assert pmp["key_mfbpl_match"].equals(pmp["mfbpl_match"])
    assert pmp["key_m_caution_1870_1880"].equals(pmp["m_caution_1870_1880"])
    assert pmp["key_m_caution_1850_1860"].equals(pmp["m_caution_1850_1860"])


def test_caution_comp_012(
    preprocessing,
    spark,
    preprocessing_conf_19thc_caution_conf,
    matching,
):
    """Test multiple clause comparison with 0, 1, and 2 outcome values."""
    preprocessing.run_step(0)
    preprocessing.run_step(1)

    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(preprocessing_conf_19thc_caution_conf)

    pmp = spark.table("potential_matches_prepped").toPandas().sort_values(["id_a"])

    assert pmp["key_mbpl_match"].equals(pmp["mbpl_match"])
    assert pmp["key_fbpl_match"].equals(pmp["fbpl_match"])
    assert pmp["key_mfbpl_match"].equals(pmp["mfbpl_match"])
    assert pmp["key_m_caution_1870_1880"].equals(pmp["m_caution_1870_1880"])
    assert pmp["key_m_caution_1850_1860"].equals(pmp["m_caution_1850_1860"])

    assert pmp["key_intermediate_mbpl_range_not_equals"].equals(
        pmp["intermediate_mbpl_range_not_equals"]
    )
    assert pmp["key_intermediate_mbpl_range_not_zero_and_not_equals"].equals(
        pmp["intermediate_mbpl_range_not_zero_and_not_equals"]
    )
    assert pmp["key_intermediate_mother_birthyr_abs_diff_5"].equals(
        pmp["intermediate_mother_birthyr_abs_diff_5"]
    )
    assert pmp["key_intermediate_stepmom_parent_step_change"].equals(
        pmp["intermediate_stepmom_parent_step_change"]
    )
    assert pmp["key_intermediate_momloc_present_both_years"].equals(
        pmp["intermediate_momloc_present_both_years"]
    )
    assert pmp["key_m_caution_cc3_012"].equals(pmp["m_caution_cc3_012"])
    assert pmp["key_m_caution_cc4_012"].equals(pmp["m_caution_cc4_012"])
