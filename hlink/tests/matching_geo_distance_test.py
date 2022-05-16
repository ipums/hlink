# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import math
import pandas as pd
import numpy as np
from hlink.linking.matching.link_step_score import LinkStepScore


def test_step_2_geo_distance_1_key_jaro_winkler(
    spark, matching_conf, matching, state_dist_path
):
    """Test matching step 2 to ensure that comparison features are generated (both regular (represented by J/W) and as requiring a distance lookup file)"""

    matching_conf["comparison_features"] = [
        {
            "alias": "state_distance",
            "comparison_type": "geo_distance",
            "key_count": 1,
            "table_name": "state_distance_lookup",
            "distances_file": state_dist_path,
            "column_name": "bpl",
            "loc_a": "statecode1",
            "loc_b": "statecode2",
            "distance_col": "dist",
        },
        {
            "alias": "distance_squared",
            "comparison_type": "geo_distance",
            "key_count": 1,
            "table_name": "state_distance_lookup",
            "distances_file": state_dist_path,
            "column_name": "bpl",
            "loc_a": "statecode1",
            "loc_b": "statecode2",
            "distance_col": "dist",
            "power": 2,
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
    ]
    matching_conf["comparisons"] = {
        "comp_a": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        # This confusing "comp_b" section is here to include the state_distance
        # and distance_squared comparison features in the output of potential matches
        "comp_b": {
            "comp_a": {
                "comp_a": {
                    "feature_name": "state_distance",
                    "threshold": 0,
                    "comparison_type": "threshold",
                },
                "comp_b": {
                    "feature_name": "distance_squared",
                    "threshold": 0,
                    "comparison_type": "threshold",
                },
                "operator": "AND",
            },
            "comp_b": {
                "feature_name": "namelast_jw",
                "threshold": 0.8,
                "comparison_type": "threshold",
            },
            "operator": "AND",
        },
        "operator": "OR",
    }

    matching.run_step(0)
    matching.run_step(1)

    # Create pandas DFs of the step_2 potential matches table
    potential_matches_df = spark.table("potential_matches").toPandas()

    # Make assertions on the data
    assert len(potential_matches_df.id_a) == 5
    assert len(potential_matches_df.state_distance) == 5
    assert potential_matches_df.query("id_a == 10")["state_distance"].iloc[0] == 1427.1
    assert (
        potential_matches_df.query("id_a == 10")["distance_squared"].iloc[0] == 2036329
    )
    assert (
        potential_matches_df.query("id_a == 20 and id_b == 30")["namelast_jw"].iloc[0]
        == 1
    )
    assert pd.isnull(
        potential_matches_df.query("id_a == 30 and id_b == 30")["state_distance"].iloc[
            0
        ]
    )
    assert (
        potential_matches_df.query("id_a == 30 and id_b == 50")["namelast_jw"]
        .round(2)
        .iloc[0]
        == 0.85
    )


def test_step_2_geo_distance_ids_only(spark, matching_conf, matching, state_dist_path):
    """Test matching step 2 to ensure that comparison features are generated (both regular (represented by J/W) and as requiring a distance lookup file)"""

    matching_conf["comparison_features"] = [
        {
            "alias": "state_distance",
            "comparison_type": "geo_distance",
            "key_count": 1,
            "table_name": "state_distance_lookup",
            "distances_file": state_dist_path,
            "column_name": "bpl",
            "loc_a": "statecode1",
            "loc_b": "statecode2",
            "distance_col": "dist",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
    ]
    matching_conf["comparisons"] = {
        "comp_a": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        # This confusing "comp_b" section is here to include the state_distance
        # in the output of potential matches
        "comp_b": {
            "comp_a": {
                "feature_name": "state_distance",
                "threshold": 0,
                "comparison_type": "threshold",
            },
            "comp_b": {
                "feature_name": "namelast_jw",
                "threshold": 0.8,
                "comparison_type": "threshold",
            },
            "operator": "AND",
        },
        "operator": "OR",
    }
    matching_conf["streamline_potential_match_generation"] = True

    matching.run_step(0)
    matching.run_step(1)

    # Create pandas DFs of the step_2 potential matches table
    potential_matches_df = spark.table("potential_matches").toPandas()

    # Make assertions on the data
    assert len(potential_matches_df.id_a) == 5
    assert len(potential_matches_df.id_b) == 5
    assert potential_matches_df.shape == (5, 4)


def test_step_2_geo_distance_2_keys(
    spark, preprocessing, matching, matching_conf_counties, county_dist_path
):
    """Test county distance code transform"""
    matching_conf_counties["column_mappings"] = [
        {"column_name": "county_p", "alias": "county"},
        {"column_name": "statefip_p", "alias": "statefip"},
        {"column_name": "namelast"},
        {"column_name": "sex"},
    ]
    matching_conf_counties["blocking"] = [{"column_name": "sex"}]
    matching_conf_counties["comparison_features"] = [
        {
            "alias": "county_distance",
            "comparison_type": "geo_distance",
            "key_count": 2,
            "table_name": "county_distance_lookup",
            "distances_file": county_dist_path,
            "column_names": ["county", "state"],
            "source_column_a": "county",
            "source_column_b": "statefip",
            "loc_a_0": "county0",
            "loc_a_1": "county1",
            "loc_b_0": "state0",
            "loc_b_1": "state1",
            "distance_col": "distance",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
    ]

    matching_conf_counties["comparisons"] = {
        "comp_a": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        # This confusing "comp_b" section is here to include the state_distance
        # in the output of potential matches
        "comp_b": {
            "comp_a": {
                "feature_name": "county_distance",
                "threshold": 0,
                "comparison_type": "threshold",
            },
            "comp_b": {
                "feature_name": "namelast_jw",
                "threshold": 0.8,
                "comparison_type": "threshold",
            },
            "operator": "AND",
        },
        "operator": "OR",
    }

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    matching.run_step(0)
    matching.run_step(1)
    matches = spark.table("potential_matches").toPandas()
    assert (
        math.floor(
            matches.query("id_a == 10 and id_b == 40")["county_distance"].iloc[0]
        )
        == 1613771
    )
    assert math.isnan(
        matches.query("id_a == 30 and id_b == 40")["county_distance"].iloc[0]
    )


def test_step_2_geo_distance_secondary_lookup(
    spark,
    preprocessing,
    matching,
    matching_conf_counties,
    county_dist_path,
    state_dist_path,
):
    """Test county distance code transform"""
    matching_conf_counties["column_mappings"] = [
        {"column_name": "county_p", "alias": "county"},
        {"column_name": "statefip_p", "alias": "statefip"},
        {"column_name": "namelast"},
        {"column_name": "sex"},
    ]
    matching_conf_counties["blocking"] = [{"column_name": "sex"}]
    matching_conf_counties["comparison_features"] = [
        {
            "alias": "county_distance",
            "comparison_type": "geo_distance",
            "key_count": 2,
            "table_name": "county_distance_lookup",
            "distances_file": county_dist_path,
            "column_names": ["county", "state"],
            "source_column_a": "county",
            "source_column_b": "statefip",
            "loc_a_0": "county0",
            "loc_a_1": "county1",
            "loc_b_0": "state0",
            "loc_b_1": "state1",
            "distance_col": "distance",
            "secondary_key_count": 1,
            "secondary_table_name": "state_distance_lookup",
            "secondary_distances_file": state_dist_path,
            "secondary_source_column": "statefip",
            "secondary_loc_a": "statecode1",
            "secondary_loc_b": "statecode2",
            "secondary_distance_col": "dist",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
    ]

    matching_conf_counties["comparisons"] = {
        "comp_a": {
            "feature_name": "namelast_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        # This confusing "comp_b" section is here to include the state_distance
        # in the output of potential matches
        "comp_b": {
            "comp_a": {
                "feature_name": "county_distance",
                "threshold": 0,
                "comparison_type": "threshold",
            },
            "comp_b": {
                "feature_name": "namelast_jw",
                "threshold": 0.8,
                "comparison_type": "threshold",
            },
            "operator": "AND",
        },
        "operator": "OR",
    }

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    matching.run_step(0)
    matching.run_step(1)
    matches = spark.table("potential_matches").toPandas()
    assert (
        math.floor(
            matches.query("id_a == 10 and id_b == 40")["county_distance"].iloc[0]
        )
        == 1613771
    )
    assert (
        math.floor(
            matches.query("id_a == 30 and id_b == 40")["county_distance"].iloc[0]
        )
        == 785
    )
    assert (
        math.floor(
            matches.query("id_a == 30 and id_b == 20")["county_distance"].iloc[0]
        )
        == 695
    )


def test_step_2_geo_distance_1_and_2_keys(
    spark,
    preprocessing,
    matching,
    matching_conf_counties,
    county_dist_path,
    state_dist_path,
):
    """Test county distance code transform"""
    matching_conf_counties["column_mappings"] = [
        {"column_name": "county_p", "alias": "county"},
        {"column_name": "statefip_p", "alias": "statefip"},
        {"column_name": "namelast"},
        {"column_name": "sex"},
    ]
    matching_conf_counties["blocking"] = [{"column_name": "sex"}]
    matching_conf_counties["comparisons"] = {
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

    matching_conf_counties["comparison_features"] = [
        {
            "alias": "county_distance",
            "comparison_type": "geo_distance",
            "key_count": 2,
            "table_name": "county_distance_lookup",
            "distances_file": county_dist_path,
            "column_names": ["county", "state"],
            "source_column_a": "county",
            "source_column_b": "statefip",
            "loc_a_0": "county0",
            "loc_a_1": "county1",
            "loc_b_0": "state0",
            "loc_b_1": "state1",
            "distance_col": "distance",
        },
        {
            "alias": "state_distance",
            "comparison_type": "geo_distance",
            "key_count": 1,
            "table_name": "state_distance_lookup",
            "distances_file": state_dist_path,
            "column_name": "statefip",
            "loc_a": "statecode1",
            "loc_b": "statecode2",
            "distance_col": "dist",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
    ]
    matching_conf_counties["training"] = {}
    matching_conf_counties["training"]["dependent_var"] = "matching"
    matching_conf_counties["training"]["independent_vars"] = []

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    matching.run_step(0)
    matching.run_step(1)

    LinkStepScore(matching)._create_features(matching_conf_counties)

    matches = spark.table("potential_matches_prepped").toPandas()
    assert (
        math.floor(
            matches.query("id_a == 10 and id_b == 40")["county_distance"].iloc()[0]
        )
        == 1613771
    )

    assert (
        matches.query("id_a == 10 and id_b == 40")["state_distance"].iloc()[0] == 917.5
    )

    assert np.isnan(
        matches.query("id_a == 30 and id_b == 40")["county_distance"].iloc()[0]
    )
