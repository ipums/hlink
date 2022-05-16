# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pytest


@pytest.mark.skip(
    reason="We still want to test that these aggregate features are being created correctly, but we need to refactor this test to account for the fact that aggregate features are now being created in a different step (step 4 doesn't exist anymore and the functionality was moved in the code)."
)
def test_step_4_aggregate_features(
    spark, matching_conf, matching, potential_matches_agg_path
):
    """Test adding aggregate features (hits, hits2, exact_all_mult, etc.) to potential matches"""
    matching_conf["id_column"] = "histid"
    matching_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {"alias": "exact"},
        {"alias": "exact_all"},
    ]
    matching_conf["training"] = {
        "independent_vars": [
            "namelast_jw",
            "exact",
            "exact_all",
            "hits",
            "hits2",
            "exact_mult",
            "exact_all_mult",
            "exact_all_mult2",
        ]
    }

    potential_matches = matching.spark.read.csv(
        potential_matches_agg_path, header=True, inferSchema=True
    )
    potential_matches.write.mode("overwrite").saveAsTable("potential_matches")
    matching.step_4_aggregate_features()

    pm_df = matching.spark.table("potential_matches").toPandas()

    assert pm_df.shape == (30, 21)
    assert (
        pm_df.query(
            "namelast_clean_a == 'cridlebaugh' and histid_b == '001B8A74-3795-4997-BC5B-2A07257668F9'"
        )["exact"].iloc[0]
        == 1
    )
    assert (
        pm_df.query(
            "namelast_clean_a == 'cridlebaugh' and histid_b == '001B8A74-3795-4997-BC5B-2A07257668F9'"
        )["exact_all"].iloc[0]
        == 1
    )
    assert (
        pm_df.query(
            "namelast_clean_a == 'cridlebaugh' and histid_b == '001B8A74-3795-4997-BC5B-2A07257668F9'"
        )["hits"].iloc[0]
        == 3
    )
    assert (
        pm_df.query(
            "namelast_clean_a == 'cridlebaugh' and histid_b == '001B8A74-3795-4997-BC5B-2A07257668F9'"
        )["hits2"].iloc[0]
        == 9
    )
    assert (
        pm_df.query(
            "namelast_clean_a == 'cridlebaugh' and histid_b == '001B8A74-3795-4997-BC5B-2A07257668F9'"
        )["exact_mult"].iloc[0]
        == 3
    )
    assert (
        pm_df.query(
            "namelast_clean_a == 'cridlebaugh' and histid_b == '001B8A74-3795-4997-BC5B-2A07257668F9'"
        )["exact_all_mult"].iloc[0]
        == 3
    )
    assert (
        pm_df.query(
            "namelast_clean_a == 'cridlebaugh' and histid_b == '001B8A74-3795-4997-BC5B-2A07257668F9'"
        )["exact_all_mult2"].iloc[0]
        == 9
    )
