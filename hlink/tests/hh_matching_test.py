# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from hlink.linking.matching.link_step_score import LinkStepScore
from hlink.tests.conftest import load_table_from_csv


def test_step_0_filter_and_pair(spark, hh_matching_stubs, hh_matching, conf):
    """Test hh_matching step 0 to make sure hh_blocked_matches is created correctly"""

    conf["id_column"] = "histid"
    conf["hh_training"] = {"prediction_col": "prediction"}
    path_a, path_b, path_matches, path_pred_matches = hh_matching_stubs

    load_table_from_csv(hh_matching, path_a, "prepped_df_a")
    load_table_from_csv(hh_matching, path_b, "prepped_df_b")
    load_table_from_csv(hh_matching, path_matches, "scored_potential_matches")
    load_table_from_csv(hh_matching, path_pred_matches, "predicted_matches")

    hh_matching.run_step(0)

    # Create pandas DFs of the step_0 blocked matches table
    blocked_matches_hh_df = spark.table("hh_blocked_matches").toPandas()

    # Make assertions on the data
    assert blocked_matches_hh_df.shape[0] == 9
    assert blocked_matches_hh_df.query("serialp_a == 1").shape == (9, 4)
    assert blocked_matches_hh_df.query("serialp_b == 8").shape == (6, 4)
    assert blocked_matches_hh_df.query("serialp_b == 7").shape == (3, 4)


def test_household_matching_training_integration(
    spark, hh_training, hh_matching, hh_training_conf, hh_integration_test_data
):
    """Test all hh_training and hh_matching steps to ensure they work as a pipeline"""
    path_a, path_b, path_pms = hh_integration_test_data

    load_table_from_csv(hh_matching, path_a, "prepped_df_a")
    load_table_from_csv(hh_matching, path_b, "prepped_df_b")
    load_table_from_csv(hh_matching, path_pms, "predicted_matches")

    hh_training.run_step(0)
    hh_training.run_step(1)

    assert spark.table("hh_training_data").toPandas().shape == (198, 60)
    hhtf = spark.table("hh_training_features").toPandas()
    assert hhtf.shape == (10, 10)
    assert all(
        elem in list(hhtf.columns)
        for elem in [
            "histid_a",
            "histid_b",
            "serialp_a",
            "serialp_b",
            "match",
            "byrdiff",
            "ssex",
            "srelate",
            "namefrst_jw",
            "namelast_jw",
        ]
    )

    hh_training.run_step(2)

    hh_matching.run_step(0)

    assert spark.table("indiv_matches").count() == 20
    assert spark.table("unmatched_a").count() == 80
    assert spark.table("unmatched_b").count() == 89
    assert spark.table("hh_blocked_matches").count() == 247

    hh_matching.run_step(1)
    hh_matching.run_step(2)

    assert spark.table("hh_potential_matches_prepped").toPandas().shape == (247, 9)
    hhspms = spark.table("hh_scored_potential_matches").toPandas()
    assert all(
        elem in list(hhspms.columns)
        for elem in [
            "serialp_a",
            "serialp_b",
            "histid_a",
            "histid_b",
            "byrdiff",
            "srelate",
            "namelast_jw",
            "namefrst_jw",
            "ssex",
            "byrdiff_imp",
            "namefrst_jw_imp",
            "namelast_jw_imp",
            "ssex_onehotencoded",
            "srelate_onehotencoded",
            "features_vector",
            "rawPrediction",
            "probability_array",
            "probability",
            "prediction",
        ]
    )

    # Nancy Brummit (A) does match Nancy Brummitt (B)
    assert (
        hhspms.query(
            "histid_a == 'C2A27CFE-5E4C-47A7-9C48-CB07FC64E49F' and histid_b == '34EF9DE2-CFEF-492D-BE92-26865C3CEE84'"
        )["prediction"].iloc[0]
        == 1
    )

    # Sam Brummit (A) does not match Nancy Brummitt (B)
    assert (
        hhspms.query(
            "histid_a == 'A733F5D3-38F7-4D61-8F02-9DB4BD2D64E0' and histid_b == '34EF9DE2-CFEF-492D-BE92-26865C3CEE84'"
        )["prediction"].iloc[0]
        == 0
    )

    # Margaret Kenefick (A) does match Margaret Kenefick (B)
    assert (
        hhspms.query(
            "histid_a == '006CA56D-1620-459E-9EB1-DCCE5E08949E' and histid_b == '2670520B-6B0A-4B2A-8A04-B876BB5A707C'"
        )["prediction"].iloc[0]
        == 1
    )

    # Margaret Kenefick (A) does not match William Kenefick (B)
    assert (
        hhspms.query(
            "histid_a == '006CA56D-1620-459E-9EB1-DCCE5E08949E' and histid_b == '68977EB4-AE65-4D6F-9AD8-144AFA6FBF27'"
        )["prediction"].iloc[0]
        == 0
    )


def test_hh_agg_features(
    spark, hh_agg_features_test_data, hh_matching, hh_agg_feat_conf
):
    """Ensure proper creation of aggregate features on hh potential matches"""

    path_a, path_b, path_pms = hh_agg_features_test_data

    load_table_from_csv(hh_matching, path_a, "prepped_df_a")
    load_table_from_csv(hh_matching, path_b, "prepped_df_b")
    load_table_from_csv(hh_matching, path_pms, "hh_potential_matches")

    hh_matching.training_conf = "hh_training"
    hh_matching.table_prefix = "hh_"

    LinkStepScore(hh_matching)._create_features(hh_agg_feat_conf)

    pm = spark.table("hh_potential_matches_prepped").toPandas()

    # Make assertions on the data
    assert pm.query(
        "histid_a == 'B4DFC0CB-205F-4087-B95D-81992AFBBF0E' and histid_b == '0A9BDD32-CF94-4E60-ACE1-D2745C305795'"
    )["mardurmatch"].iloc[0]
    assert (
        round(
            pm.query(
                "histid_a == 'B4DFC0CB-205F-4087-B95D-81992AFBBF0E' and histid_b == '0A9BDD32-CF94-4E60-ACE1-D2745C305795'"
            )["jw_max_a"].iloc[0],
            2,
        )
        == 0.46
    )
    assert (
        pm.query(
            "histid_a == 'B4DFC0CB-205F-4087-B95D-81992AFBBF0E' and histid_b == '0A9BDD32-CF94-4E60-ACE1-D2745C305795'"
        )["jw_max_b"].iloc[0]
        == 0
    )
    assert (
        pm.query(
            "histid_a == 'B4DFC0CB-205F-4087-B95D-81992AFBBF0E' and histid_b == '0A9BDD32-CF94-4E60-ACE1-D2745C305795'"
        )["f1_match"].iloc[0]
        == 1
    )
    assert (
        pm.query(
            "histid_a == 'B4DFC0CB-205F-4087-B95D-81992AFBBF0E' and histid_b == '0A9BDD32-CF94-4E60-ACE1-D2745C305795'"
        )["f2_match"].iloc[0]
        == 1
    )

    assert (
        pm.query(
            "histid_a == '6244C5B1-DCB6-47F2-992E-A408225C2AE2' and histid_b == '625DA33B-0623-4060-87E7-2F542C9B5524'"
        )["jw_max_a"].iloc[0]
        == 0
    )
    assert (
        pm.query(
            "histid_a == '6244C5B1-DCB6-47F2-992E-A408225C2AE2' and histid_b == '625DA33B-0623-4060-87E7-2F542C9B5524'"
        )["jw_max_b"].iloc[0]
        == 0
    )
    assert (
        pm.query(
            "histid_a == '6244C5B1-DCB6-47F2-992E-A408225C2AE2' and histid_b == '625DA33B-0623-4060-87E7-2F542C9B5524'"
        )["f1_match"].iloc[0]
        == 1
    )
    assert (
        pm.query(
            "histid_a == '6244C5B1-DCB6-47F2-992E-A408225C2AE2' and histid_b == '625DA33B-0623-4060-87E7-2F542C9B5524'"
        )["f2_match"].iloc[0]
        == 2
    )

    assert (
        pm.query(
            "histid_a == '709916FD-D95D-4D22-B5C0-0C3ADBF88EEC' and histid_b == '51342DAE-AC53-4605-8DD9-FC5E94C235F8'"
        )["jw_max_a"].iloc[0]
        == 1
    )
    assert (
        pm.query(
            "histid_a == '709916FD-D95D-4D22-B5C0-0C3ADBF88EEC' and histid_b == '51342DAE-AC53-4605-8DD9-FC5E94C235F8'"
        )["jw_max_b"].iloc[0]
        == 1
    )
    assert (
        pm.query(
            "histid_a == '709916FD-D95D-4D22-B5C0-0C3ADBF88EEC' and histid_b == '51342DAE-AC53-4605-8DD9-FC5E94C235F8'"
        )["f1_match"].iloc[0]
        == 2
    )
    assert (
        pm.query(
            "histid_a == '709916FD-D95D-4D22-B5C0-0C3ADBF88EEC' and histid_b == '51342DAE-AC53-4605-8DD9-FC5E94C235F8'"
        )["f2_match"].iloc[0]
        == 0
    )
    assert not pm.query(
        "histid_a == '709916FD-D95D-4D22-B5C0-0C3ADBF88EEC' and histid_b == '51342DAE-AC53-4605-8DD9-FC5E94C235F8'"
    )["mardurmatch"].iloc[0]

    assert (
        pm.query(
            "histid_a == '43C3C7F5-39E2-461D-B4F1-A0C5EA1750A4' and histid_b == '99CF8208-1F3D-4B62-80A4-C95FDD5D41F2'"
        )["jw_max_a"].iloc[0]
        == 1
    )
    assert (
        pm.query(
            "histid_a == '43C3C7F5-39E2-461D-B4F1-A0C5EA1750A4' and histid_b == '99CF8208-1F3D-4B62-80A4-C95FDD5D41F2'"
        )["jw_max_b"].iloc[0]
        == 1
    )
    assert (
        pm.query(
            "histid_a == '43C3C7F5-39E2-461D-B4F1-A0C5EA1750A4' and histid_b == '99CF8208-1F3D-4B62-80A4-C95FDD5D41F2'"
        )["f1_match"].iloc[0]
        == 2
    )
    assert (
        pm.query(
            "histid_a == '43C3C7F5-39E2-461D-B4F1-A0C5EA1750A4' and histid_b == '99CF8208-1F3D-4B62-80A4-C95FDD5D41F2'"
        )["f2_match"].iloc[0]
        == 1
    )


def test_step_0_1_hh_blocking_and_filtering(
    spark, hh_matching_stubs, hh_matching, conf
):
    """Test hh post-blocking filter works on hh_blocked_matches using a comparison feature"""

    conf["id_column"] = "histid"
    conf["hh_training"] = {"prediction_col": "prediction"}
    conf["hh_comparisons"] = {
        "comparison_type": "threshold",
        "feature_name": "agediff",
        "threshold_expr": "<= 10",
    }
    conf["comparison_features"] = [
        {"alias": "agediff", "column_name": "birthyr", "comparison_type": "abs_diff"}
    ]

    path_a, path_b, path_matches, path_pred_matches = hh_matching_stubs

    load_table_from_csv(hh_matching, path_a, "prepped_df_a")
    load_table_from_csv(hh_matching, path_b, "prepped_df_b")
    load_table_from_csv(hh_matching, path_matches, "scored_potential_matches")
    load_table_from_csv(hh_matching, path_pred_matches, "predicted_matches")

    hh_matching.run_step(0)
    hh_matching.run_step(1)

    # Create pandas DFs of the step_2 potential matches table
    blocked_matches_hh_df = spark.table("hh_blocked_matches").toPandas()
    potential_matches_hh_df = spark.table("hh_potential_matches").toPandas()

    # Make assertions on the data
    assert blocked_matches_hh_df.shape[0] == 9
    assert blocked_matches_hh_df.query("serialp_a == 1").shape == (9, 4)
    assert blocked_matches_hh_df.query("serialp_b == 8").shape == (6, 4)
    assert blocked_matches_hh_df.query("serialp_b == 7").shape == (3, 4)

    assert potential_matches_hh_df.shape[0] == 3
    assert potential_matches_hh_df.query("histid_a == '1004A'").shape[0] == 2
    assert potential_matches_hh_df.query("histid_a == '1005A'").shape[0] == 1

    assert all(elem <= 10 for elem in list(potential_matches_hh_df["agediff"]))
