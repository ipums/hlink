# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import hlink.tests
import pandas as pd
import pytest
import hlink.linking.core.threshold as threshold_core
from hlink.linking.matching.link_step_score import LinkStepScore


@pytest.mark.skip(
    reason="We still want to test that whatever 'secondary_threshold' became is being applied correctly, but we need to refactor this test to account for the fact that this was totally renamed and is now being carried out in a different step (step 3 doesn't exist anymore)."
)
def test_step_3_uniq_and_secondary_threshold(spark, matching_conf, matching):
    """Test a secondary threshold with uniqueness"""
    matching_conf["comparison_features"] = [
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
    ]

    matching_conf["comparisons"] = {
        "comp_a": {
            "feature_name": "namefrst_jw",
            "threshold": 0.8,
            "comparison_type": "threshold",
        },
        "comp_b": {
            "feature_name": "namelast_jw",
            "comparison_type": "threshold",
            "threshold": 0.8,
        },
        "operator": "AND",
    }

    matching_conf["secondary_threshold"] = {
        "threshold_a": {
            "feature_name": "namefrst_jw",
            "comparison_type": "threshold",
            "threshold": 0.9,
        },
        "threshold_b": {
            "feature_name": "namelast_jw",
            "comparison_type": "threshold",
            "threshold": 0.9,
        },
        "unique_true": {"id_a": "id_a", "id_b": "id_b"},
        "operator": "AND",
        "secondary": True,
    }

    matching.step_0_explode()
    matching.step_1_match()
    hlink.linking.matching._step_2_score.__create_features(matching, matching_conf)

    # Create pandas DFs of the step_2 potential matches table
    potential_matches_df = spark.table("potential_matches_prepped").toPandas()

    #    matching.step_3_secondary_threshold()
    # unique_matches_df = spark.table("potential_matches").toPandas()
    unique_high_matches_df = spark.table("potential_matches_prepped").toPandas()

    assert len(potential_matches_df.id_a) == 5
    # assert (len(unique_matches_df.id_a) == 1)
    # assert (unique_matches_df.query("id_a == 10 and id_b == 10")["namelast_jw"].iloc[0] > 0.8)
    # assert (unique_matches_df.query("id_a == 10 and id_b == 10")["namelast_jw"].iloc[0] < 0.9)
    # assert (unique_matches_df.query("id_a == 10 and id_b == 10")["namefrst_jw"].iloc[0] > 0.8)
    # assert (unique_matches_df.query("id_a == 10 and id_b == 10")["namefrst_jw"].iloc[0] > 0.9)
    assert unique_high_matches_df.empty


# TODO: is there a step 3 anymore?
def test_step_3_skip_on_no_conf(spark, matching_conf, matching, capsys):
    """Test matching step 3 doesn't run if no training config"""

    matching_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        }
    ]

    matching.run_step(0)
    matching.run_step(1)
    matching.run_step(2)

    captured = capsys.readouterr()

    assert (
        "WARNING: Skipping step 'score'. Your config file either does not contain a 'training' section or a 'chosen_model' section within the 'training' section."
        in captured.out
    )


# TODO: is there a step 3 any more?
def test_step_3_alpha_beta_thresholds(
    spark, matching, matching_conf, threshold_ratio_data_path_2
):
    """Test matching step 3 with both probability and ratio thresholds"""

    matching.spark.read.csv(
        threshold_ratio_data_path_2, header=True, inferSchema=True
    ).write.mode("overwrite").saveAsTable("score_tmp")
    score_tmp = matching.spark.table("score_tmp")

    matching_conf["id_column"] = "histid"
    matching_conf["training"]["decision"] = "drop_duplicate_with_threshold_ratio"
    matching_conf["drop_data_from_scored_matches"] = True
    threshold_ratio = 1.0
    alpha_threshold = 0.5

    predictions = threshold_core.predict_using_thresholds(
        score_tmp,
        alpha_threshold,
        threshold_ratio,
        matching_conf["training"],
        matching_conf["id_column"],
    )
    predictions.write.mode("overwrite").saveAsTable("predictions")

    link_step_score = LinkStepScore(matching)
    link_step_score._save_table_with_requested_columns(
        "pm", "pmp", predictions, "histid_a", "histid_b"
    )
    link_step_score._save_predicted_matches(matching_conf, "histid_a", "histid_b")

    tp = predictions.toPandas()
    pm = matching.spark.table("predicted_matches").toPandas()

    assert sorted(tp.columns) == [
        "histid_a",
        "histid_b",
        "prediction",
        "probability",
        "ratio",
        "second_best_prob",
    ]
    assert sorted(pm.columns) == ["histid_a", "histid_b", "prediction", "probability"]
    assert tp["prediction"].sum() == 5
    assert pm["prediction"].sum() == 3
    assert pd.isnull(tp.query("histid_a == '0a' and histid_b == '1b'")["ratio"].iloc[0])
    assert pd.notnull(
        tp.query("histid_a == '0a' and histid_b == '1b'")["second_best_prob"].iloc[0]
    )
    assert tp.query("histid_a == '0a' and histid_b == '1b'")["prediction"].iloc[0] == 0
    assert pd.notnull(
        tp.query("histid_a == '0a' and histid_b == '0b'")["second_best_prob"].iloc[0]
    )
    assert tp.query("histid_a == '0a' and histid_b == '0b'")["prediction"].iloc[0] == 0
    assert tp.query("histid_a == '1a' and histid_b == '3b'")["prediction"].iloc[0] == 1
    assert tp.query("histid_a == '2a' and histid_b == '4b'")["prediction"].iloc[0] == 1
    assert tp.query("histid_a == '3a' and histid_b == '4b'")["prediction"].iloc[0] == 1

    assert tp.query("histid_a == '6a' and histid_b == '9b'")["prediction"].iloc[0] == 0
    assert pd.isnull(tp.query("histid_a == '6a' and histid_b == '9b'")["ratio"].iloc[0])

    assert "4b" not in list(pm["histid_b"])
    assert "10b" not in list(pm["histid_b"])
    assert "0a" not in list(pm["histid_a"])

    assert tp.query("histid_a == '5a' and histid_b == '7b'")["prediction"].iloc[0] == 1
    assert tp.query("histid_a == '5a' and histid_b == '6b'")["prediction"].iloc[0] == 0
