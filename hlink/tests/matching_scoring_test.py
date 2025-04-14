# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pandas as pd
import hlink.linking.core.threshold as threshold_core
from hlink.linking.matching.link_step_score import LinkStepScore


def test_step_2_skip_on_no_conf(spark, matching_conf, matching, capsys):
    """Test matching step 2 doesn't run if no training config"""

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


def test_step_2_alpha_beta_thresholds(
    spark, matching, matching_conf, threshold_ratio_data_path_2
):
    """Test matching step 2 with both probability and ratio thresholds"""

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
        matching_conf["id_column"],
        matching_conf["training"].get("decision"),
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
    assert pd.notnull(
        tp.query("histid_a == '6a' and histid_b == '9b'")["ratio"].iloc[0]
    )
    assert pd.isnull(tp.query("histid_a == '6a' and histid_b == '8b'")["ratio"].iloc[0])

    assert "4b" not in list(pm["histid_b"])
    assert "10b" not in list(pm["histid_b"])
    assert "0a" not in list(pm["histid_a"])

    assert tp.query("histid_a == '5a' and histid_b == '7b'")["prediction"].iloc[0] == 1
    assert tp.query("histid_a == '5a' and histid_b == '6b'")["prediction"].iloc[0] == 0


def test_step_2_aggregate_features(
    spark, matching_conf, matching, agg_features_datasources
):
    matching_conf["id_column"] = "histid"
    matching_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "exact",
            "column_names": ["namefrst_unstd", "namelast_clean"],
            "comparison_type": "all_equals",
        },
        {
            "alias": "exact_all",
            "column_names": ["namefrst_unstd", "namelast_clean", "bpl"],
            "comparison_type": "all_equals",
        },
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
        ],
        "chosen_model": {
            "type": "probit",
            "threshold": 0.5,
        },
        "dependent_var": "match",
    }

    potential_matches_path, prepped_df_a_path, prepped_df_b_path = (
        agg_features_datasources
    )
    spark.read.csv(potential_matches_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("potential_matches")

    spark.read.csv(prepped_df_a_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_a")
    spark.read.csv(prepped_df_b_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_b")

    link_step_score = LinkStepScore(matching)
    link_step_score._create_features(matching_conf)

    pm_prepped = spark.table("potential_matches_prepped").toPandas()

    filtered = pm_prepped.query(
        "histid_a == '0202928A-AC3E-48BB-8568-3372067F35C7' and histid_b == '001B8A74-3795-4997-BC5B-2A07257668F9'"
    )

    assert filtered["exact"].item()
    assert filtered["exact_all"].item()
    assert filtered["hits"].item() == 3
    assert filtered["hits2"].item() == 9
    assert filtered["exact_mult"].item()
    assert filtered["exact_all_mult"].item() == 3
    assert filtered["exact_all_mult2"].item() == 9
