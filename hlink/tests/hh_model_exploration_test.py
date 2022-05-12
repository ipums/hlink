# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pandas as pd


def test_all_hh_mod_ev(
    spark,
    main,
    hh_training_conf,
    hh_integration_test_data,
    hh_model_exploration,
    hh_training_data_path,
):
    """Integration test for hh model eval steps 0, 1, and 2 with two models"""
    path_a, path_b, path_pms = hh_integration_test_data
    hh_model_exploration.spark.read.csv(
        path_a, header=True, inferSchema=True
    ).write.mode("overwrite").saveAsTable("prepped_df_a")
    hh_model_exploration.spark.read.csv(
        path_b, header=True, inferSchema=True
    ).write.mode("overwrite").saveAsTable("prepped_df_b")

    hh_model_exploration.run_step(0)
    hh_model_exploration.run_step(1)
    hh_model_exploration.run_step(2)

    prc = spark.table(
        "hh_model_eval_precision_recall_curve_logistic_regression__"
    ).toPandas()
    assert all(
        elem in list(prc.columns)
        for elem in ["params", "precision", "recall", "threshold_gt_eq"]
    )
    prc_rf = spark.table(
        "hh_model_eval_precision_recall_curve_random_forest__maxdepth___5_0___numtrees___75_0_"
    ).toPandas()
    assert all(
        elem in list(prc_rf.columns)
        for elem in ["params", "precision", "recall", "threshold_gt_eq"]
    )

    tr = spark.table("hh_model_eval_training_results").toPandas()
    assert all(
        elem in list(tr.columns)
        for elem in [
            "model",
            "parameters",
            "alpha_threshold",
            "threshold_ratio",
            "precision_test_mean",
            "precision_test_sd",
            "recall_test_mean",
            "recall_test_sd",
            "mcc_test_sd",
            "mcc_test_mean",
            "precision_train_mean",
            "precision_train_sd",
            "recall_train_mean",
            "recall_train_sd",
            "pr_auc_mean",
            "pr_auc_sd",
            "mcc_train_mean",
            "mcc_train_sd",
            "maxDepth",
            "numTrees",
        ]
    )
    assert tr.__len__() == 2
    assert (
        tr.query("model == 'logistic_regression'")["precision_test_mean"].iloc[0] > 0.9
    )
    assert tr.query("model == 'logistic_regression'")["alpha_threshold"].iloc[0] == 0.5
    assert tr.query("model == 'random_forest'")["maxDepth"].iloc[0] == 5
    assert tr.query("model == 'random_forest'")["pr_auc_mean"].iloc[0] > 0.9
    assert tr.query("model == 'logistic_regression'")["pr_auc_mean"].iloc[0] > 0.9
    assert tr.query("model == 'logistic_regression'")["recall_test_mean"].iloc[0] > 0.9

    preds = spark.table("hh_model_eval_predictions").toPandas()
    assert all(
        elem in list(preds.columns)
        for elem in [
            "histid_a",
            "histid_b",
            "probability_array",
            "probability",
            "second_best_prob",
            "ratio",
            "prediction",
            "match",
        ]
    )
    pm0 = preds.query(
        "histid_a == '790BFA1D-E8A8-4C96-B005-6C36DD5154F0' and histid_b == 'B0BFE02E-DDEE-4DD1-86E3-15F297D18DAE'"
    )
    assert pm0["second_best_prob"].round(2).iloc[0] >= 0.90
    assert pm0["ratio"].round(2).iloc[0] >= 1.01
    assert pm0["prediction"].iloc[0] == 0
    assert pm0["probability"].round(2).iloc[0] >= 0.91

    pred_train = spark.table("hh_model_eval_predict_train").toPandas()
    assert all(
        elem in list(pred_train.columns)
        for elem in [
            "histid_a",
            "histid_b",
            "probability_array",
            "probability",
            "second_best_prob",
            "ratio",
            "prediction",
            "match",
        ]
    )
    pm1 = pred_train.query(
        "histid_a == '014470B1-63E5-4A64-BB9A-70F5A9340130' and histid_b == '4E713690-5206-41FD-ABE1-5F545E55A5BB'"
    )
    assert pm1["match"].iloc[0] == 1
    assert pm1["probability"].iloc[0] > 0.9
    assert pm1["second_best_prob"].iloc[0] < 0.2
    assert pd.isnull(pm1["ratio"].iloc[0])
    assert pm1["prediction"].iloc[0] == 1

    main.do_drop_all("")
