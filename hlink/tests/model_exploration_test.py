# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pytest
import pandas as pd

import hlink.linking.core.threshold as threshold_core
from hlink.linking.model_exploration.link_step_train_test_models import (
    LinkStepTrainTestModels,
)


def test_all(
    spark,
    main,
    training_conf,
    model_exploration,
    state_dist_path,
    training_data_doubled_path,
):
    """Test training step 2 with probit model"""
    training_conf["comparison_features"] = [
        {
            "alias": "regionf",
            "column_name": "region",
            "comparison_type": "fetch_a",
            "categorical": True,
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "state_distance",
            "key_count": 1,
            "column_name": "bpl",
            "comparison_type": "geo_distance",
            "loc_a": "statecode1",
            "loc_b": "statecode2",
            "distance_col": "dist",
            "table_name": "state_distances_lookup",
            "distances_file": state_dist_path,
        },
    ]

    training_conf["training"]["dataset"] = training_data_doubled_path
    training_conf["training"]["dependent_var"] = "match"
    training_conf["training"]["use_training_data_features"] = False
    training_conf["training"]["decision"] = "drop_duplicate_with_threshold_ratio"
    training_conf["training"]["n_training_iterations"] = 4
    training_conf["training"]["seed"] = 120
    training_conf["training"]["independent_vars"] = [
        "namelast_jw",
        "regionf",
        "state_distance",
    ]
    training_conf["training"]["model_parameters"] = [
        {"type": "probit", "threshold": 0.8, "threshold_ratio": [1.01, 1.3]},
        {
            "type": "random_forest",
            "maxDepth": 5.0,
            "numTrees": 75.0,
            "threshold_ratio": 1.2,
            "threshold": 0.2,
        },
    ]
    training_conf["training"]["get_precision_recall_curve"] = True

    model_exploration.run_step(0)
    model_exploration.run_step(1)
    model_exploration.run_step(2)

    prc = spark.table("model_eval_precision_recall_curve_probit__").toPandas()
    assert all(
        elem in list(prc.columns)
        for elem in ["params", "precision", "recall", "threshold_gt_eq"]
    )
    prc_rf = spark.table(
        "model_eval_precision_recall_curve_random_forest__maxdepth___5_0___numtrees___75_0_"
    ).toPandas()
    assert all(
        elem in list(prc_rf.columns)
        for elem in ["params", "precision", "recall", "threshold_gt_eq"]
    )

    tr = spark.table("model_eval_training_results").toPandas()

    assert tr.__len__() == 3
    assert tr.query("threshold_ratio == 1.01")["precision_test_mean"].iloc[0] >= 0.5
    assert tr.query("threshold_ratio == 1.3")["alpha_threshold"].iloc[0] == 0.8
    assert tr.query("model == 'random_forest'")["maxDepth"].iloc[0] == 5
    assert tr.query("model == 'random_forest'")["pr_auc_mean"].iloc[0] > 0.8
    assert (
        tr.query("threshold_ratio == 1.01")["pr_auc_mean"].iloc[0]
        == tr.query("threshold_ratio == 1.3")["pr_auc_mean"].iloc[0]
    )

    preds = spark.table("model_eval_predictions").toPandas()
    assert (
        preds.query("id_a == 20 and id_b == 30")["second_best_prob"].round(2).iloc[0]
        >= 0.6
    )
    assert (
        preds.query("id_a == 20 and id_b == 30")["probability"].round(2).iloc[0] > 0.5
    )
    assert preds.query("id_a == 30 and id_b == 30")["prediction"].iloc[0] == 0
    assert pd.isnull(
        preds.query("id_a == 10 and id_b == 30")["second_best_prob"].iloc[0]
    )

    pred_train = spark.table("model_eval_predict_train").toPandas()
    assert pred_train.query("id_a == 20 and id_b == 50")["match"].iloc[0] == 0
    assert pd.isnull(
        pred_train.query("id_a == 10 and id_b == 50")["second_best_prob"].iloc[1]
    )
    assert pred_train.query("id_a == 20 and id_b == 50")["prediction"].iloc[1] == 1

    main.do_drop_all("")


def test_step_2_param_grid(spark, main, training_conf, model_exploration, fake_self):
    """Test matching step 2 training to see if the custom param grid builder is working"""

    training_conf["training"]["model_parameters"] = [
        {"type": "random_forest", "maxDepth": [3, 4, 5], "numTrees": [50, 100]},
        {"type": "probit", "threshold": [0.5, 0.7]},
    ]

    link_step = LinkStepTrainTestModels(model_exploration)
    param_grid = link_step._custom_param_grid_builder(training_conf)

    expected = [
        {"maxDepth": 3, "numTrees": 50, "type": "random_forest"},
        {"maxDepth": 3, "numTrees": 100, "type": "random_forest"},
        {"maxDepth": 4, "numTrees": 50, "type": "random_forest"},
        {"maxDepth": 4, "numTrees": 100, "type": "random_forest"},
        {"maxDepth": 5, "numTrees": 50, "type": "random_forest"},
        {"maxDepth": 5, "numTrees": 100, "type": "random_forest"},
        {"type": "probit", "threshold": [0.5, 0.7]},
    ]

    assert len(param_grid) == len(expected)
    assert all([m in expected for m in param_grid])

    main.do_drop_all("")


# -------------------------------------
# Tests that probably should be moved
# -------------------------------------


@pytest.fixture(scope="function")
def feature_conf(training_conf):
    training_conf["comparison_features"] = [
        {
            "alias": "regionf",
            "column_name": "region",
            "comparison_type": "fetch_a",
            "categorical": True,
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
    ]

    training_conf["training"]["independent_vars"] = ["namelast_jw", "regionf"]

    training_conf["training"]["model_parameters"] = []
    training_conf["training"]["n_training_iterations"] = 2
    return training_conf


def test_step_2_probability_ratio_threshold(
    spark, main, feature_conf, model_exploration, threshold_ratio_data_path
):
    """Test probability threshold ratio decision boundary to remove too close multi-matches"""
    feature_conf["id_column"] = "histid"
    feature_conf["training"]["dataset"] = threshold_ratio_data_path
    feature_conf["training"]["decision"] = "drop_duplicate_with_threshold_ratio"
    threshold_ratio = 1.2
    alpha_threshold = 0.5

    model_exploration.run_step(0)
    predictions = spark.table("model_eval_training_data")
    threshold_predictions = threshold_core._apply_threshold_ratio(
        predictions.drop("prediction"),
        alpha_threshold,
        threshold_ratio,
        feature_conf["id_column"],
    )
    tp = threshold_predictions.toPandas()
    assert sorted(tp.columns) == [
        "histid_a",
        "histid_b",
        "prediction",
        "probability",
        "ratio",
        "second_best_prob",
    ]
    assert tp["prediction"].sum() == 4
    assert pd.isnull(
        tp.query("histid_a == 6 and histid_b == 6")["second_best_prob"].iloc[0]
    )
    assert tp.query("histid_a == 6 and histid_b == 6")["prediction"].iloc[0] == 1
    assert pd.isnull(
        tp.query("histid_a == 7 and histid_b == 7")["second_best_prob"].iloc[0]
    )
    assert tp.query("histid_a == 7 and histid_b == 7")["prediction"].iloc[0] == 0

    assert tp.query("histid_a == 2 and histid_b == 2")["prediction"].iloc[0] == 1
    assert tp.query("histid_a == 1 and histid_b == 1")["prediction"].iloc[0] == 1
    assert tp.query("histid_a == 1 and histid_b == 0")["prediction"].iloc[0] == 0
    assert tp.query("histid_a == 0 and histid_b == 0")["prediction"].iloc[0] == 0
    assert tp.query("histid_a == 0 and histid_b == 0")["ratio"].iloc[0] > 1
    assert pd.isnull(tp.query("histid_a == 0 and histid_b == 1")["ratio"].iloc[0])


def test_step_1_OneHotEncoding(
    spark, feature_conf, model_exploration, state_dist_path, training_data_path
):
    """Test matching step 2 training to see if the OneHotEncoding is working"""

    model_exploration.run_step(0)
    model_exploration.run_step(1)

    training_v = spark.table("model_eval_training_vectorized").toPandas()
    columns_expected = [
        "match",
        "id_a",
        "id_b",
        "namelast_jw",
        "regionf",
        "namelast_jw_imp",
        "regionf_onehotencoded",
        "features_vector",
    ]
    assert training_v.shape[0] == 9
    assert all([c in training_v.columns for c in columns_expected])
    assert len(training_v["features_vector"][0]) == 5


def test_step_2_scale_values(
    spark, feature_conf, model_exploration, state_dist_path, training_data_path
):
    feature_conf["training"]["scale_data"] = True

    model_exploration.run_step(0)
    model_exploration.run_step(1)

    training_v = spark.table("model_eval_training_vectorized").toPandas()

    assert training_v.shape == (9, 9)
    assert len(training_v["features_vector"][0]) == 5
    assert training_v["features_vector"][0][0].round(2) == 2.85


def test_step_2_train_random_forest_spark(
    spark, main, feature_conf, model_exploration, state_dist_path
):
    """Test training step 2 with random forest model"""
    feature_conf["training"]["model_parameters"] = [
        {
            "type": "random_forest",
            "maxDepth": 3,
            "numTrees": 3,
            "featureSubsetStrategy": "sqrt",
        }
    ]
    feature_conf["training"]["output_suspicious_TD"] = True
    feature_conf["training"]["n_training_iterations"] = 10

    model_exploration.run_step(0)
    model_exploration.run_step(1)
    model_exploration.run_step(2)

    tr = spark.table("model_eval_training_results").toPandas()
    # assert tr.shape == (1, 18)
    assert tr.query("model == 'random_forest'")["pr_auc_mean"].iloc[0] > 0.7
    assert tr.query("model == 'random_forest'")["maxDepth"].iloc[0] == 3

    FNs = spark.table("model_eval_repeat_fns").toPandas()
    assert FNs.shape == (3, 4)
    assert FNs.query("id_a == 30")["count"].iloc[0] > 5

    TPs = spark.table("model_eval_repeat_tps").toPandas()
    assert TPs.shape == (2, 4)

    TNs = spark.table("model_eval_repeat_tns").toPandas()
    assert TNs.shape == (6, 4)

    main.do_drop_all("")


def test_step_2_train_logistic_regression_spark(
    spark, main, feature_conf, model_exploration, state_dist_path, training_data_path
):
    """Test training step 2 with logistic regression model"""
    feature_conf["training"]["model_parameters"] = [
        {"type": "logistic_regression", "threshold": 0.7}
    ]

    model_exploration.run_step(0)
    model_exploration.run_step(1)
    model_exploration.run_step(2)

    tr = spark.table("model_eval_training_results").toPandas()

    # assert tr.shape == (1, 16)
    assert tr.query("model == 'logistic_regression'")["pr_auc_mean"].iloc[0] == 0.8125
    assert (
        round(tr.query("model == 'logistic_regression'")["alpha_threshold"].iloc[0], 1)
        == 0.7
    )
    main.do_drop_all("")


def test_step_2_train_decision_tree_spark(
    spark, main, feature_conf, model_exploration, state_dist_path, training_data_path
):
    """Test training step 2 with decision tree model"""
    feature_conf["training"]["model_parameters"] = [
        {"type": "decision_tree", "maxDepth": 3, "minInstancesPerNode": 1, "maxBins": 7}
    ]

    model_exploration.run_step(0)
    model_exploration.run_step(1)
    model_exploration.run_step(2)

    tr = spark.table("model_eval_training_results").toPandas()

    # assert tr.shape == (1, 18)
    assert tr.query("model == 'decision_tree'")["precision_test_mean"].iloc[0] > 0
    assert tr.query("model == 'decision_tree'")["maxDepth"].iloc[0] == 3
    assert tr.query("model == 'decision_tree'")["minInstancesPerNode"].iloc[0] == 1
    assert tr.query("model == 'decision_tree'")["maxBins"].iloc[0] == 7

    main.do_drop_all("")


def test_step_2_train_gradient_boosted_trees_spark(
    spark, main, feature_conf, model_exploration, state_dist_path, training_data_path
):
    """Test training step 2 with gradient boosted tree model"""
    feature_conf["training"]["model_parameters"] = [
        {
            "type": "gradient_boosted_trees",
            "maxDepth": 5,
            "minInstancesPerNode": 1,
            "maxBins": 5,
        }
    ]

    model_exploration.run_step(0)
    model_exploration.run_step(1)
    model_exploration.run_step(2)

    tr = spark.table("model_eval_training_results").toPandas()
    preds = spark.table("model_eval_predictions").toPandas()

    assert "probability_array" in list(preds.columns)

    # assert tr.shape == (1, 18)
    assert (
        tr.query("model == 'gradient_boosted_trees'")["precision_test_mean"].iloc[0] > 0
    )
    assert tr.query("model == 'gradient_boosted_trees'")["maxDepth"].iloc[0] == 5
    assert (
        tr.query("model == 'gradient_boosted_trees'")["minInstancesPerNode"].iloc[0]
        == 1
    )
    assert tr.query("model == 'gradient_boosted_trees'")["maxBins"].iloc[0] == 5

    main.do_drop_all("")


def test_step_2_interact_categorial_vars(
    spark, training_conf, model_exploration, state_dist_path, training_data_path
):
    """Test matching step 2 training to see if the OneHotEncoding is working"""

    training_conf["comparison_features"] = [
        {
            "alias": "regionf",
            "column_name": "region",
            "comparison_type": "fetch_a",
            "categorical": True,
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "td_match",
            "column_name": "match",
            "comparison_type": "fetch_td",
            "categorical": True,
        },
    ]

    training_conf["pipeline_features"] = [
        {
            "input_columns": ["regionf", "td_match"],
            "output_column": "regionf_interacted_tdmatch",
            "transformer_type": "interaction",
        }
    ]

    training_conf["training"]["dependent_var"] = "match"
    training_conf["training"]["independent_vars"] = [
        "namelast_jw",
        "regionf",
        "regionf_interacted_tdmatch",
    ]

    model_exploration.run_step(0)
    model_exploration.run_step(1)

    prepped_data = spark.table("model_eval_training_vectorized").toPandas()

    assert prepped_data.shape == (9, 11)
    assert list(
        prepped_data.query("id_a == 10 and id_b == 10")["regionf_onehotencoded"].iloc[0]
    ) == [0, 1, 0, 0]
    assert list(
        prepped_data.query("id_a == 20 and id_b == 50")["regionf_onehotencoded"].iloc[0]
    ) == [0, 0, 1, 0]
    assert list(
        prepped_data.query("id_a == 10 and id_b == 10")["td_match_onehotencoded"].iloc[
            0
        ]
    ) == [0, 1, 0]
    assert list(
        prepped_data.query("id_a == 20 and id_b == 50")["td_match_onehotencoded"].iloc[
            0
        ]
    ) == [1, 0, 0]
    assert list(
        prepped_data.query("id_a == 10 and id_b == 50")[
            "regionf_interacted_tdmatch"
        ].iloc[0]
    ) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert list(
        prepped_data.query("id_a == 10 and id_b == 10")[
            "regionf_interacted_tdmatch"
        ].iloc[0]
    ) == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    assert list(
        prepped_data.query("id_a == 30 and id_b == 50")[
            "regionf_interacted_tdmatch"
        ].iloc[0]
    ) == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    assert (
        len(
            list(
                prepped_data.query("id_a == 10 and id_b == 10")["features_vector"].iloc[
                    0
                ]
            )
        )
        == 17
    )


def test_step_2_VectorAssembly(
    spark, main, training_conf, model_exploration, state_dist_path, training_data_path
):
    """Test training step 1 training to see if the OneHotEncoding is working"""
    training_conf["comparison_features"] = [
        {
            "alias": "regionf",
            "column_name": "region",
            "comparison_type": "fetch_a",
            "categorical": True,
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "state_distance",
            "column_name": "bpl",
            "key_count": 1,
            "comparison_type": "geo_distance",
            "loc_a": "statecode1",
            "loc_b": "statecode2",
            "distance_col": "dist",
            "table_name": "state_distances_lookup",
            "distances_file": state_dist_path,
        },
    ]

    training_conf["training"]["independent_vars"] = [
        "namelast_jw",
        "regionf",
        "state_distance",
    ]

    training_conf["training"]["model_parameters"] = []

    model_exploration.run_step(0)
    model_exploration.run_step(1)

    vdf = spark.table("model_eval_training_vectorized").toPandas()

    assert len(vdf.query("id_a == 20 and id_b == 30")["features_vector"].iloc[0]) == 6
    assert 3187 in (
        vdf.query("id_a == 20 and id_b == 30")["features_vector"].iloc[0].values.round()
    )
    assert sorted(
        vdf.query("id_a == 10 and id_b == 50")["features_vector"].iloc[0].values.round()
    ) == [1, 1909]
    main.do_drop_all("")


def test_step_2_split_by_id_a(
    spark,
    main,
    training_conf,
    model_exploration,
    state_dist_path,
    training_data_path,
    fake_self,
):
    """Tests train-test-split which keeps all potential_matches of an id_a together in the same split"""

    training_conf["training"]["n_training_iterations"] = 4
    training_conf["training"]["split_by_id_a"] = True

    prepped_data = spark.read.csv(training_data_path, header=True)
    id_a = training_conf["id_column"] + "_a"
    n_training_iterations = training_conf["training"].get("n_training_iterations", 10)
    seed = training_conf["training"].get("seed", 2133)

    link_step = LinkStepTrainTestModels(model_exploration)
    splits = link_step._get_splits(prepped_data, id_a, n_training_iterations, seed)

    assert len(splits) == 4

    assert splits[0][0].toPandas()["id_a"].unique().tolist() == ["10", "20", "30"]
    assert splits[0][1].toPandas()["id_a"].unique().tolist() == []

    assert splits[1][0].toPandas()["id_a"].unique().tolist() == ["10", "20"]
    assert splits[1][1].toPandas()["id_a"].unique().tolist() == ["30"]

    main.do_drop_all("")


@pytest.mark.skip(
    reason="Need to get tests working for new version of feature importances"
)
def test_step_3_get_feature_importances_random_forest(
    spark,
    training_conf,
    training,
    state_dist_path,
    datasource_training_input,
    potential_matches_path,
    spark_test_tmp_dir_path,
    model_exploration,
):
    """Test running the chosen model on potential matches dataset"""
    td_path, pa_path, pb_path = datasource_training_input

    training_conf["comparison_features"] = [
        {
            "alias": "regionf",
            "column_name": "region",
            "comparison_type": "fetch_a",
            "categorical": True,
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "state_distance",
            "column_name": "bpl",
            "key_count": 1,
            "comparison_type": "geo_distance",
            "loc_a": "statecode1",
            "loc_b": "statecode2",
            "distance_col": "dist",
            "table_name": "state_distances_lookup",
            "distances_file": state_dist_path,
        },
    ]

    training_conf["training"]["dataset"] = td_path
    training_conf["training"]["dependent_var"] = "match"
    training_conf["training"]["independent_vars"] = [
        "namelast_jw",
        "regionf",
        "state_distance",
    ]
    training_conf["training"]["chosen_model"] = {
        "type": "random_forest",
        "maxDepth": 6,
        "numTrees": 100,
        "featureSubsetStrategy": "sqrt",
    }

    # training_conf["training"]["use_potential_matches_features"] = True
    training_conf["training"]["score_with_model"] = True
    training_conf["training"]["feature_importances"] = True
    training_conf["spark_tmp_dir"] = spark_test_tmp_dir_path
    training_conf["drop_data_from_scored_matches"] = True

    training.spark.read.csv(pa_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_a")
    training.spark.read.csv(pb_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_b")
    training.spark.read.csv(
        potential_matches_path, header=True, inferSchema=True
    ).write.mode("overwrite").saveAsTable("potential_matches")

    training.run_step(0)
    training.run_step(1)
    training.run_step(2)

    model_exploration.run_step(3)

    fi_df = training.spark.table("feature_importances").toPandas()

    assert fi_df.shape == (6, 3)
    assert 1 > fi_df.query("idx == 0")["score"].iloc()[0] >= 0
    assert "regionf_onehotencoded_2" in list(fi_df["name"])
    assert "regionf_onehotencoded_invalidValues" in list(fi_df["name"])


@pytest.mark.skip(
    reason="Need to get tests working for new version of feature importances"
)
def test_step_3_get_feature_importances_probit(
    spark,
    training_conf,
    training,
    state_dist_path,
    datasource_training_input,
    potential_matches_path,
    spark_test_tmp_dir_path,
    matching,
):
    """Test running the chosen model on potential matches dataset"""
    td_path, pa_path, pb_path = datasource_training_input

    training_conf["comparison_features"] = [
        {
            "alias": "regionf",
            "column_name": "region",
            "comparison_type": "fetch_a",
            "categorical": True,
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "state_distance",
            "key_count": 1,
            "column_name": "bpl",
            "comparison_type": "geo_distance",
            "loc_a": "statecode1",
            "loc_b": "statecode2",
            "distance_col": "dist",
            "table_name": "state_distances_lookup",
            "distances_file": state_dist_path,
        },
    ]

    training_conf["training"]["dataset"] = td_path
    training_conf["training"]["dependent_var"] = "match"
    training_conf["training"]["independent_vars"] = [
        "namelast_jw",
        "regionf",
        "state_distance",
    ]

    training_conf["training"]["chosen_model"] = {"type": "probit", "threshold": 0.5}

    # training_conf["training"]["use_potential_matches_features"] = True
    training_conf["training"]["score_with_model"] = True
    training_conf["training"]["feature_importances"] = True
    training_conf["spark_tmp_dir"] = spark_test_tmp_dir_path
    training_conf["drop_data_from_scored_matches"] = True

    training.spark.read.csv(pa_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_a")
    training.spark.read.csv(pb_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_b")
    training.spark.read.csv(
        potential_matches_path, header=True, inferSchema=True
    ).write.mode("overwrite").saveAsTable("potential_matches")

    training.run_step(0)
    training.run_step(1)
    training.run_step(2)
    matching.run_step(2)
    training.run_step(3)

    fi_df = training.spark.table("feature_importances").toPandas()

    assert fi_df.shape == (6, 3)
    assert 25 > fi_df.query("idx == 0")["score"].iloc()[0] >= -5
    assert "regionf_onehotencoded_2" in list(fi_df["name"])
    assert "regionf_onehotencoded_invalidValues" in list(fi_df["name"])
