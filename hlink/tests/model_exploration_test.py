# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
from collections import Counter

import pandas as pd
from pyspark.sql import SparkSession
import pytest

import hlink.linking.core.threshold as threshold_core
from hlink.linking.model_exploration.link_step_train_test_models import (
    LinkStepTrainTestModels,
    _custom_param_grid_builder,
    _get_model_parameters,
    _get_confusion_matrix,
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
    training_conf["training"]["n_training_iterations"] = 3

    model_exploration.run_step(0)
    model_exploration.run_step(1)
    model_exploration.run_step(2)

    tr = spark.table("model_eval_training_results").toPandas()
    print(f"Test all results: {tr}")

    assert tr.__len__() == 2
    # TODO this should be a valid test once we fix the results output
    # assert tr.query("threshold_ratio == 1.01")["precision_test_mean"].iloc[0] >= 0.5
    assert tr.query("threshold_ratio == 1.3")["alpha_threshold"].iloc[0] == 0.8

    # The old behavior was to process all the model types, but now we select the best
    # model before moving forward to testing the threshold combinations. So the
    # Random Forest results aren't made now.
    # assert tr.query("model == 'random_forest'")["maxDepth"].iloc[0] == 5
    # assert tr.query("model == 'random_forest'")["pr_auc_mean"].iloc[0] > 0.8
    # assert (
    #    tr.query("threshold_ratio == 1.01")["pr_auc_mean"].iloc[0]
    #   == tr.query("threshold_ratio == 1.3")["pr_auc_mean"].iloc[0]
    # )

    # TODO these asserts will mostly succeed if you change the random number seed: Basically the
    """
    preds = spark.table("model_eval_predictions").toPandas()
    assert (
        preds.query("id_a == 20 and id_b == 30")["probability"].round(2).iloc[0] > 0.5
    )

    assert (
        preds.query("id_a == 20 and id_b == 30")["second_best_prob"].round(2).iloc[0]
        >= 0.6
    )

    assert preds.query("id_a == 30 and id_b == 30")["prediction"].iloc[0] == 0
    assert pd.isnull(
        preds.query("id_a == 10 and id_b == 30")["second_best_prob"].iloc[0]
    )

    pred_train = spark.table("model_eval_predict_train").toPandas()
    assert pred_train.query("id_a == 20 and id_b == 50")["match"].iloc[0] == 0
    """
    # assert pd.isnull(
    #     pred_train.query("id_a == 10 and id_b == 50")["second_best_prob"].iloc[1]
    # )
    # assert pred_train.query("id_a == 20 and id_b == 50")["prediction"].iloc[1] == 1

    main.do_drop_all("")


def test_custom_param_grid_builder():
    """Test matching step 2's custom param grid builder"""
    model_parameters = [
        {"type": "random_forest", "maxDepth": [3, 4, 5], "numTrees": [50, 100]},
        {"type": "probit", "threshold": [0.5, 0.7]},
    ]
    param_grid = _custom_param_grid_builder(model_parameters)

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
    assert all(m in expected for m in param_grid)


def test_get_model_parameters_error_if_list_empty(training_conf):
    """
    It's an error if the model_parameters list is empty, since in that case there
    aren't any models to evaluate.
    """
    training_conf["training"]["model_parameters"] = []

    with pytest.raises(ValueError, match="model_parameters is empty"):
        _get_model_parameters(training_conf["training"])


def test_get_model_parameters_default_behavior(training_conf):
    """
    When there's no training.param_grid attribute or
    training.model_parameter_search attribute, the default is to use the
    "explicit" strategy, testing each element of model_parameters in turn.
    """
    training_conf["training"]["model_parameters"] = [
        {"type": "random_forest", "maxDepth": 3, "numTrees": 50},
        {"type": "probit", "threshold": 0.7},
    ]
    assert "param_grid" not in training_conf["training"]
    assert "model_parameter_search" not in training_conf["training"]

    model_parameters = _get_model_parameters(training_conf["training"])

    assert model_parameters == [
        {"type": "random_forest", "maxDepth": 3, "numTrees": 50},
        {"type": "probit", "threshold": 0.7},
    ]


def test_get_model_parameters_param_grid_false(training_conf, capsys):
    """
    When training.param_grid is set to False, model exploration uses the "explicit"
    strategy. The model_parameters are returned unchanged.

    This prints a deprecation warning because param_grid is deprecated.
    """
    training_conf["training"]["model_parameters"] = [
        {"type": "logistic_regression", "threshold": 0.3, "threshold_ratio": 1.4},
    ]
    training_conf["training"]["param_grid"] = False

    model_parameters = _get_model_parameters(training_conf["training"])

    assert model_parameters == [
        {"type": "logistic_regression", "threshold": 0.3, "threshold_ratio": 1.4},
    ]

    output = capsys.readouterr()
    assert "Deprecation Warning: training.param_grid is deprecated" in output.err


def test_get_model_parameters_param_grid_true(training_conf, capsys):
    """
    When training.param_grid is set to True, model exploration uses the "grid"
    strategy, exploding model_parameters.

    This prints a deprecation warning because param_grid is deprecated.
    """
    training_conf["training"]["model_parameters"] = [
        {
            "type": "random_forest",
            "maxDepth": [5, 10, 15],
            "numTrees": [50, 100],
            "threshold": 0.5,
        },
    ]
    training_conf["training"]["param_grid"] = True

    model_parameters = _get_model_parameters(training_conf["training"])
    # 3 settings for maxDepth * 2 settings for numTrees = 6 total settings
    assert len(model_parameters) == 6

    output = capsys.readouterr()
    assert "Deprecation Warning: training.param_grid is deprecated" in output.err


def test_get_model_parameters_search_strategy_explicit(training_conf):
    """
    When training.model_parameter_search.strategy is set to "explicit",
    model_parameters pass through unchanged.
    """
    training_conf["training"]["model_parameters"] = [
        {"type": "random_forest", "maxDepth": 15, "numTrees": 100, "threshold": 0.5},
        {"type": "probit", "threshold": 0.8, "threshold_ratio": 1.3},
    ]
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "explicit",
    }
    assert "param_grid" not in training_conf["training"]

    model_parameters = _get_model_parameters(training_conf["training"])

    assert model_parameters == [
        {"type": "random_forest", "maxDepth": 15, "numTrees": 100, "threshold": 0.5},
        {"type": "probit", "threshold": 0.8, "threshold_ratio": 1.3},
    ]


def test_get_model_parameters_search_strategy_grid(training_conf):
    """
    When training.model_parameter_search.strategy is set to "grid",
    model_parameters are exploded.
    """
    training_conf["training"]["model_parameters"] = [
        {
            "type": "random_forest",
            "maxDepth": [5, 10, 15],
            "numTrees": [50, 100],
            "threshold": 0.5,
        },
    ]
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "grid",
    }
    assert "param_grid" not in training_conf

    model_parameters = _get_model_parameters(training_conf["training"])
    # 3 settings for maxDepth * 2 settings for numTrees = 6 total settings
    assert len(model_parameters) == 6


def test_get_model_parameters_search_strategy_explicit_with_param_grid_true(
    training_conf, capsys
):
    """
    When both model_parameter_search and param_grid are set, model_parameter_search
    takes precedence.

    This prints a deprecation warning because param_grid is deprecated.
    """
    training_conf["training"]["model_parameters"] = [
        {
            "type": "random_forest",
            "maxDepth": 10,
            "numTrees": 75,
            "threshold": 0.7,
        }
    ]
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "explicit",
    }
    # model_parameter_search takes precedence over this
    training_conf["training"]["param_grid"] = True

    model_parameters = _get_model_parameters(training_conf["training"])
    assert model_parameters == [
        {"type": "random_forest", "maxDepth": 10, "numTrees": 75, "threshold": 0.7}
    ]

    output = capsys.readouterr()
    assert "Deprecation Warning: training.param_grid is deprecated" in output.err


def test_get_model_parameters_search_strategy_grid_with_param_grid_false(
    training_conf, capsys
):
    """
    When both model_parameter_search and param_grid are set, model_parameter_search
    takes precedence.

    This prints a deprecation warning because param_grid is deprecated.
    """
    training_conf["training"]["model_parameters"] = [
        {
            "type": "random_forest",
            "maxDepth": [5, 10, 15],
            "numTrees": [50, 100],
            "threshold": 0.5,
        },
    ]
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "grid",
    }
    # model_parameter_search takes precedence over this
    training_conf["training"]["param_grid"] = False

    model_parameters = _get_model_parameters(training_conf["training"])
    assert len(model_parameters) == 6

    output = capsys.readouterr()
    assert "Deprecation Warning: training.param_grid is deprecated" in output.err


def test_get_model_parameters_search_strategy_randomized_sample_from_lists(
    training_conf,
):
    """
    Strategy "randomized" accepts lists for parameter values, but it does not work
    the same way as the "grid" strategy. It randomly samples values from the lists
    num_samples times to create parameter combinations.
    """
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "randomized",
        "num_samples": 37,
    }
    training_conf["training"]["model_parameters"] = [
        {
            "type": "decision_tree",
            "maxDepth": [1, 5, 10, 20],
            "maxBins": [10, 20, 40],
        }
    ]

    model_parameters = _get_model_parameters(training_conf["training"])

    # Note that if we used strategy grid, we would get a list of length 4 * 3 = 12 instead
    assert len(model_parameters) == 37

    for parameter_choice in model_parameters:
        assert parameter_choice["type"] == "decision_tree"
        assert parameter_choice["maxDepth"] in {1, 5, 10, 20}
        assert parameter_choice["maxBins"] in {10, 20, 40}


def test_get_model_parameters_search_strategy_randomized_sample_from_distributions(
    training_conf,
):
    """
    The "randomized" strategy also accepts dictionary values for parameters.
    These dictionaries define distributions from which the parameters should be
    sampled.

    For example, {"distribution": "randint", "low": 1, "high": 20} means to
    pick a random integer between 1 and 20, each integer with an equal chance.
    And {"distribution": "uniform", "low": 0.0, "high": 100.0} means to pick a
    random float between 0.0 and 100.0 with a uniform distribution.
    """
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "randomized",
        "num_samples": 15,
    }
    training_conf["training"]["model_parameters"] = [
        {
            "type": "decision_tree",
            "maxDepth": {"distribution": "randint", "low": 1, "high": 20},
            "minInfoGain": {"distribution": "uniform", "low": 0.0, "high": 100.0},
            "minWeightFractionPerNode": {
                "distribution": "normal",
                "mean": 10.0,
                "standard_deviation": 2.5,
            },
        }
    ]

    model_parameters = _get_model_parameters(training_conf["training"])

    assert len(model_parameters) == 15

    for parameter_choice in model_parameters:
        assert parameter_choice["type"] == "decision_tree"
        assert 1 <= parameter_choice["maxDepth"] <= 20
        assert 0.0 <= parameter_choice["minInfoGain"] <= 100.0
        # Technically a normal distribution can return any value, even ones very
        # far from its mean. So we can't assert on the value returned here. But
        # there definitely should be a value of some sort in the dictionary.
        assert "minWeightFractionPerNode" in parameter_choice


def test_get_model_parameters_search_strategy_randomized_take_values(training_conf):
    """
    If a value is neither a list nor a table, the "randomized" strategy just passes
    it along as a value. This lets the user easily pin some parameters to a particular
    value and randomize others.
    """
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "randomized",
        "num_samples": 25,
    }
    training_conf["training"]["model_parameters"] = [
        {
            "type": "random_forest",
            "maxDepth": 7,
            "impurity": "entropy",
            "minInfoGain": 0.5,
            "numTrees": {"distribution": "randint", "low": 10, "high": 100},
            "subsamplingRate": [0.5, 1.0, 1.5],
        }
    ]

    model_parameters = _get_model_parameters(training_conf["training"])

    assert len(model_parameters) == 25

    for parameter_choice in model_parameters:
        assert parameter_choice["type"] == "random_forest"
        assert parameter_choice["maxDepth"] == 7
        assert parameter_choice["impurity"] == "entropy"
        assert parameter_choice["minInfoGain"] == 0.5
        assert 10 <= parameter_choice["numTrees"] <= 100
        assert parameter_choice["subsamplingRate"] in {0.5, 1.0, 1.5}


def test_get_model_parameters_search_strategy_randomized_multiple_models(training_conf):
    """
    When there are multiple models for the "randomized" strategy, it randomly
    samples the model before sampling the parameters for that model. Setting
    the training.seed attribute lets us assert more precisely the counts for
    each model type.
    """
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "randomized",
        "num_samples": 100,
    }
    training_conf["training"]["seed"] = 101
    training_conf["training"]["model_parameters"] = [
        {
            "type": "random_forest",
            "minInfoGain": {"distribution": "uniform", "low": 0.1, "high": 0.9},
        },
        {"type": "probit"},
    ]

    model_parameters = _get_model_parameters(training_conf["training"])

    counter = Counter(parameter_choice["type"] for parameter_choice in model_parameters)
    assert counter["random_forest"] == 47
    assert counter["probit"] == 53


def test_get_model_parameters_search_strategy_randomized_uses_seed(training_conf):
    """
    The "randomized" strategy uses training.seed to allow reproducible runs.
    """
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "randomized",
        "num_samples": 5,
    }
    training_conf["training"]["seed"] = 35830969
    training_conf["training"]["model_parameters"] = [
        {
            "type": "random_forest",
            "maxDepth": {"distribution": "randint", "low": 1, "high": 10},
            "numTrees": [1, 10, 100, 1000],
        }
    ]

    model_parameters = _get_model_parameters(training_conf["training"])

    assert model_parameters == [
        {"type": "random_forest", "maxDepth": 8, "numTrees": 100},
        {"type": "random_forest", "maxDepth": 2, "numTrees": 1},
        {"type": "random_forest", "maxDepth": 4, "numTrees": 100},
        {"type": "random_forest", "maxDepth": 9, "numTrees": 10},
        {"type": "random_forest", "maxDepth": 7, "numTrees": 100},
    ]


def test_get_model_parameters_search_strategy_randomized_unknown_distribution(
    training_conf,
):
    """
    Passing a distrbution other than "uniform", "randint", or "normal" is an error.
    """
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "randomized",
        "num_samples": 10,
    }
    training_conf["training"]["model_parameters"] = [
        {
            "type": "decision_tree",
            "minInfoGain": {"distribution": "laplace", "location": 0.0, "scale": 1.0},
        }
    ]

    with pytest.raises(
        ValueError,
        match="Unknown distribution 'laplace'. Please choose one of 'randint', 'uniform', or 'normal'.",
    ):
        _get_model_parameters(training_conf["training"])


def test_get_model_parameters_search_strategy_randomized_thresholds(training_conf):
    """
    Even when the model parameters are selected with strategy "randomized", the
    thresholds are still treated with a "grid" strategy.
    _get_model_parameters() is not in charge of creating the threshold matrix,
    so it passes the threshold and threshold_ratio through unchanged.
    """
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "randomized",
        "num_samples": 25,
    }
    training_conf["training"]["model_parameters"] = [
        {
            "type": "random_forest",
            "maxDepth": [1, 10, 100],
            "threshold": [0.3, 0.5, 0.7, 0.8, 0.9],
            "threshold_ratio": 1.2,
        }
    ]

    model_parameters = _get_model_parameters(training_conf["training"])

    for parameter_choice in model_parameters:
        assert parameter_choice["type"] == "random_forest"
        assert parameter_choice["threshold"] == [0.3, 0.5, 0.7, 0.8, 0.9]
        assert parameter_choice["threshold_ratio"] == 1.2


def test_get_model_parameters_unknown_search_strategy(training_conf):
    training_conf["training"]["model_parameter_search"] = {
        "strategy": "something",
    }
    training_conf["training"]["model_parameters"] = [{"type": "probit"}]

    with pytest.raises(
        ValueError,
        match="Unknown model_parameter_search strategy 'something'. "
        "Please choose one of 'explicit', 'grid', or 'randomized'.",
    ):
        _parameters = _get_model_parameters(training_conf["training"])


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
    training_conf["training"]["n_training_iterations"] = 3
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
    assert all(c in training_v.columns for c in columns_expected)
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
    feature_conf["training"]["n_training_iterations"] = 3

    model_exploration.run_step(0)
    model_exploration.run_step(1)
    model_exploration.run_step(2)

    tr = spark.table("model_eval_training_results").toPandas()
    print(f"training results {tr}")
    # assert tr.shape == (1, 18)
    assert tr.query("model == 'random_forest'")["pr_auc_mean"].iloc[0] > 2.0 / 3.0
    #  assert tr.query("model == 'random_forest'")["maxDepth"].iloc[0] == 3

    # TODO probably remove these since we're not planning to test suspicious data anymore.
    # I disabled the saving of suspicious in this test config so these are invalid currently.
    """
    FNs = spark.table("model_eval_repeat_fns").toPandas()
    assert FNs.shape == (3, 4)
    assert FNs.query("id_a == 30")["count"].iloc[0] == 3

    TPs = spark.table("model_eval_repeat_tps").toPandas()
    assert TPs.shape == (0, 4)

    TNs = spark.table("model_eval_repeat_tns").toPandas()
    assert TNs.shape == (6, 4)
    """

    main.do_drop_all("")


def test_step_2_train_logistic_regression_spark(
    spark, main, feature_conf, model_exploration, state_dist_path, training_data_path
):
    """Test training step 2 with logistic regression model"""
    feature_conf["training"]["model_parameters"] = [
        {"type": "logistic_regression", "threshold": 0.7}
    ]
    feature_conf["training"]["n_training_iterations"] = 3

    model_exploration.run_step(0)
    model_exploration.run_step(1)
    model_exploration.run_step(2)

    tr = spark.table("model_eval_training_results").toPandas()
    # assert tr.count == 3

    assert tr.shape == (1, 14)
    # This is now 0.83333333333.... I'm not sure it's worth testing against
    # assert tr.query("model == 'logistic_regression'")["pr_auc_mean"].iloc[0] == 0.75
    assert tr.query("model == 'logistic_regression'")["pr_auc_mean"].iloc[0] > 0.74
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
    feature_conf["training"]["n_training_iterations"] = 3

    model_exploration.run_step(0)
    model_exploration.run_step(1)
    model_exploration.run_step(2)

    tr = spark.table("model_eval_training_results").toPandas()

    print(f"Decision tree results: {tr}")

    assert tr.shape == (1, 14)
    # assert tr.query("model == 'decision_tree'")["precision_mean"].iloc[0] > 0
    #  assert tr.query("model == 'decision_tree'")["maxDepth"].iloc[0] == 3
    #  assert tr.query("model == 'decision_tree'")["minInstancesPerNode"].iloc[0] == 1
    #  assert tr.query("model == 'decision_tree'")["maxBins"].iloc[0] == 7

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
    feature_conf["training"]["n_training_iterations"] = 3

    model_exploration.run_step(0)
    model_exploration.run_step(1)
    model_exploration.run_step(2)

    tr = spark.table("model_eval_training_results").toPandas()
    preds = spark.table("model_eval_predictions").toPandas()

    assert "probability_array" in list(preds.columns)

    # import pdb
    # pdb.set_trace()

    training_results = tr.query("model == 'gradient_boosted_trees'")

    # print(f"XX training_results: {training_results}")

    # assert tr.shape == (1, 18)
    # TODO once the train_tgest results are properly combined this should pass
    # assert (
    #    tr.query("model == 'gradient_boosted_trees'")["precision_test_mean"].iloc[0] > 0
    # )
    #  assert tr.query("model == 'gradient_boosted_trees'")["maxDepth"].iloc[0] == 5
    #  assert (
    #  tr.query("model == 'gradient_boosted_trees'")["minInstancesPerNode"].iloc[0]
    #  == 1
    #  )
    #  assert tr.query("model == 'gradient_boosted_trees'")["maxBins"].iloc[0] == 5

    main.do_drop_all("")


def test_step_2_interact_categorical_vars(
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


def test_get_confusion_matrix(spark: SparkSession) -> None:
    # 1 true negative (0, 0)
    # 2 false negatives (1, 0)
    # 3 false postives (0, 1)
    # 4 true positives (1, 1)
    rows = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ]
    predictions = spark.createDataFrame(rows, schema=["match", "prediction"])
    true_positives, false_positives, false_negatives, true_negatives = (
        _get_confusion_matrix(predictions, "match")
    )

    assert true_positives == 4
    assert false_positives == 3
    assert false_negatives == 2
    assert true_negatives == 1
