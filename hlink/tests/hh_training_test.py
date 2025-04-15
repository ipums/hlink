# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pytest
from pyspark.ml import Pipeline
import hlink.linking.core.pipeline as pipeline_core
from hlink.tests.markers import requires_lightgbm, requires_xgboost
from hlink.tests.conftest import load_table_from_csv


def test_all_steps(
    spark,
    hh_training_conf,
    hh_training,
    state_dist_path,
    training_data_path,
    spark_test_tmp_dir_path,
):
    hh_training_conf["id_column"] = "id"
    hh_training_conf["comparison_features"] = [
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
        {
            "alias": "exact",
            "column_names": ["namelast", "namelast"],
            "comparison_type": "all_equals",
        },
        {
            "alias": "exact_all",
            "column_names": ["namelast", "bpl", "sex"],
            "comparison_type": "all_equals",
        },
    ]
    hh_training_conf["hh_training"]["feature_importances"] = True
    hh_training_conf["hh_training"]["dataset"] = training_data_path
    hh_training_conf["hh_training"]["dependent_var"] = "match"
    hh_training_conf["hh_training"]["independent_vars"] = [
        "namelast_jw",
        "regionf",
        "state_distance",
        "exact_mult",
        "exact_all_mult",
        "hits",
        "namelast_jw_buckets",
    ]
    hh_training_conf["pipeline_features"] = [
        {
            "input_column": "namelast_jw",
            "output_column": "namelast_jw_buckets",
            "transformer_type": "bucketizer",
            "categorical": True,
            "splits": [0, 0.25, 0.5, 0.75, 0.99, 1],
        }
    ]
    hh_training_conf["hh_training"]["chosen_model"] = {
        "type": "random_forest",
        "maxDepth": 6,
        "numTrees": 100,
        "featureSubsetStrategy": "sqrt",
    }
    hh_training_conf["hh_training"]["score_with_model"] = True
    hh_training_conf["spark_tmp_dir"] = spark_test_tmp_dir_path

    hh_training.link_run.trained_models["trained_model"] = None

    hh_training.run_step(0)

    hh_training.run_step(1)
    tf = spark.table("hh_training_features").toPandas()
    assert tf.query("id_a == 20 and id_b == 30")["exact"].iloc[0]
    assert not tf.query("id_a == 20 and id_b == 50")["exact"].iloc[0]
    assert not tf.query("id_a == 20 and id_b == 30")["exact_mult"].iloc[0]
    assert not tf.query("id_a == 20 and id_b == 10")["exact_mult"].iloc[0]

    hh_training.run_step(2)
    p = hh_training.link_run.trained_models["hh_pre_pipeline"]
    m = hh_training.link_run.trained_models["hh_trained_model"]
    transformed_df = m.transform(
        p.transform(spark.table("hh_training_features"))
    ).toPandas()
    row = transformed_df.query("id_a == 10 and id_b == 50").iloc[0]
    assert row.prediction == 0
    assert row.state_distance_imp.round(0) == 1909

    hh_training.run_step(3)

    tf = spark.table("hh_training_feature_importances").toPandas()
    for var in hh_training_conf["hh_training"]["independent_vars"]:
        assert not tf.loc[tf["feature_name"].str.startswith(f"{var}", na=False)].empty
    assert all(
        [
            col in ["feature_name", "category", "coefficient_or_importance"]
            for col in tf.columns
        ]
    )
    assert (tf["coefficient_or_importance"] >= 0).all() and (
        tf["coefficient_or_importance"] <= 1
    ).all()

    assert (
        0.4
        <= tf.query("feature_name == 'namelast_jw'")["coefficient_or_importance"].item()
        <= 0.5
    )
    assert (
        0.1
        <= tf.query("feature_name == 'namelast_jw_buckets' and category == 4")[
            "coefficient_or_importance"
        ].item()
        <= 0.2
    )
    assert (
        0.1
        <= tf.query("feature_name == 'state_distance'")[
            "coefficient_or_importance"
        ].item()
        <= 0.21
    )
    assert (
        tf.query("feature_name == 'regionf' and category == 0")[
            "coefficient_or_importance"
        ].item()
        <= 0.1
    )


def test_step_3_interacted_categorical_features(
    hh_training_conf,
    hh_training,
    hh_training_data_path,
    spark,
    hh_matching,
    hh_integration_test_data,
):
    """Test all hh_training and hh_matching steps to ensure they work as a pipeline"""

    hh_training_conf["hh_training"]["dataset"] = hh_training_data_path
    hh_training_conf["hh_training"]["dependent_var"] = "match"
    hh_training_conf["hh_training"]["independent_vars"] = [
        "namelast_jw",
        "namefrst_jw",
        "byrdiff",
        "ssex",
        "srelate",
        "ssex_interacted_namelast_jw",
    ]

    # Interacting a categorical feature with another feature creates a new categorical
    # feature. We should get coefficients for this new categorical feature as well.
    hh_training_conf["pipeline_features"] = [
        {
            "input_columns": ["ssex", "namelast_jw"],
            "output_column": "ssex_interacted_namelast_jw",
            "transformer_type": "interaction",
        }
    ]
    hh_training_conf["hh_training"]["chosen_model"] = {
        "type": "random_forest",
        "maxDepth": 6,
        "numTrees": 100,
        "featureSubsetStrategy": "sqrt",
    }

    hh_training_conf["hh_training"]["score_with_model"] = True
    hh_training_conf["hh_training"]["feature_importances"] = True

    path_a, path_b, path_pms = hh_integration_test_data

    load_table_from_csv(hh_matching, path_a, "prepped_df_a")
    load_table_from_csv(hh_matching, path_b, "prepped_df_b")
    load_table_from_csv(hh_matching, path_pms, "predicted_matches")

    hh_training.run_step(0)
    hh_training.run_step(1)
    hh_training.run_step(2)
    hh_training.run_step(3)

    tf = spark.table("hh_training_feature_importances").toPandas()
    assert (
        0.0
        <= tf.query("feature_name == 'ssex_interacted_namelast_jw' and category == 0")[
            "coefficient_or_importance"
        ].item()
        <= 1.0
    )
    assert (
        0.4
        <= tf.query("feature_name == 'namefrst_jw'")["coefficient_or_importance"].item()
        <= 0.5
    )


def test_step_3_with_probit_model(
    spark,
    hh_training_conf,
    hh_training,
    hh_matching,
    hh_training_data_path,
    hh_integration_test_data,
):
    """Run hh_training step 3 with a probit ML model."""
    hh_training_conf["comparison_features"] = [
        {
            "alias": "byrdiff",
            "column_name": "clean_birthyr",
            "comparison_type": "abs_diff",
        },
        {
            "alias": "ssex",
            "column_name": "sex",
            "comparison_type": "equals",
            "categorical": True,
        },
        {
            "alias": "srelate",
            "column_name": "relate",
            "comparison_type": "equals",
            "categorical": True,
        },
        {
            "alias": "namefrst_jw",
            "column_name": "namefrst_unstd",
            "comparison_type": "jaro_winkler",
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
    ]
    hh_training_conf["hh_training"]["dataset"] = hh_training_data_path
    hh_training_conf["hh_training"]["dependent_var"] = "match"
    hh_training_conf["hh_training"]["independent_vars"] = [
        "ssex",
        "srelate",
        "namefrst_jw",
        "byrdiff",
    ]

    hh_training_conf["hh_training"]["chosen_model"] = {
        "type": "probit",
        "threshold": 0.5,
    }
    hh_training_conf["hh_training"]["score_with_model"] = True
    hh_training_conf["hh_training"]["feature_importances"] = True

    prepped_df_a_path, prepped_df_b_path, path_pms = hh_integration_test_data

    spark.read.csv(prepped_df_a_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_a")
    spark.read.csv(prepped_df_b_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_b")
    load_table_from_csv(hh_matching, path_pms, "predicted_matches")

    hh_training.run_step(0)
    hh_training.run_step(1)
    hh_training.run_step(2)
    hh_training.run_step(3)

    tfi = spark.table("hh_training_feature_importances").toPandas()
    assert (
        33.5
        <= tfi.query("feature_name == 'namefrst_jw'")[
            "coefficient_or_importance"
        ].item()
        <= 34.0
    )
    assert (
        tfi.query("feature_name == 'srelate' and category == 0")[
            "coefficient_or_importance"
        ].item()
        == 0
    )
    assert (
        -0.7
        <= tfi.query("feature_name == 'ssex' and category == 1")[
            "coefficient_or_importance"
        ].item()
        <= -0.6
    )
    assert (
        -0.3
        <= tfi.query("feature_name == 'byrdiff'")["coefficient_or_importance"].item()
        <= -0.2
    )


@requires_lightgbm
def test_lightgbm_with_interacted_features(
    spark, hh_training, hh_training_conf, hh_integration_test_data
):
    """
    Interacted features add colons to vector attribute names, which cause
    problems for LightGBM. Hlink handles this automatically by renaming the
    vector attributes to remove the colons before invoking LightGBM.
    """
    prepped_df_a_path, prepped_df_b_path, hh_training_data_path = (
        hh_integration_test_data
    )
    hh_training_conf["comparison_features"] = [
        {
            "alias": "bpl",
            "column_name": "bpl_clean",
            "comparison_type": "fetch_a",
            "categorical": True,
        },
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
    ]
    hh_training_conf["pipeline_features"] = [
        {
            "input_columns": ["bpl", "namelast_jw"],
            "output_column": "bpl_interacted_namelast_jw",
            "transformer_type": "interaction",
        }
    ]
    hh_training_conf["hh_training"]["dataset"] = hh_training_data_path
    hh_training_conf["hh_training"]["dependent_var"] = "match"
    hh_training_conf["hh_training"]["independent_vars"] = [
        "namelast_jw",
        "bpl",
        "bpl_interacted_namelast_jw",
    ]
    hh_training_conf["hh_training"]["chosen_model"] = {
        "type": "lightgbm",
        "maxDepth": 7,
        "numIterations": 5,
        "minDataInLeaf": 1,
        "threshold": 0.5,
    }
    hh_training_conf["hh_training"]["score_with_model"] = True
    hh_training_conf["hh_training"]["feature_importances"] = True
    prepped_df_a = spark.read.csv(prepped_df_a_path, header=True, inferSchema=True)
    prepped_df_b = spark.read.csv(prepped_df_b_path, header=True, inferSchema=True)

    prepped_df_a.write.mode("overwrite").saveAsTable("prepped_df_a")
    prepped_df_b.write.mode("overwrite").saveAsTable("prepped_df_b")

    hh_training.run_all_steps()

    importances_df = spark.table("hh_training_feature_importances")
    assert importances_df.columns == [
        "feature_name",
        "category",
        "weight",
        "gain",
    ]


@requires_lightgbm
def test_lightgbm_with_bucketized_features(
    spark, hh_training, hh_training_conf, hh_integration_test_data
):
    """
    Bucketized features add commas to vector attribute names, which cause
    problems for LightGBM. Hlink handles this automatically by renaming the
    vector attributes to remove the commas before invoking LightGBM.
    """
    prepped_df_a_path, prepped_df_b_path, hh_training_data_path = (
        hh_integration_test_data
    )
    hh_training_conf["comparison_features"] = [
        {
            "alias": "namelast_jw",
            "column_name": "namelast_clean",
            "comparison_type": "jaro_winkler",
        },
    ]
    hh_training_conf["pipeline_features"] = [
        {
            "input_column": "namelast_jw",
            "output_column": "namelast_jw_buckets",
            "transformer_type": "bucketizer",
            "categorical": True,
            "splits": [0.0, 0.33, 0.67, 1.0],
        }
    ]
    hh_training_conf["hh_training"]["dataset"] = hh_training_data_path
    hh_training_conf["hh_training"]["dependent_var"] = "match"
    hh_training_conf["hh_training"]["independent_vars"] = [
        "namelast_jw_buckets",
    ]
    hh_training_conf["hh_training"]["chosen_model"] = {
        "type": "lightgbm",
        "threshold": 0.5,
    }
    hh_training_conf["hh_training"]["score_with_model"] = True
    hh_training_conf["hh_training"]["feature_importances"] = True

    prepped_df_a = spark.read.csv(prepped_df_a_path, header=True, inferSchema=True)
    prepped_df_b = spark.read.csv(prepped_df_b_path, header=True, inferSchema=True)

    prepped_df_a.write.mode("overwrite").saveAsTable("prepped_df_a")
    prepped_df_b.write.mode("overwrite").saveAsTable("prepped_df_b")

    hh_training.run_all_steps()

    importances_df = spark.table("hh_training_feature_importances")
    assert importances_df.columns == [
        "feature_name",
        "category",
        "weight",
        "gain",
    ]


@requires_xgboost
def test_step_3_with_xgboost_model(
    spark,
    hh_training,
    hh_training_conf,
    hh_matching_stubs,
):
    prepped_df_a_path, prepped_df_b_path, path_matches, _ = hh_matching_stubs
    hh_training_conf["comparison_features"] = [
        {
            "alias": "byrdiff",
            "column_name": "birthyr",
            "comparison_type": "abs_diff",
        },
        {
            "alias": "namefrst_jw",
            "column_name": "namefrst",
            "comparison_type": "jaro_winkler",
        },
    ]
    hh_training_conf["hh_training"]["dataset"] = path_matches
    hh_training_conf["hh_training"]["dependent_var"] = "prediction"
    hh_training_conf["hh_training"]["independent_vars"] = ["namefrst_jw", "byrdiff"]
    hh_training_conf["hh_training"]["chosen_model"] = {
        "type": "xgboost",
        "max_depth": 2,
        "eta": 0.5,
        "threshold": 0.7,
        "threshold_ratio": 1.3,
    }
    hh_training_conf["hh_training"]["score_with_model"] = True
    hh_training_conf["hh_training"]["feature_importances"] = True

    spark.read.csv(prepped_df_a_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_a")
    spark.read.csv(prepped_df_b_path, header=True, inferSchema=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_b")

    hh_training.run_step(0)
    hh_training.run_step(1)
    hh_training.run_step(2)
    hh_training.run_step(3)

    importances_df = spark.table("hh_training_feature_importances")
    assert importances_df.columns == [
        "feature_name",
        "category",
        "weight",
        "gain",
    ]


def test_step_3_requires_table(hh_training_conf, hh_training):
    hh_training_conf["hh_training"]["feature_importances"] = True
    with pytest.raises(RuntimeError, match="Missing input tables"):
        hh_training.run_step(3)


def test_step_3_skipped_on_no_feature_importances(
    hh_training_conf, hh_training, spark, capsys
):
    """Step 3 is skipped when there is no hh_training.feature_importances attribute
    in the config."""
    assert "feature_importances" not in hh_training_conf["hh_training"]
    mock_tf_prepped = spark.createDataFrame(
        [], "id_a: int, id_b: int, namelast_jw_imp: float, match: boolean"
    )
    mock_tf_prepped.write.saveAsTable("hh_training_features_prepped")
    hh_training.run_step(3)

    output = capsys.readouterr().out
    assert "Skipping the save model metadata hh_training step" in output


def test_step_3_skipped_on_false_feature_importances(
    hh_training_conf, hh_training, spark, capsys
):
    """Step 3 is skipped when hh_training.feature_importances is set to false in
    the config."""
    hh_training_conf["hh_training"]["feature_importances"] = False
    mock_tf_prepped = spark.createDataFrame(
        [], "id_a: int, id_b: int, namelast_jw_imp: float, match: boolean"
    )
    mock_tf_prepped.write.saveAsTable("hh_training_features_prepped")

    hh_training.run_step(3)

    output = capsys.readouterr().out
    assert "Skipping the save model metadata hh_training step" in output


def test_step_3_model_not_found(hh_training_conf, hh_training, spark):
    """Step 3 raises an exception when the trained model is not available."""
    hh_training_conf["hh_training"]["feature_importances"] = True
    mock_tf_prepped = spark.createDataFrame(
        [], "id_a: int, id_b: int, namelast_jw_imp: float, match: boolean"
    )
    mock_tf_prepped.write.saveAsTable("hh_training_features_prepped")
    with pytest.raises(
        RuntimeError,
        match="Model not found!  Please run hh_training step 2 to generate and train the chosen model",
    ):
        hh_training.run_step(3)
