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
    hh_training_data_path,
    potential_matches_path,
    spark_test_tmp_dir_path,
    hh_matching,
):
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

    hh_training_conf["hh_training"]["dataset"] = hh_training_data_path
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
    hh_training_conf["hh_training"]["feature_importance"] = True
    hh_training_conf["spark_tmp_dir"] = spark_test_tmp_dir_path

    hh_training.link_run.trained_models["trained_model"] = None

    hh_training.run_step(0)

    hh_training.run_step(1)
    tf = spark.table("hh_training_features").toPandas()
    assert tf.query("id_a == 20 and id_b == 30")["exact"].iloc[0]


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
    spark, hh_training_conf, hh_training, state_dist_path, datasource_training_input
):
    hh_training_data_path, prepped_df_a_path, prepped_df_b_path = (
        datasource_training_input
    )
    """Run training step 3 with a probit ML model."""
    hh_training_conf["id_column"] = "id"
    hh_training_conf["column_mappings"] = [
        
    ]
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
    hh_training_conf["hh_training"]["dataset"] = hh_training_data_path
    hh_training_conf["hh_training"]["dependent_var"] = "match"
    hh_training_conf["hh_training"]["independent_vars"] = [
        "namelast_jw",
        "regionf",
        "state_distance",
    ]

    hh_training_conf["hh_training"]["chosen_model"] = {
        "type": "probit",
        "threshold": 0.5,
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

    tfi = spark.table("hh_training_feature_importances").toPandas()
    assert (
        8.9
        <= tfi.query("feature_name == 'namelast_jw'")[
            "coefficient_or_importance"
        ].item()
        <= 9.0
    )
    assert (
        tfi.query("feature_name == 'regionf' and category == 0")[
            "coefficient_or_importance"
        ].item()
        == 0
    )
    assert (
        -7.6
        <= tfi.query("feature_name == 'regionf' and category == 1")[
            "coefficient_or_importance"
        ].item()
        <= -7.5
    )
    assert (
        6.4
        <= tfi.query("feature_name == 'regionf' and category == 99")[
            "coefficient_or_importance"
        ].item()
        <= 6.5
    )
