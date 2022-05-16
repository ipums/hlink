# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pytest
from pyspark.ml import Pipeline
import hlink.linking.core.pipeline as pipeline_core


@pytest.mark.quickcheck
def test_all_steps(
    spark,
    training_conf,
    training,
    state_dist_path,
    training_data_path,
    potential_matches_path,
    spark_test_tmp_dir_path,
    matching,
    training_validation_path,
):
    """Test running the chosen model on potential matches dataset"""
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

    training_conf["training"]["dataset"] = training_data_path
    training_conf["training"]["dependent_var"] = "match"
    training_conf["training"]["independent_vars"] = [
        "namelast_jw",
        "regionf",
        "state_distance",
        "exact_mult",
        "exact_all_mult",
        "hits",
        "namelast_jw_buckets",
    ]
    training_conf["pipeline_features"] = [
        {
            "input_column": "namelast_jw",
            "output_column": "namelast_jw_buckets",
            "transformer_type": "bucketizer",
            "categorical": True,
            "splits": [0, 0.25, 0.5, 0.75, 0.99, 1],
        }
    ]
    training_conf["training"]["chosen_model"] = {
        "type": "random_forest",
        "maxDepth": 6,
        "numTrees": 100,
        "featureSubsetStrategy": "sqrt",
    }

    # training_conf["training"]["use_potential_matches_features"] = True
    training_conf["training"]["score_with_model"] = True
    training_conf["spark_tmp_dir"] = spark_test_tmp_dir_path

    training.link_run.trained_models["trained_model"] = None

    training.run_step(0)

    training.run_step(1)
    tf = spark.table("training_features").toPandas()
    assert tf.query("id_a == 20 and id_b == 30")["exact"].iloc[0]
    assert not tf.query("id_a == 20 and id_b == 50")["exact"].iloc[0]
    assert not tf.query("id_a == 20 and id_b == 30")["exact_mult"].iloc[0]
    assert not tf.query("id_a == 20 and id_b == 10")["exact_mult"].iloc[0]

    training.run_step(2)

    # m = PipelineModel.load(spark_test_tmp_dir_path + "/chosen_model")
    p = training.link_run.trained_models["pre_pipeline"]
    m = training.link_run.trained_models["trained_model"]
    transformed_df = m.transform(
        p.transform(spark.table("training_features"))
    ).toPandas()
    row = transformed_df.query("id_a == 10 and id_b == 50").iloc[0]
    assert row.prediction == 0
    assert row.state_distance_imp.round(0) == 1909


def test_step_2_bucketizer(spark, main, conf):
    """Test a bucketized feature using spark pipeline function"""
    data = [
        (0.0, 0, 0),
        (3.0, 1, 1),
        (5.0, 2, 0),
        (6.0, 3, 1),
        (9.0, 4, 0),
        (10.0, 5, 1),
        (11.0, 6, 0),
        (23.0, 7, 1),
    ]
    dataFrame = spark.createDataFrame(data, ["immyear_abs_diff", "test_id", "match"])
    dataFrame.createOrReplaceTempView("training_features")

    conf["pipeline_features"] = [
        {
            "input_column": "immyear_abs_diff",
            "output_column": "immyear_caution",
            "transformer_type": "bucketizer",
            "categorical": True,
            "splits": [0, 6, 11, 999],
        }
    ]
    conf["training"] = {
        "dependent_var": "match",
        "independent_vars": ["immyear_abs_diff", "immyear_caution"],
    }
    conf["comparison_features"] = []

    ind_vars = conf["training"]["independent_vars"]
    tf = spark.table("training_features")
    pipeline_stages = pipeline_core.generate_pipeline_stages(
        conf, ind_vars, tf, "training"
    )
    prep_pipeline = Pipeline(stages=pipeline_stages)
    prep_model = prep_pipeline.fit(tf)
    prepped_data = prep_model.transform(tf)
    prepped_data = prepped_data.toPandas()

    assert prepped_data.shape == (8, 7)
    assert list(prepped_data.query("test_id == 0")["features_vector"].iloc[0]) == [
        0,
        1,
        0,
        0,
        0,
    ]
    assert list(prepped_data.query("test_id == 1")["features_vector"].iloc[0]) == [
        3,
        1,
        0,
        0,
        0,
    ]
    assert list(prepped_data.query("test_id == 3")["features_vector"].iloc[0]) == [
        6,
        0,
        1,
        0,
        0,
    ]
    assert list(prepped_data.query("test_id == 6")["features_vector"].iloc[0]) == [
        11,
        0,
        0,
        1,
        0,
    ]

    main.do_drop_all("")


def test_step_2_interaction(spark, main, conf):
    """Test interacting two and three features using spark pipeline function"""
    data = [
        (0.0, 0.0, 0.0),
        (3.0, 1.0, 1.0),
        (5.0, 2.0, 0.0),
        (6.0, 3.0, 1.0),
        (9.0, 4.0, 0.0),
        (10.0, 5.0, 1.0),
        (11.0, 6.0, 0.0),
        (23.0, 7.0, 1.0),
    ]
    dataFrame = spark.createDataFrame(data, ["var0", "var1", "var2"])
    dataFrame.createOrReplaceTempView("training_features")

    conf["pipeline_features"] = [
        {
            "input_columns": ["var0", "var1"],
            "output_column": "interacted_vars01",
            "transformer_type": "interaction",
        },
        {
            "input_columns": ["var0", "var1", "var2"],
            "output_column": "interacted_vars012",
            "transformer_type": "interaction",
        },
    ]

    conf["training"] = {
        "dependent_var": "var2",
        "independent_vars": ["interacted_vars01", "interacted_vars012"],
    }
    conf["comparison_features"] = []

    ind_vars = conf["training"]["independent_vars"]
    tf = spark.table("training_features")
    pipeline_stages = pipeline_core.generate_pipeline_stages(
        conf, ind_vars, tf, "training"
    )
    prep_pipeline = Pipeline(stages=pipeline_stages)
    prep_model = prep_pipeline.fit(tf)
    prepped_data = prep_model.transform(tf)
    prepped_data = prepped_data.toPandas()

    assert prepped_data.shape == (8, 8)
    assert prepped_data.query("var1 == 0")["interacted_vars01"].iloc[0][0] == 0
    assert prepped_data.query("var1 == 2")["interacted_vars01"].iloc[0][0] == 10
    assert prepped_data.query("var1 == 2")["interacted_vars012"].iloc[0][0] == 0
    assert prepped_data.query("var1 == 3")["interacted_vars01"].iloc[0][0] == 18
    assert prepped_data.query("var1 == 3")["interacted_vars012"].iloc[0][0] == 18

    main.do_drop_all("")
