# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink


def test_apply_chosen_model_RF(
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
    training_conf["spark_tmp_dir"] = spark_test_tmp_dir_path
    training_conf["drop_data_from_scored_matches"] = False

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

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "prediction"
        ].iloc()[0]
        == 1
    )


def test_apply_chosen_model_RF_pm_IDs_only(
    spark,
    training_conf,
    training,
    state_dist_path,
    training_data_path,
    potential_matches_path_ids_only,
    spark_test_tmp_dir_path,
    matching,
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
    ]

    training_conf["training"]["dataset"] = training_data_path
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
    training_conf["training"]["score_with_model"] = True
    training_conf["spark_tmp_dir"] = spark_test_tmp_dir_path
    training_conf["drop_data_from_scored_matches"] = True

    potential_matches = training.spark.read.csv(
        potential_matches_path_ids_only, header=True, inferSchema=True
    )
    potential_matches.write.mode("overwrite").saveAsTable("potential_matches")

    training.run_step(0)
    training.run_step(1)
    training.run_step(2)
    matching.run_step(2)

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    assert pm_df.shape == (9, 5)
    assert pm_df.query("id_a == 10 and id_b == 10")["prediction"].iloc()[0] == 1
    assert pm_df.query("id_a == 20 and id_b == 30")["prediction"].iloc()[0] == 1
    assert (
        round(
            list(
                pm_df.query("id_a == 10 and id_b == 50")["probability_array"].iloc()[0]
            )[0],
            0,
        )
        == 1
    )
    assert (
        round(
            list(
                pm_df.query("id_a == 10 and id_b == 50")["probability_array"].iloc()[0]
            )[1],
            0,
        )
        == 0
    )


def test_apply_chosen_model_probit_pm_IDs_only(
    spark,
    training_conf,
    training,
    state_dist_path,
    training_data_path,
    potential_matches_path_ids_only,
    spark_test_tmp_dir_path,
    matching,
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
    ]

    training_conf["training"]["dataset"] = training_data_path
    training_conf["training"]["dependent_var"] = "match"
    training_conf["training"]["independent_vars"] = [
        "namelast_jw",
        "regionf",
        "state_distance",
    ]
    training_conf["training"]["chosen_model"] = {"type": "probit", "threshold": 0.5}
    training_conf["training"]["score_with_model"] = True
    training_conf["spark_tmp_dir"] = spark_test_tmp_dir_path
    training_conf["drop_data_from_scored_matches"] = True

    potential_matches = training.spark.read.csv(
        potential_matches_path_ids_only, header=True, inferSchema=True
    )
    potential_matches.write.mode("overwrite").saveAsTable("potential_matches")

    training.run_step(0)
    training.run_step(1)
    training.run_step(2)
    matching.run_step(2)

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    assert pm_df.shape == (9, 4)
    assert pm_df.query("id_a == 10 and id_b == 10")["prediction"].iloc()[0] == 1
    assert pm_df.query("id_a == 20 and id_b == 30")["prediction"].iloc()[0] == 1
    assert (
        round(pm_df.query("id_a == 10 and id_b == 50")["probability"].iloc()[0], 0) == 0
    )
    assert (
        round(pm_df.query("id_a == 10 and id_b == 10")["probability"].iloc()[0], 0) == 1
    )


def test_apply_chosen_model_RF_pm_IDs_only_full_data_out(
    spark,
    training_conf,
    training,
    state_dist_path,
    training_data_path,
    potential_matches_path_ids_only,
    spark_test_tmp_dir_path,
    matching,
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
    ]

    training_conf["training"]["dataset"] = training_data_path
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
    training_conf["training"]["score_with_model"] = True
    training_conf["spark_tmp_dir"] = spark_test_tmp_dir_path
    training_conf["drop_data_from_scored_matches"] = False

    potential_matches = training.spark.read.csv(
        potential_matches_path_ids_only, header=True, inferSchema=True
    )
    potential_matches.write.mode("overwrite").saveAsTable("potential_matches")

    training.run_step(0)
    training.run_step(1)
    training.run_step(2)
    matching.run_step(2)

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    assert pm_df.shape == (9, 13)
    assert pm_df.query("id_a == 10 and id_b == 10")["prediction"].iloc()[0] == 1
    assert pm_df.query("id_a == 20 and id_b == 30")["prediction"].iloc()[0] == 1
    assert (
        round(
            list(
                pm_df.query("id_a == 10 and id_b == 50")["probability_array"].iloc()[0]
            )[0],
            0,
        )
        == 1
    )
    assert (
        round(
            list(
                pm_df.query("id_a == 10 and id_b == 50")["probability_array"].iloc()[0]
            )[1],
            0,
        )
        == 0
    )


def test_apply_chosen_model_probit(
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

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "prediction"
        ].iloc()[0]
        == 1
    )


def test_step_3_apply_chosen_model_logistic_regression(
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
    training_conf["training"]["chosen_model"] = {
        "type": "logistic_regression",
        "threshold": 0.8,
    }
    # training_conf["training"]["use_potential_matches_features"] = True
    training_conf["training"]["score_with_model"] = True
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

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "prediction"
        ].iloc()[0]
        == 1
    )


def test_step_3_apply_chosen_model_decision_tree(
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
    training_conf["training"]["chosen_model"] = {
        "type": "decision_tree",
        "maxDepth": 6.0,
        "minInstancesPerNode": 2.0,
        "maxBins": 4.0,
    }
    # training_conf["training"]["use_potential_matches_features"] = True
    training_conf["training"]["score_with_model"] = True
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

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "prediction"
        ].iloc()[0]
        == 1
    )


def test_step_3_apply_chosen_model_boosted_trees(
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
    training_conf["training"]["chosen_model"] = {
        "type": "gradient_boosted_trees",
        "maxDepth": 4.0,
        "minInstancesPerNode": 1.0,
        "maxBins": 6.0,
    }
    # training_conf["training"]["use_potential_matches_features"] = True
    training_conf["training"]["score_with_model"] = True
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

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    # assert "probability_array" not in list(pm_df.columns)
    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "prediction"
        ].iloc()[0]
        == 1
    )


def test_step_3_apply_chosen_model_RF_threshold(
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
    training_conf["training"]["decision"] = "drop_duplicate_with_threshold_ratio"
    training_conf["training"]["threshold_ratio"] = 1.3
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

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "prediction"
        ].iloc()[0]
        == 1
    )


def test_step_3_apply_chosen_model_probit_threshold(
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
    training_conf["training"]["decision"] = "drop_duplicate_with_threshold_ratio"
    training_conf["training"]["threshold_ratio"] = 1.3
    training_conf["training"]["independent_vars"] = [
        "namelast_jw",
        "regionf",
        "state_distance",
    ]
    training_conf["training"]["chosen_model"] = {"type": "probit", "threshold": 0.5}
    # training_conf["training"]["use_potential_matches_features"] = True
    training_conf["training"]["score_with_model"] = True
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

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "prediction"
        ].iloc()[0]
        == 1
    )


def test_step_3_apply_chosen_model_boosted_trees_threshold(
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
    training_conf["training"]["chosen_model"] = {
        "type": "gradient_boosted_trees",
        "maxDepth": 4.0,
        "minInstancesPerNode": 1.0,
        "maxBins": 6.0,
    }
    training_conf["training"]["decision"] = "drop_duplicate_with_threshold_ratio"
    training_conf["training"]["threshold_ratio"] = 1.3
    # training_conf["training"]["use_potential_matches_features"] = True
    training_conf["training"]["score_with_model"] = True
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

    pm_df = training.spark.table("scored_potential_matches").toPandas()

    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '0202928A-AC3E-48BB-8568-3372067F35C7'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_a == '81E992C0-3796-4BE7-B02E-9CAD0289C6EC'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "probability"
        ].iloc()[0]
        < 0.5
    )
    assert (
        pm_df.query("id_b == '033FD0FA-C523-42B5-976A-751E830F7021'")[
            "prediction"
        ].iloc()[0]
        == 0
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "probability"
        ].iloc()[0]
        > 0.5
    )
    assert (
        pm_df.query("id_b == '00849961-E52F-42F2-9B70-052606223052'")[
            "prediction"
        ].iloc()[0]
        == 1
    )
