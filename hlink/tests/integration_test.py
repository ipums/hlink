# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink


def test_input_args_preprocessing(spark, main, integration_conf):
    main.link_run.config = integration_conf
    main.do_run_all_steps("preprocessing training matching")

    scored_matches = main.spark.table("scored_potential_matches").toPandas()
    row = scored_matches.query("id_a == 10 and id_b == 10").iloc[0]

    assert all(
        elem not in list(scored_matches.columns)
        for elem in [
            "region_a",
            "region_b",
            "age_a",
            "age_b",
            "serialp_a",
            "serialp_b",
            "bpl_a",
            "bpl_b",
        ]
    )
    assert all(
        elem in list(scored_matches.columns)
        for elem in [
            "id_a",
            "id_b",
            "namelast_jw",
            "regionf",
            "hits",
            "sex_equals",
            "namelast_jw_imp",
            "sex_equals_imp",
            "hits_imp",
            "regionf_onehotencoded",
            "sex_regionf_interaction",
            "features_vector",
            "rawPrediction",
            "probability_array",
            "probability",
            "second_best_prob",
            "ratio",
            "prediction",
        ]
    )
    assert row.probability.round(2) > 0.6
    assert row.prediction == 1
