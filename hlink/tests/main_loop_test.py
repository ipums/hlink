# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import os
import pandas as pd
import pytest
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from hlink.linking.link_run import link_task_choices


@pytest.mark.quickcheck
def test_do_get_steps(capsys, main, spark):
    for task in link_task_choices:
        task_inst = getattr(main.link_run, task)
        steps = task_inst.get_steps()
        main.do_set_link_task(task)
        main.do_get_steps("")
        output = capsys.readouterr().out
        for step in steps:
            if str(step) not in output:
                print(type(step))
                print(step)
                print(output)
            assert str(step) in output


@pytest.mark.quickcheck
def test_do_set_link_task(capsys, main):
    main.current_link_task = main.link_run.matching
    main.do_set_link_task("preprocessing")
    assert main.current_link_task is main.link_run.preprocessing
    output = capsys.readouterr().out
    assert "preprocessing" in output.lower()


def test_output_csv_array_and_vector_data(
    main, preprocessing, spark, preprocessing_conf_household_data
):
    """Test if csv output works for array and vector data."""
    preprocessing_conf_household_data["feature_selections"] = [
        {
            "output_col": "namefrst_related",
            "input_col": "namefrst_clean",
            "transform": "related_individuals",
            "family_id": "serial",
            "relate_col": "relate",
            "top_code": 10,
            "bottom_code": 3,
        },
        {
            "input_column": "namelast_clean",
            "output_column": "namelast_clean_bigrams",
            "transform": "bigrams",
        },
        {
            "input_column": "namelast_clean",
            "output_column": "namelast_clean_soundex",
            "transform": "soundex",
        },
        {
            "input_column": "namefrst_orig",
            "output_column": "namefrst_orig_soundex",
            "transform": "soundex",
        },
        {
            "input_columns": ["namelast_clean_soundex", "namefrst_orig_soundex"],
            "output_column": "namelast_frst_soundex",
            "transform": "array",
        },
    ]

    preprocessing.run_step(0)
    preprocessing.run_step(1)

    data_a = preprocessing.spark.table("prepped_df_a")
    # data_a.withColumn('dense_vector_ex', Vectors.dense([0.0, 0.5, 0.6, 0.8]))

    encoder = OneHotEncoder(inputCols=["pernum"], outputCols=["pernum_onehotencoded"])
    model = encoder.fit(data_a)
    data_e = model.transform(data_a)

    assembler = VectorAssembler(
        inputCols=["bpl", "sex", "pernum_onehotencoded"], outputCol="feature_vector"
    )
    data_v = assembler.transform(data_e)

    preprocessing.run_register_python("prepped_df_v", lambda: data_v)

    current_dir = os.getcwd()
    output_path_v = os.path.join(current_dir, "output_data/array_vector_test.csv")

    main.do_csv(args=f"prepped_df_v {output_path_v}")

    assert os.path.isfile(output_path_v)

    prepped_v = pd.read_csv(output_path_v)
    assert prepped_v.shape == (58, 24)

    assert (
        prepped_v.query("namelast_orig == 'Beebe'")["feature_vector"].iloc[0]
        == "(7,[0,1,3],[10.0,2.0,1.0])"
    )
    assert (
        prepped_v.query("namelast_orig == 'Morgan'")["feature_vector"].iloc[0]
        == "(7,[0,1],[10.0,1.0])"
    )
    main.do_drop_all("")
    os.remove(output_path_v)


def test_crosswalk_reporting(
    main,
    capsys,
    spark,
    crosswalk_input_paths,
    crosswalk_validation_path,
    crosswalk_with_round_validation_path,
    tmp_path,
):
    (
        raw_df_a_path,
        raw_df_b_path,
        predicted_matches_path,
        hh_predicted_matches_path,
    ) = crosswalk_input_paths

    spark.read.csv(
        raw_df_a_path, header=True, inferSchema=True
    ).createOrReplaceTempView("raw_df_a")
    spark.read.csv(
        raw_df_b_path, header=True, inferSchema=True
    ).createOrReplaceTempView("raw_df_b")
    spark.read.csv(
        predicted_matches_path, header=True, inferSchema=True
    ).createOrReplaceTempView("predicted_matches")
    spark.read.csv(
        hh_predicted_matches_path, header=True, inferSchema=True
    ).createOrReplaceTempView("hh_predicted_matches")

    output_path = os.path.join(tmp_path, "crosswalk.csv")
    main.do_x_crosswalk(args=f"{output_path} histid,age")
    assert [row for row in open(output_path)] == [
        row for row in open(crosswalk_validation_path)
    ]

    output_path = os.path.join(tmp_path, "crosswalk_with_round.csv")
    print(f"Testing  {output_path} with round column")
    main.do_x_crosswalk(args=f"{output_path} --include-rounds histid,age")

    assert [row for row in open(output_path)] == [
        row for row in open(crosswalk_with_round_validation_path)
    ]
