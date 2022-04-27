# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink


def test_men_only(spark, blocking_explode_conf, matching_test_input, matching, main):
    """Exclude women from potential matches."""
    table_a, table_b = matching_test_input
    table_a.createOrReplaceTempView("prepped_df_a")
    table_b.createOrReplaceTempView("prepped_df_b")

    # Get the number of men and women; the numbers in the
    # potential matching universe should be <= men - women
    prepped_a = spark.table("prepped_df_a").toPandas()
    prepped_b = spark.table("prepped_df_b").toPandas()

    men_a = len(prepped_a.query("sex == 1"))
    men_b = len(prepped_b.query("sex == 1"))
    women_a = len(prepped_a.query("sex == 2"))
    women_b = len(prepped_b.query("sex == 2"))

    unknown_a = len(prepped_a) - (men_a + women_a)
    unknown_b = len(prepped_b) - (men_b + women_b)

    # There can be unknown SEX values
    assert (unknown_a + men_a + women_a) == len(prepped_a)
    assert (unknown_b + men_b + women_b) == len(prepped_b)

    # For the test setup to be valid there must be women to
    # start with in the fixtures
    assert women_b > 0
    assert women_a > 0

    # Limit the universe to just men.
    blocking_explode_conf["potential_matches_universe"] = [{"expression": "sex == 1"}]

    matching.run_step(0)

    # The explode step will take the spark versions of these as inputs
    univ_a = spark.table("match_universe_df_a").toPandas()
    univ_b = spark.table("match_universe_df_b").toPandas()

    # The exploded step will produce these tables
    exploded_univ_a = spark.table("exploded_df_a").toPandas()
    exploded_univ_b = spark.table("exploded_df_b").toPandas()

    assert all(
        elem in list(exploded_univ_a.columns)
        for elem in ["namefrst", "namelast", "sex", "birthyr_3"]
    )
    assert all(
        elem in list(exploded_univ_b.columns)
        for elem in ["namefrst", "namelast", "sex", "birthyr_3"]
    )

    # Check there are no women in either universe.
    assert len(univ_a.query("sex == 2")) == 0
    assert len(univ_b.query("sex == 2")) == 0

    # Shows there aren't any unknown values of SEX
    assert men_a == len(univ_a)
    assert men_b == len(univ_b)

    assert men_a == len(univ_a.query("sex == 1"))
    assert men_b == len(univ_b.query("sex == 1"))

    # Check that there are no women in the exploded rows
    assert len(exploded_univ_a.query("sex == 2")) == 0
    assert len(exploded_univ_b.query("sex == 2")) == 0
