# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pytest


def test_jw_empty_strings(spark):
    [row] = spark.sql("SELECT jw('', '') AS jw_result").collect()
    assert row.jw_result < 0.0001


@pytest.mark.parametrize("left,right", [("z", "z"), ("ambulate", "ambulate")])
def test_jw_matching_strings(spark, left: str, right: str):
    [row] = spark.sql(f"SELECT jw('{left}', '{right}') AS jw_result").collect()
    assert row.jw_result > 0.9999


@pytest.mark.parametrize("left,right", [("zzzzzz", "sleepy"), ("no", "shared letters")])
def test_jw_completely_different_strings(spark, left: str, right: str):
    [row] = spark.sql(f"SELECT jw('{left}', '{right}') AS jw_result").collect()
    assert row.jw_result < 0.0001


@pytest.mark.parametrize("left,right", [("", "discombobulated"), ("GlyphidMenace", "")])
def test_jw_one_empty_string(spark, left: str, right: str):
    [row] = spark.sql(f"SELECT jw('{left}', '{right}') AS jw_result").collect()
    assert row.jw_result < 0.0001
