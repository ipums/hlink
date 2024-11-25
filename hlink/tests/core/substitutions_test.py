# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pathlib import Path

from pyspark.sql import Row, SparkSession

from hlink.linking.core.substitutions import generate_substitutions, _load_substitutions


def test_load_substitutions(tmp_path: Path) -> None:
    file_contents = """a,b
    to this,from this"""

    tmp_file = tmp_path / "substitutions.csv"
    tmp_file.write_text(file_contents)
    sub_froms, sub_tos = _load_substitutions(str(tmp_file))

    assert sub_froms == ["b", "from this"]
    assert sub_tos == ["a", "to this"]


def test_generate_substitutions(spark: SparkSession, tmp_path: Path) -> None:
    tmp_file = tmp_path / "substitutions.csv"
    tmp_file.write_text(
        """rose,rosie
        sophia,sophy
        sophia,sofia
        amanda,mandy
        jane,jean"""
    )

    df = spark.createDataFrame(
        [("agnes", 2), ("mandy", 2), ("sophy", 2), ("rosie", 2), ("jean", 1)],
        schema=["first_name", "sex"],
    )

    substitution_columns = [
        {
            "column_name": "first_name",
            "substitutions": [
                {
                    "join_column": "sex",
                    "join_value": 2,
                    "substitution_file": str(tmp_file),
                }
            ],
        }
    ]

    subbed_df = generate_substitutions(spark, df, substitution_columns)
    rows = subbed_df.select("first_name", "sex").collect()

    assert rows == [
        Row(first_name="agnes", sex=2),
        Row(first_name="amanda", sex=2),
        Row(first_name="sophia", sex=2),
        Row(first_name="rose", sex=2),
        # Note that this name is not substituted because we join on sex=2
        Row(first_name="jean", sex=1),
    ]
