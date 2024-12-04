# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from collections import namedtuple
from typing import Any

from pyspark import SparkContext
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import concat_ws, lit, regexp_replace, split, when


def generate_substitutions(
    spark: SparkSession,
    df_selected: DataFrame,
    substitution_columns: list[dict[str, Any]],
) -> DataFrame:
    for substitution_column in substitution_columns:
        column_name = substitution_column["column_name"]
        for substitution in substitution_column["substitutions"]:
            if (
                "regex_word_replace" in substitution
                and substitution["regex_word_replace"]
            ):
                df_selected = _apply_regex_substitution(
                    df_selected, column_name, substitution, spark.sparkContext
                )
            elif "substitution_file" in substitution:
                df_selected = _apply_substitution(
                    df_selected, column_name, substitution, spark.sparkContext
                )
            else:
                raise KeyError(
                    "You must supply a substitution file and either specify regex_word_replace=true or supply a join value."
                )
    return df_selected


def _load_substitutions(file_name: str) -> tuple[list[str], list[str]]:
    """Reads in the substitution file and returns a 2-tuple representing it.

    Parameters
    ----------
    file_name: name of substitution file

    Returns
    ----------
    A 2-tuple where the first value is an array of values to be replaced and the second is an array of values to use
    when replacing words in the first array.
    """
    sub_froms = []
    sub_tos = []
    with open(file_name, mode="r", encoding="utf-8-sig") as f:
        for line in f:
            sub_to, sub_from = line.strip().lower().split(",")
            sub_froms.append(sub_from)
            sub_tos.append(sub_to)
    return (sub_froms, sub_tos)


def _apply_substitution(
    df: DataFrame, column_name: str, substitution: dict[str, Any], sc: SparkContext
) -> DataFrame:
    """Returns a new df with the values in the column column_name replaced using substitutions defined in substitution_file."""
    substitution_file = substitution["substitution_file"]
    join_value = substitution["join_value"]
    join_column = substitution["join_column"]
    join_column_alias = join_column + "_sub"
    sub_froms, sub_tos = _load_substitutions(substitution_file)
    subs = list(zip(sub_froms, sub_tos))
    Sub = namedtuple("Sub", ["sub_from", "sub_to"])
    sub_df = (
        sc.parallelize(subs, 1)
        .map(lambda s: Sub(s[0], s[1]))
        .toDF()
        .withColumn(join_column_alias, lit(join_value))
    )
    join_statement = (sub_df["sub_from"] == split(df[column_name], " ")[0]) & (
        sub_df[join_column_alias] == df[join_column]
    )
    df_sub = df.join(sub_df.hint("broadcast"), join_statement, "left_outer").drop(
        "join_column_alias"
    )
    df_sub_select = (
        when(df_sub["sub_to"].isNull(), df_sub[column_name])
        .otherwise(concat_ws(" ", df_sub["sub_to"], split(df_sub[column_name], " ")[1]))
        .alias(column_name)
    )
    df_sub_selects = list(set(df.columns) - set([column_name])) + [df_sub_select]
    return df_sub.select(df_sub_selects)


def _apply_regex_substitution(
    df: DataFrame, column_name: str, substitution: dict[str, Any], sc: SparkContext
) -> DataFrame:
    """Returns a new df with the values in the column column_name replaced using substitutions defined in substitution_file."""

    substitution_file = substitution["substitution_file"]
    sub_froms, sub_tos = _load_substitutions(substitution_file)
    subs = dict(zip(sub_froms, sub_tos))
    col = column_name
    df.checkpoint()

    for sub_from, sub_to in subs.items():
        col = regexp_replace(
            col, r"(?:(?<=\s)|(?<=^))(" + sub_from + r")(?:(?=\s)|(?=$))", sub_to
        )
    return df.withColumn(column_name, col)
