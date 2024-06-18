# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from jinja2 import Environment, PackageLoader


def test_potential_matches_sql_template() -> None:
    loader = PackageLoader("hlink.linking.matching")
    jinja_env = Environment(loader=loader)
    template = jinja_env.get_template("potential_matches.sql")
    context = {
        "dataset_columns": ["AGE", "SEX"],
        "feature_columns": [],
        "blocking_columns": [["AGE_3"], ["SEX"]],
    }
    query = template.render(context).strip()
    query_lines = query.splitlines()
    query_lines_clean = [line.strip() for line in query_lines]

    assert query_lines_clean == [
        "SELECT DISTINCT",
        "",
        "a.AGE as AGE_a",
        ",b.AGE as AGE_b",
        "",
        ",a.SEX as SEX_a",
        ",b.SEX as SEX_b",
        "",
        "",
        "FROM exploded_df_a a",
        "JOIN exploded_df_b b ON",
        "",
        "(a.AGE_3 = b.AGE_3) AND",
        "",
        "(a.SEX = b.SEX)",
    ]


def test_potential_matches_sql_template_or_groups() -> None:
    loader = PackageLoader("hlink.linking.matching")
    jinja_env = Environment(loader=loader)
    template = jinja_env.get_template("potential_matches.sql")
    context = {
        "dataset_columns": ["AGE", "SEX", "BPL"],
        "feature_columns": [],
        "blocking_columns": [["AGE_3", "SEX"], ["BPL"]],
    }
    query = template.render(context).strip()
    query_lines = query.splitlines()
    query_lines_clean = [line.strip() for line in query_lines]

    assert query_lines_clean == [
        "SELECT DISTINCT",
        "",
        "a.AGE as AGE_a",
        ",b.AGE as AGE_b",
        "",
        ",a.SEX as SEX_a",
        ",b.SEX as SEX_b",
        "",
        ",a.BPL as BPL_a",
        ",b.BPL as BPL_b",
        "",
        "",
        "FROM exploded_df_a a",
        "JOIN exploded_df_b b ON",
        "",
        "(a.AGE_3 = b.AGE_3 OR a.SEX = b.SEX) AND",
        "",
        "(a.BPL = b.BPL)",
    ]
