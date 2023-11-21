from numpy import select
import pytest
import pandas as pd

from hlink.linking.core.column_mapping import select_column_mapping


TEST_DF_1 = pd.DataFrame(
    {
        "id": [0, 1, 2, 3, 4, 5],
        "age": [19, 37, 27, 101, 59, 22],
        "occupation": [
            "farmer",
            "computer scientist",
            "waitress",
            "retired",
            "lawyer",
            "doctor",
        ],
    }
)

TEST_DF_2 = pd.DataFrame(
    {
        "identifier": [1000, 1002, 1004, 1006],
        "age": [73, 55, 10, 18],
        "occ": ["retired", "childcare", None, None],
    }
)


@pytest.mark.parametrize("is_a", [True, False])
def test_select_column_mapping_basic(spark, is_a):
    """
    A single column mapping with just column_name specified selects that column
    from the dataframe.
    """
    column_mapping = {
        "column_name": "age",
    }
    df = spark.createDataFrame(TEST_DF_1)

    df_selected, column_selects = select_column_mapping(column_mapping, df, is_a, [])

    assert column_selects == ["age"]
    assert set(df_selected.columns) == {"age", "id", "occupation"}
    assert df_selected.count() == 6


@pytest.mark.parametrize("is_a", [True, False])
@pytest.mark.parametrize("alias", ["age", "myage", "num_years"])
def test_select_column_mapping_alias(spark, is_a, alias):
    """
    alias sets the output name for the column mapping. It can be the same as the
    input column name or different.
    """
    column_mapping = {
        "column_name": "age",
        "alias": alias,
    }
    df = spark.createDataFrame(TEST_DF_1)

    df_selected, column_selects = select_column_mapping(column_mapping, df, is_a, [])

    assert column_selects == [alias]
    # The alias is an additional column that is later selected out with column_selects
    assert set(df_selected.columns) == {"age", alias, "occupation", "id"}
    assert df_selected.count() == 6
    assert (
        df_selected.filter(df_selected.age == df_selected[alias]).count()
        == df_selected.count()
    )


def test_select_column_mapping_set_value_column_a(spark):
    """
    set_value_column_a overrides the input column for dataset A and sets all of its
    values to the given value. Dataset B is unaffected.
    """
    column_mapping = {
        "column_name": "age",
        "set_value_column_a": 44,
    }
    df_a = spark.createDataFrame(TEST_DF_1)
    df_b = spark.createDataFrame(TEST_DF_2)

    df_selected_a, column_selects_a = select_column_mapping(
        column_mapping, df_a, is_a=True, column_selects=[]
    )
    df_selected_b, column_selects_b = select_column_mapping(
        column_mapping, df_b, is_a=False, column_selects=[]
    )

    assert column_selects_a == column_selects_b == ["age"]

    assert (
        df_selected_a.filter(df_selected_a.age == 44).count() == df_selected_a.count()
    )
    assert df_selected_b.filter(df_selected_b.age == 44).count() == 0


def test_select_column_mapping_set_value_column_b(spark):
    """
    set_value_column_b overrides the input column for dataset B and sets all of its
    values to the given value. Dataset A is unaffected.
    """
    column_mapping = {
        "column_name": "age",
        "set_value_column_b": 44,
    }
    df_a = spark.createDataFrame(TEST_DF_1)
    df_b = spark.createDataFrame(TEST_DF_2)

    df_selected_a, column_selects_a = select_column_mapping(
        column_mapping, df_a, is_a=True, column_selects=[]
    )
    df_selected_b, column_selects_b = select_column_mapping(
        column_mapping, df_b, is_a=False, column_selects=[]
    )

    assert column_selects_a == column_selects_b == ["age"]

    assert (
        df_selected_b.filter(df_selected_b.age == 44).count() == df_selected_b.count()
    )
    assert df_selected_a.filter(df_selected_a.age == 44).count() == 0


def test_select_column_mapping_error_missing_column_name(spark):
    df = spark.createDataFrame(TEST_DF_1)
    with pytest.raises(KeyError):
        select_column_mapping({}, df, is_a=False, column_selects=[])
