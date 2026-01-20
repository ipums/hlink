from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, lit, reverse
import pytest
import pandas as pd

from hlink.linking.core.column_mapping import apply_transform, select_column_mapping

TEST_DF_1 = pd.DataFrame(
    {
        "id": [0, 1, 2, 3, 4, 5],
        "age": [19, 37, 27, 101, 59, 22],
        "occupation": [
            "FARMER",
            "COMPUTER SCIENTIST",
            "WAITRESS",
            "RETIRED",
            "LAWYER",
            "DOCTOR",
        ],
    }
)

TEST_DF_2 = pd.DataFrame(
    {
        "identifier": [1000, 1002, 1004, 1006],
        "age": [73, 55, 10, 18],
        "occ": ["RETIRED", "CHILDCARE", None, None],
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


def test_select_column_mapping_transforms_add_to_a(spark):
    """
    column_mappings support transforms, which modify the values of the column as
    it is read in. These often apply to both dataset A and dataset B, but some
    apply only to a single dataset.

    add_to_a applies only to dataset A.
    """
    column_mapping = {
        "column_name": "age",
        "alias": "age_at_dataset_b",
        "transforms": [{"type": "add_to_a", "value": 11}],
    }
    df_a = spark.createDataFrame(TEST_DF_1)
    df_b = spark.createDataFrame(TEST_DF_2)

    df_selected_a, column_selects_a = select_column_mapping(
        column_mapping,
        df_a,
        is_a=True,
        column_selects=[],
    )
    df_selected_b, column_selects_b = select_column_mapping(
        column_mapping,
        df_b,
        is_a=False,
        column_selects=[],
    )

    assert column_selects_a == column_selects_b == ["age_at_dataset_b"]
    ages_a = df_selected_a.select("age_at_dataset_b").toPandas()
    assert ages_a["age_at_dataset_b"].to_list() == [30, 48, 38, 112, 70, 33]

    ages_b = df_selected_b.select("age_at_dataset_b").toPandas()
    assert ages_b["age_at_dataset_b"].to_list() == [73, 55, 10, 18]


def test_select_column_mapping_column_selects_preserved(spark):
    """
    select_column_mapping() appends column names to the end of column_selects and
    then returns the new, longer list. You can even map the same column multiple
    times with different aliases.
    """
    column_mapping_1 = {
        "column_name": "occupation",
        "alias": "occ_with_underscores",
        "transforms": [{"type": "concat_two_cols", "column_to_append": "age"}],
    }
    column_mapping_2 = {
        "column_name": "occupation",
    }
    df_a = spark.createDataFrame(TEST_DF_1)

    df_selected, column_selects = select_column_mapping(
        column_mapping_1,
        df_a,
        is_a=True,
        column_selects=[],
    )

    assert column_selects == ["occ_with_underscores"]

    df_selected, column_selects = select_column_mapping(
        column_mapping_2, df_a, is_a=True, column_selects=column_selects
    )

    assert set(column_selects) == {"occ_with_underscores", "occupation"}


def test_select_column_mapping_override_column_a(spark):
    """
    override_column_a lets the user specify a different column name for
    dataset A. override_transforms are applied only to dataset A in
    this case.
    """
    column_mapping = {
        "column_name": "occ",
        "override_column_a": "occupation",
        "override_transforms": [{"type": "lowercase_strip"}],
    }
    df_a = spark.createDataFrame(TEST_DF_1)
    df_b = spark.createDataFrame(TEST_DF_2)

    df_selected_a, column_selects_a = select_column_mapping(
        column_mapping,
        df_a,
        is_a=True,
        column_selects=[],
    )

    df_selected_b, column_selects_b = select_column_mapping(
        column_mapping,
        df_b,
        is_a=False,
        column_selects=[],
    )

    assert column_selects_a == column_selects_b == ["occ"]

    occ_a = df_selected_a.select("occ").toPandas()
    assert occ_a["occ"].to_list() == [
        "farmer",
        "computer scientist",
        "waitress",
        "retired",
        "lawyer",
        "doctor",
    ]

    occ_b = df_selected_b.select("occ").toPandas()
    assert occ_b["occ"].to_list() == [
        "RETIRED",
        "CHILDCARE",
        None,
        None,
    ]


def test_select_column_mapping_override_column_b(spark):
    """
    override_column_b lets the user specify a different column name for
    dataset B. override_transforms are applied only to dataset B in
    this case, and transforms are applied only to dataset A.
    """
    column_mapping = {
        "column_name": "occupation",
        "override_column_b": "occ",
        "override_transforms": [
            {"type": "concat_two_cols", "column_to_append": "identifier"}
        ],
        "transforms": [
            {"type": "lowercase_strip"},
            {"type": "concat_two_cols", "column_to_append": "id"},
        ],
    }
    df_a = spark.createDataFrame(TEST_DF_1)
    df_b = spark.createDataFrame(TEST_DF_2)

    df_selected_a, column_selects_a = select_column_mapping(
        column_mapping,
        df_a,
        is_a=True,
        column_selects=[],
    )

    df_selected_b, column_selects_b = select_column_mapping(
        column_mapping,
        df_b,
        is_a=False,
        column_selects=[],
    )

    assert column_selects_a == column_selects_b == ["occupation"]

    occ_a = df_selected_a.select("occupation").toPandas()
    assert occ_a["occupation"].to_list() == [
        "farmer0",
        "computer scientist1",
        "waitress2",
        "retired3",
        "lawyer4",
        "doctor5",
    ]

    occ_b = df_selected_b.select("occupation").toPandas()
    assert occ_b["occupation"].to_list() == [
        "RETIRED1000",
        "CHILDCARE1002",
        None,
        None,
    ]


def test_select_column_mapping_error_missing_column_name(spark):
    """
    Without a column_name key in the column_mapping, the function raises
    a KeyError.
    """
    df = spark.createDataFrame(TEST_DF_1)
    with pytest.raises(KeyError):
        select_column_mapping({}, df, is_a=False, column_selects=[])


@pytest.mark.parametrize("is_a", [True, False])
def test_select_column_mapping_custom_transform(spark, is_a) -> None:
    df = spark.createDataFrame(TEST_DF_1)

    def transform_reverse(input_col, transform, context):
        return reverse(input_col)

    column_mapping = {
        "column_name": "occupation",
        "alias": "occupation_reversed",
        "transforms": [
            {"type": "reverse"},
        ],
    }

    df_selected, _ = select_column_mapping(
        column_mapping, df, is_a, [], custom_transforms={"reverse": transform_reverse}
    )

    rows = df_selected.sort("id").select("occupation_reversed").collect()
    assert rows == [
        Row(occupation_reversed="REMRAF"),
        Row(occupation_reversed="TSITNEICS RETUPMOC"),
        Row(occupation_reversed="SSERTIAW"),
        Row(occupation_reversed="DERITER"),
        Row(occupation_reversed="REYWAL"),
        Row(occupation_reversed="ROTCOD"),
    ]


@pytest.mark.parametrize("is_a", [True, False])
def test_select_column_mapping_custom_transform_override(spark, is_a) -> None:
    """
    Custom column mapping transforms override the built-in transforms. This
    means that adding more built-in transforms does not break user code that
    already has custom transforms with the same name.
    """
    column_mapping = {
        "column_name": "occupation",
        "alias": "overriden",
        "transforms": [{"type": "lowercase_strip"}],
    }

    def transform_lowercase_strip(input_col, transform, context):
        return lit("***")

    df = spark.createDataFrame(TEST_DF_1)
    df_selected, _ = select_column_mapping(
        column_mapping,
        df,
        is_a,
        [],
        custom_transforms={"lowercase_strip": transform_lowercase_strip},
    )

    rows = df_selected.sort("id").select("overriden").collect()
    assert rows == [
        Row(overriden="***"),
        Row(overriden="***"),
        Row(overriden="***"),
        Row(overriden="***"),
        Row(overriden="***"),
        Row(overriden="***"),
    ]


@pytest.mark.parametrize("is_a", [True, False])
def test_apply_transform_helpful_missing_attribute_error(spark, is_a) -> None:
    transform = {"type": "remove_stop_words"}
    with pytest.raises(ValueError, match="Missing required attribute 'values'"):
        apply_transform(col("testing"), transform, is_a)


@pytest.mark.parametrize("is_a", [True, False])
def test_apply_transform_when_value(spark: SparkSession, is_a: bool) -> None:
    """The when_value transform supports simple if-then-otherwise logic on
    columns:

    if the column is equal to "when_value"
    then return "if_value"
    otherwise return "else_value"
    """
    transform = {"type": "when_value", "value": 6, "if_value": 0, "else_value": 1}
    column_select = col("marst")
    output_col = apply_transform(column_select, transform, is_a)

    df = spark.createDataFrame([[3], [6], [2], [6], [1]], "marst:integer")
    transformed = df.select("marst", output_col.alias("output"))
    result = transformed.collect()

    assert result == [
        Row(marst=3, output=1),
        Row(marst=6, output=0),
        Row(marst=2, output=1),
        Row(marst=6, output=0),
        Row(marst=1, output=1),
    ]


@pytest.mark.parametrize("is_a", [True, False])
def test_apply_transform_remove_punctuation(spark: SparkSession, is_a: bool) -> None:
    transform = {"type": "remove_punctuation"}
    input_col = col("input")
    output_col = apply_transform(input_col, transform, is_a)

    df = spark.createDataFrame(
        [
            # All of these characters are considered punctuation and should be removed
            ["?-\\/\"':,.[]{}"],
            ["abcdefghijklmnop"],
            # The address of the Minnesota state capitol
            ["75 Rev. Dr. Martin Luther King, Jr. Blvd. Saint Paul, MN 55155"],
        ],
        "input:string",
    )
    transformed = df.select(output_col.alias("output"))
    result = transformed.collect()

    assert result == [
        Row(output=""),
        Row(output="abcdefghijklmnop"),
        Row(output="75 Rev Dr Martin Luther King Jr Blvd Saint Paul MN 55155"),
    ]


@pytest.mark.parametrize("values", [[1], [1, 2, 3]])
@pytest.mark.parametrize("is_a", [True, False])
def test_apply_transform_substring_error_when_not_exactly_2_values(
    values: list[int], is_a: bool
) -> None:
    """
    The substring transform takes a list of exactly two values, which are the
    start position of the substring and its length. If the list has the wrong
    number of values, then apply_transform() raises an error.

    TODO: It would be simpler to have two separate attributes for the substring
    start and length, like this:

    {
        "type": "substring",
        "start_index": 0,
        "length": 4,
    }

    See issue #146. Making these changes would eliminate the need for this
    test.
    """
    input_col = col("input")
    transform = {"type": "substring", "values": values}

    with pytest.raises(ValueError, match="Length of substr transform should be 2"):
        apply_transform(input_col, transform, is_a)


@pytest.mark.parametrize("is_a", [True, False])
def test_apply_transform_error_when_unrecognized_transform_type(is_a: bool) -> None:
    column_select = col("test")
    transform = {"type": "not_supported"}
    with pytest.raises(ValueError, match="Invalid transform type"):
        apply_transform(column_select, transform, is_a)


@pytest.mark.parametrize("is_a", [True, False])
def test_apply_transform_mapping(spark: SparkSession, is_a: bool) -> None:
    transform = {"type": "mapping", "mappings": {"first": "abcd", "second": "efg"}}
    input_col = col("input")
    output_col = apply_transform(input_col, transform, is_a)

    df = spark.createDataFrame(
        [
            ["first"],
            ["second"],
            ["third"],
            ["secondagain"],
        ],
        "input:string",
    )

    transformed = df.select(output_col.alias("output"))
    rows = transformed.collect()

    # Note that the mapping must exactly match the value to transform it, so the
    # value "secondagain" is unchanged.
    assert rows == [
        Row(output="abcd"),
        Row(output="efg"),
        Row(output="third"),
        Row(output="secondagain"),
    ]


@pytest.mark.parametrize("is_a", [True, False])
def test_apply_transform_mapping_integer_column(
    spark: SparkSession, is_a: bool
) -> None:
    """
    The mapping transform works over integer columns, and you can cast the output
    to an integer by passing output_type = "int".
    """
    transform = {
        "type": "mapping",
        "mappings": {"1": "10", "2": "30", "3": ""},
        "output_type": "int",
    }
    input_col = col("input")
    output_col = apply_transform(input_col, transform, is_a)

    df = spark.createDataFrame(
        [
            [5],
            [4],
            [3],
            [2],
            [1],
        ],
        "input:integer",
    )

    transformed = df.select(output_col.alias("output"))
    rows = transformed.collect()

    assert rows == [
        Row(output=5),
        Row(output=4),
        Row(output=None),
        Row(output=30),
        Row(output=10),
    ]
