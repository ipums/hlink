import pytest
from hlink.linking.table import Table
from pyspark.sql.types import StructType, StructField, StringType


@pytest.fixture()
def simple_schema():
    return StructType([StructField("test", StringType())])


@pytest.mark.quickcheck
@pytest.mark.parametrize("table_name", ["this_table_does_not_exist", "@@@", "LOL rofl"])
def test_exists_table_does_not_exist(spark, table_name):
    t = Table(spark, table_name, "table used for testing")
    assert not t.exists()


@pytest.mark.quickcheck
@pytest.mark.parametrize("table_name", ["table_for_testing_Table_class"])
def test_exists_table_does_exist(spark, table_name, simple_schema):
    t = Table(spark, table_name, "table used for testing")
    spark.catalog.createTable(table_name, schema=simple_schema)
    print([table.name for table in spark.catalog.listTables()])
    assert t.exists()
    spark.sql(f"DROP TABLE {table_name}")


@pytest.mark.quickcheck
@pytest.mark.parametrize("table_name", ["table_for_testing_Table_class"])
def test_drop_table_does_exist(spark, table_name, simple_schema):
    t = Table(spark, table_name, "table used for testing")
    spark.catalog.createTable(table_name, schema=simple_schema)
    assert t.exists()
    t.drop()
    assert not t.exists()


@pytest.mark.parametrize("table_name", ["this_table_does_not_exist", "@@@", "LOL rofl"])
def test_drop_table_does_not_exist(spark, table_name):
    # Check that dropping a table that doesn't exist doesn't throw errors
    # or somehow create the table.
    t = Table(spark, table_name, "table used for testing")
    assert not t.exists()
    t.drop()
    assert not t.exists()


@pytest.mark.quickcheck
@pytest.mark.parametrize("table_name", ["table_for_testing_Table_class"])
def test_df_table_does_exist(spark, table_name, simple_schema):
    t = Table(spark, table_name, "table used for testing")
    spark.catalog.createTable(table_name, schema=simple_schema)
    assert t.exists()
    assert t.df() is not None
    spark.sql(f"DROP TABLE {table_name}")


@pytest.mark.parametrize("table_name", ["this_table_does_not_exist", "@@@", "LOL rofl"])
def test_df_table_does_not_exist(spark, table_name):
    t = Table(spark, table_name, "table used for testing")
    assert t.df() is None


@pytest.mark.parametrize(
    "table_name", ["table_for_testing_Table_class", "camelCaseTable", "@@@", "LOL rofl"]
)
def test_name_is_unchanged(spark, table_name):
    t = Table(spark, table_name, "table used for testing")
    assert t.name == table_name
