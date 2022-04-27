# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.sql.types import StructType, StructField, StringType
import pyspark.sql.functions as pyspark_funcs
import pandas as pd


def run_and_show_sql(spark, sql, limit=100, truncate=True):
    """Run the given sql query and print the results.

    Args:
        spark (SparkSession)
        sql (str): the sql query to run
        limit (int, optional): The maximum number of rows to show. Defaults to 100.
        truncate (bool, optional): Whether to shorten long strings in the output. Defaults to True.
    """
    spark.sql(sql).show(limit, truncate=truncate)


def show_table_row_count(spark, table_name):
    spark.sql(f"SELECT COUNT(*) FROM {table_name}").show()


def show_table_columns(spark, table_name):
    spark.sql(f"DESCRIBE {table_name}").show(1000, truncate=False)


def show_column_summary(spark, table_name, col_name):
    spark.table(table_name).select(col_name).summary().show()


def show_column_tab(spark, table_name, col_name):
    """Print a tabulation of the given column in the given table."""
    spark.table(table_name).groupBy(col_name).count().orderBy(col_name).show(
        100, truncate=False
    )


def show_table(spark, table_name, limit=10, truncate=True):
    """Print the first `limit` rows of the table with the given name.

    Args:
        spark (SparkSession)
        table_name (str): the name of the table to show
        limit (int, optional): How many rows of the table to show. Defaults to 10.
        truncate (bool, optional): Whether to truncate long strings in the output or not. Defaults to True.
    """
    spark.sql(f"SELECT * FROM {table_name}").show(limit, truncate=truncate)


def list_tables(link_run, list_all=False):
    """Print some information on the currently existing spark tables.

    Args:
        link_run (LinkRun)
        list_all (bool, optional): Whether to show all tables, or just those marked as important by the link run. Defaults to False.
    """
    schema = StructType(
        [
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
        ]
    )
    table_descs = {table.name: table.desc for table in link_run.known_tables.values()}
    pd_df1 = pd.DataFrame.from_dict(table_descs, orient="index")
    pd_df1.reset_index(inplace=True)
    pd_df1.rename(columns={"index": "name", 0: "description"}, inplace=True)
    df1 = link_run.spark.createDataFrame(pd_df1, schema)
    df2 = link_run.spark.sql("SHOW tables")
    if not list_all:
        # Print only important tables
        important_tables = [
            table
            for table in link_run.known_tables
            if not link_run.known_tables[table].hide
        ]
        df2 = df2.filter(df2["tableName"].isin(important_tables))
    df2.join(df1, df2.tableName == df1.name, "left").orderBy(
        "description", "tableName"
    ).drop("name").show(1000, truncate=False)


def drop_table(link_run, table_name):
    table = link_run.get_table(table_name)

    if table.exists():
        print(f"Dropping {table.name}")
        table.drop()
    else:
        print(f"Table {table.name} doesn't exist; no need to drop")


def drop_tables_satisfying(link_run, cond):
    """Drop all spark tables satisfying the given condition.

    `cond` is passed spark table info objects as returned by `spark.catalog.listTables()`.
    Tables for which `cond` evaluates to True will be dropped.

    Args:
        link_run (LinkRun)
        cond (spark table -> bool): filtering function to determine which tables should be dropped
    """
    all_tables = link_run.spark.catalog.listTables()
    satis_tables = filter(cond, all_tables)

    for table in satis_tables:
        print(f"Dropping {table.name}")
        link_run.get_table(table.name).drop()


def drop_all_tables(link_run):
    drop_tables_satisfying(link_run, (lambda _: True))


def drop_prc_tables(link_run):
    """Drop all precision_recall_curve-related tables."""
    drop_tables_satisfying(link_run, (lambda t: "precision_recall_curve" in t.name))


def persist_table(spark, table_name):
    """Make the given table permanent."""
    spark.table(table_name).write.mode("overwrite").saveAsTable(table_name)


def take_table_union(spark, table1_name, table2_name, output_table_name, mark_col_name):
    """Create the union of two tables as a new temporary table.

    Args:
        spark (SparkSession)
        table1_name (str): the name of the first table
        table2_name (str): the name of the second table
        output_table_name (str): the name of the destination table
        mark_col_name (str): the name of the column used to mark which table the row came from
    """
    t1 = spark.table(table1_name).withColumn(mark_col_name, pyspark_funcs.lit(True))
    t2 = spark.table(table2_name).withColumn(mark_col_name, pyspark_funcs.lit(False))
    new_cols = list(set(t1.columns).intersection(t2.columns))
    t1.select(new_cols).unionByName(t2.select(new_cols)).createOrReplaceTempView(
        output_table_name
    )
