# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import os
import subprocess

from pyspark.sql.types import StringType

from hlink.scripts.lib.util import report_and_log_error


def write_table_to_csv(spark, table_name, output_path, num_partitions=None):
    """Write a spark table to csv.

    `num_partitions` can be used to partition the output, creating a directory
    with multiple csv files.

    Args:
        spark (SparkSession)
        table_name (str): the name of the table to write out
        output_path (str): the output path to write the csv to
        num_partitions (int, optional): How many partitions to use when writing the csv. Defaults to None.
    """
    df = spark.table(table_name)
    selects = []
    for col in df.schema:
        if col.dataType.typeName() == "array":
            selects.append(f"array_to_string({table_name}.{col.name})")
        elif col.dataType.typeName() == "vectorudt":
            selects.append(f"vector_to_string({table_name}.{col.name})")
        elif col.dataType.typeName() == "map":
            selects.append(f"CAST({col.name} as STRING) as {col.name}")
        else:
            selects.append(col.name)
    sql_selects = ",\n ".join(f for f in selects)

    col_names = [col.name for col in df.schema]

    spark.udf.registerJavaFunction(
        "array_to_string", "com.isrdi.udfs.ArrayToString", StringType()
    )
    spark.udf.registerJavaFunction(
        "vector_to_string", "com.isrdi.udfs.VectorToString", StringType()
    )

    df_selected = spark.sql(f"SELECT {sql_selects} FROM {table_name}")
    if num_partitions is not None:
        df_selected.repartition(num_partitions).write.csv(
            output_path, sep=",", header=True, quoteAll=True
        )
    else:
        output_tmp = output_path + ".tmp"
        df_selected.write.csv(output_tmp, sep=",", header=False, quoteAll=True)

        header = '"' + '","'.join(col_names) + '"'
        commands = [
            f"echo '{header}' > {output_path}",
            f"cat {output_tmp}/* >> {output_path} ",
            f"rm -rf {output_tmp}",
        ]
        for command in commands:
            subprocess.run(command, shell=True)


def read_csv_and_write_parquet(spark, csv_path, parquet_path):
    """Read in csv and write it out to parquet."""
    spark.read.csv(csv_path, header=True, nullValue="").na.fill("").write.parquet(
        parquet_path
    )


def load_external_table(spark, input_path, table_name):
    """Load an external datasource into spark as a table."""
    spark.catalog.createExternalTable(table_name, path=input_path)


def borrow_spark_tables(spark, borrow_tables_from):
    table_dirs = [f.path for f in os.scandir(borrow_tables_from) if f.is_dir()]

    for t in table_dirs:
        try:
            table_name = os.path.basename(t)
            print(f"Borrowing:\t{table_name}...\t\t\t", end="")

            spark.catalog.createTable(table_name, path=t)
            print("SUCCEEDED")
        except Exception as err:
            print("FAILED")
            report_and_log_error("Error borrowing " + table_name, err)
