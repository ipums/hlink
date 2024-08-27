from pathlib import Path

from pyspark.sql import Row

from hlink.spark.factory import SparkFactory


def test_spark_factory_can_create_spark_session(tmp_path: Path) -> None:
    derby_dir = tmp_path / "derby"
    spark_tmp_dir = tmp_path / "tmp"
    warehouse_dir = tmp_path / "warehouse"

    factory = (
        SparkFactory()
        .set_local()
        .set_derby_dir(derby_dir)
        .set_warehouse_dir(warehouse_dir)
        .set_tmp_dir(spark_tmp_dir)
        .set_num_cores(1)
        .set_executor_cores(1)
        .set_executor_memory("1G")
    )
    spark = factory.create()

    # Make sure we can do some basic operations with the SparkSession we get back
    df = spark.createDataFrame(
        [[0, "a"], [1, "b"], [2, "c"]], "id:integer,letter:string"
    )
    expr = (df.letter == "b").alias("equals_b")
    result = df.select(expr).collect()
    assert result == [
        Row(equals_b=False),
        Row(equals_b=True),
        Row(equals_b=False),
    ]
