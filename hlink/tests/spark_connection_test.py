from pathlib import Path
import re
import sys

from hlink.spark.session import SparkConnection


def test_app_name_defaults_to_linking(tmp_path: Path) -> None:
    derby_dir = tmp_path / "derby"
    warehouse_dir = tmp_path / "warehouse"
    checkpoint_dir = tmp_path / "checkpoint"
    tmp_dir = tmp_path / "tmp"
    connection = SparkConnection(
        derby_dir, warehouse_dir, checkpoint_dir, tmp_dir, sys.executable, "test"
    )
    spark = connection.local(cores=1, executor_memory="1G")
    app_name = spark.conf.get("spark.app.name")
    assert app_name == "linking"


def test_app_name_argument(tmp_path: Path) -> None:
    derby_dir = tmp_path / "derby"
    warehouse_dir = tmp_path / "warehouse"
    checkpoint_dir = tmp_path / "checkpoint"
    tmp_dir = tmp_path / "tmp"
    connection = SparkConnection(
        derby_dir,
        warehouse_dir,
        checkpoint_dir,
        tmp_dir,
        sys.executable,
        "test",
        app_name="test_app_name",
    )
    spark = connection.local(cores=1, executor_memory="1G")
    app_name = spark.conf.get("spark.app.name")
    assert app_name == "test_app_name"


def test_sets_checkpoint_directory(tmp_path: Path) -> None:
    derby_dir = tmp_path / "derby"
    warehouse_dir = tmp_path / "warehouse"
    checkpoint_dir = tmp_path / "checkpoint"
    tmp_dir = tmp_path / "tmp"
    connection = SparkConnection(
        derby_dir,
        warehouse_dir,
        checkpoint_dir,
        tmp_dir,
        sys.executable,
        "test",
    )
    spark = connection.local(cores=1, executor_memory="1G")

    spark_checkpoint_dir = spark.sparkContext.getCheckpointDir()
    assert re.search(str(checkpoint_dir), spark_checkpoint_dir)
