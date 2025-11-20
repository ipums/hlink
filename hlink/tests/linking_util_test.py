import pytest

from hlink.linking.util import set_job_description, spark_shuffle_partitions_heuristic


@pytest.mark.parametrize(
    "dataset_size,expected_output",
    [(1, 200), (10001033, 401), (140000000, 5600), (2700000000, 10000)],
)
def test_spark_shuffle_partitions_heuristic(dataset_size, expected_output):
    output = spark_shuffle_partitions_heuristic(dataset_size)
    assert output == expected_output


def test_set_job_description(spark):
    with set_job_description("my description", spark.sparkContext):
        desc = spark.sparkContext.getLocalProperty("spark.job.description")
        assert desc == "my description"


def test_set_job_description_resets_on_error(spark):
    spark.sparkContext.setJobDescription(None)

    try:
        with set_job_description("my description", spark.sparkContext):
            raise Exception()
    except Exception:
        ...

    assert spark.sparkContext.getLocalProperty("spark.job.description") is None


def test_set_job_description_nested(spark):
    with set_job_description("outer description", spark.sparkContext):
        with set_job_description("inner description", spark.sparkContext):
            desc = spark.sparkContext.getLocalProperty("spark.job.description")
            assert desc == "inner description"
        desc = spark.sparkContext.getLocalProperty("spark.job.description")
        assert desc == "outer description"
