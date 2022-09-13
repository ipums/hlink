import pytest

from hlink.linking.util import spark_shuffle_partitions_heuristic


@pytest.mark.parametrize(
    "dataset_size,expected_output",
    [(1, 200), (10001033, 401), (140000000, 5600), (2700000000, 10000)],
)
def test_spark_shuffle_partitions_heuristic(dataset_size, expected_output):
    output = spark_shuffle_partitions_heuristic(dataset_size)
    assert output == expected_output
