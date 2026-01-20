from contextlib import contextmanager
from math import ceil

MIN_PARTITIONS = 200
MAX_PARTITIONS = 10000


def spark_shuffle_partitions_heuristic(dataset_size: int) -> int:
    """Calculate how many partitions to request from Spark based on dataset size.

    This is a heuristic / approximation of how many partitions should be requested
    from Spark so that hlink performs well. The minimum number of partitions
    returned is 200, the default for Spark. The maximum number returned is 10,000.
    """
    partitions_approx = ceil(dataset_size / 25000)
    clamped_below = max(MIN_PARTITIONS, partitions_approx)
    clamped = min(MAX_PARTITIONS, clamped_below)
    return clamped


@contextmanager
def set_job_description(desc: str | None, spark_context):
    """Set the Spark job description.

    This context manager sets the Spark job description to the given string,
    then restores the job description to its previous value on exit. Passing
    desc=None resets the job description to the Spark default.
    """
    previous_desc = spark_context.getLocalProperty("spark.job.description")
    spark_context.setJobDescription(desc)
    try:
        yield
    finally:
        spark_context.setJobDescription(previous_desc)
