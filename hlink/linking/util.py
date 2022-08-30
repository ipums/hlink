from math import ceil


MIN_PARTITIONS = 200
MAX_PARTITIONS = 10000


def spark_shuffle_partitions_heuristic(dataset_size):
    """Calculate how many partitions to request from Spark based on dataset size.

    This is a heuristic / approximation of how many partitions should be requested
    from Spark so that hlink performs well. The minimum number of partitions
    returned is 200, the default for Spark. The maximum number returned is 10,000.
    """
    partitions_approx = ceil(dataset_size / 25000)
    clamped_below = max(MIN_PARTITIONS, partitions_approx)
    clamped = min(MAX_PARTITIONS, clamped_below)
    return clamped
