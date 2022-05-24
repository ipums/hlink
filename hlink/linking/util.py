from math import ceil


def spark_shuffle_partitions_heuristic(dataset_size):
    """Calculate how many partitions to request from Spark based on dataset size.

    This is a heuristic / approximation of how many partitions should be requested
    from Spark so that hlink performs well. The minimum number of partitions
    returned is 200, the default for Spark.
    """
    partitions_approx = ceil(dataset_size / 25000)
    return max(200, partitions_approx)
