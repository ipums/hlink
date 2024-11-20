from pyspark.ml import Transformer
from pyspark.sql import DataFrame


class RenameVectorAttributes(Transformer):
    """
    A custom transformer which renames the attributes or "slot names" of a
    given input column of type vector. This is helpful when you don't have
    complete control over the names of the attributes, but you need them to
    look a certain way.

    For example, LightGBM can't handle vector attributes with colons in their
    names. But the Spark Interaction class creates vector attributes named with
    colons. So we need to rename the attributes and remove the colons before
    passing the feature vector to LightGBM for training.
    """

    def __init__(self) -> None: ...

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset
