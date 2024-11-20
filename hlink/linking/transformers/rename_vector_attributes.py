from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, Param, Params, TypeConverters
from pyspark.sql import DataFrame


class RenameVectorAttributes(Transformer, HasInputCol):
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

    strsToReplace: Param[list[str]] = Param(
        Params._dummy(),
        "strsToReplace",
        "Substrings to replace in the vector attribute names.",
        typeConverter=TypeConverters.toListString,
    )

    replaceWith: Param[str] = Param(
        Params._dummy(),
        "replaceWith",
        "The string to replace removed substrings.",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self,
        *,
        inputCol: str | None = None,
        strsToReplace: str | None = None,
        replaceWith: str | None = None,
    ) -> None:
        super(RenameVectorAttributes, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        *,
        inputCol: str | None = None,
        strsToReplace: str | None = None,
        replaceWith: str | None = None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        metadata = dataset.schema[self.getInputCol()].metadata
        attributes_by_type = metadata["ml_attr"]["attrs"]
        to_replace = self.getOrDefault("strsToReplace")
        replacement_str = self.getOrDefault("replaceWith")

        # The attributes are grouped by type, which may be numeric, binary, or
        # nominal. We don't care about the type here; we'll just rename all of
        # the attributes.
        for _attribute_type, attributes in attributes_by_type.items():
            for attribute in attributes:
                for substring in to_replace:
                    attribute["name"] = attribute["name"].replace(
                        substring, replacement_str
                    )

        return dataset
