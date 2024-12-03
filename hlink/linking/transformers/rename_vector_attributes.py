# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import logging

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, Param, Params, TypeConverters
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


class RenameVectorAttributes(Transformer, HasInputCol):
    """
    A custom transformer which renames the attributes or "slot names" of a
    given input column of type vector. This is helpful when you don't have
    complete control over the names of the attributes when they are created,
    but you still need them to look a certain way.

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
        input_col = self.getInputCol()
        to_replace = self.getOrDefault("strsToReplace")
        replacement_str = self.getOrDefault("replaceWith")
        metadata = dataset.schema[input_col].metadata

        logger.debug(
            f"Renaming the attributes of vector column '{input_col}': "
            f"replacing {to_replace} with '{replacement_str}'"
        )

        if "attrs" in metadata["ml_attr"]:
            attributes_by_type = metadata["ml_attr"]["attrs"]

            # The attributes are grouped by type, which may be numeric, binary, or
            # nominal. We don't care about the type here; we'll just rename all of
            # the attributes.
            for _attribute_type, attributes in attributes_by_type.items():
                for attribute in attributes:
                    for substring in to_replace:
                        attribute["name"] = attribute["name"].replace(
                            substring, replacement_str
                        )
        elif "vals" in metadata["ml_attr"]:
            values = metadata["ml_attr"]["vals"]

            for index in range(len(values)):
                for substring in to_replace:
                    values[index] = values[index].replace(substring, replacement_str)

        return dataset.withMetadata(input_col, metadata)
