# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable


class RenameProbColumn(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def _transform(self, dataset):
        return dataset.withColumnRenamed("probability", "probability_array").selectExpr(
            "*", "parseProbVector(probability_array, 1) as probability"
        )
