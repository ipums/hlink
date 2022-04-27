# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable
from pyspark import keyword_only


class FloatCastTransformer(
    Transformer, HasInputCols, DefaultParamsReadable, DefaultParamsWritable
):
    """
    A custom Transformer which casts the input column to a float.
    """

    @keyword_only
    def __init__(self, inputCols=None):
        super(FloatCastTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        other_cols = set(df.columns) - set(self.getInputCols())
        casted_cols = [
            f"CAST({inputCol} as float) as {inputCol}"
            for inputCol in self.getInputCols()
        ]
        return df.selectExpr(list(other_cols) + casted_cols)
