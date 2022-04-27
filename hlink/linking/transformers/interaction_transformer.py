# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark import keyword_only
from pyspark.ml.wrapper import JavaTransformer


class InteractionTransformer(
    JavaTransformer, HasInputCols, HasOutputCol, JavaMLReadable, JavaMLWritable
):
    """
    from https://github.com/apache/spark/commit/5bf5d9d854db53541956dedb03e2de8eecf65b81:
    Implements the feature interaction transform. This transformer takes in Double and Vector type
    columns and outputs a flattened vector of their feature interactions. To handle interaction,
    we first one-hot encode any nominal features. Then, a vector of the feature cross-products is
    produced.
    For example, given the input feature values `Double(2)` and `Vector(3, 4)`, the output would be
    `Vector(6, 8)` if all input features were numeric. If the first feature was instead nominal
    with four categories, the output would then be `Vector(0, 0, 0, 0, 3, 4, 0, 0)`.
    df = spark.createDataFrame([(0.0, 1.0), (2.0, 3.0)], ["a", "b"])
    interaction = Interaction(inputCols=["a", "b"], outputCol="ab")
    interaction.transform(df).show()
    +---+---+-----+
    |  a|  b|   ab|
    +---+---+-----+
    |0.0|1.0|[0.0]|
    |2.0|3.0|[6.0]|
    +---+---+-----+
    ...
    interactionPath = temp_path + "/interaction"
    interaction.save(interactionPath)
    loadedInteraction = Interaction.load(interactionPath)
    loadedInteraction.transform(df).head().ab == interaction.transform(df).head().ab
    True
    .. versionadded:: 3.0.0
    """

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        """
        __init__(self, inputCols=None, outputCol=None):
        """
        super(InteractionTransformer, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.feature.Interaction", self.uid
        )
        self._setDefault()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        """
        setParams(self, inputCols=None, outputCol=None)
         for this Interaction.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)
