from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

from hlink.linking.transformers.rename_vector_attributes import RenameVectorAttributes


def test_rename_vector_attributes(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [[0.0, 1.0], [1.0, 2.0], [3.0, 4.0]], schema=["A", "regionf_0:namelast_jw"]
    )

    assembler = VectorAssembler(
        inputCols=["A", "regionf_0:namelast_jw"], outputCol="vectorized"
    )
    remove_colons = RenameVectorAttributes()
    transformed = remove_colons.transform(assembler.transform(df))

    attrs = transformed.schema["vectorized"].metadata["ml_attr"]["attrs"]["numeric"]
    attr_names = [attr["name"] for attr in attrs]
    assert attr_names == ["A", "regionf_0_namelast_jw"]
