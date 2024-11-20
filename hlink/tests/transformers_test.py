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
    remove_colons = RenameVectorAttributes(
        inputCol="vectorized", strsToReplace=[":"], replaceWith="_"
    )
    transformed = remove_colons.transform(assembler.transform(df))

    attrs = transformed.schema["vectorized"].metadata["ml_attr"]["attrs"]["numeric"]
    attr_names = [attr["name"] for attr in attrs]
    assert attr_names == ["A", "regionf_0_namelast_jw"]


def test_rename_vector_attributes_multiple_replacements(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [[1, 2], [3, 4]], schema=["column1*has*stars", "column2*multiple/symbols"]
    )

    assembler = VectorAssembler(
        inputCols=["column1*has*stars", "column2*multiple/symbols"], outputCol="vector"
    )
    rename_attrs = RenameVectorAttributes(
        inputCol="vector", strsToReplace=["*", "/"], replaceWith=""
    )
    transformed = rename_attrs.transform(assembler.transform(df))

    attrs = transformed.schema["vector"].metadata["ml_attr"]["attrs"]["numeric"]
    attr_names = [attr["name"] for attr in attrs]
    assert attr_names == ["column1hasstars", "column2multiplesymbols"]
