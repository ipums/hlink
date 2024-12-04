# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer, VectorAssembler

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

    # Save to Java, then reload to confirm that the metadata changes are persistent
    transformed.write.mode("overwrite").saveAsTable("transformed")
    df = spark.table("transformed")

    attrs = df.schema["vectorized"].metadata["ml_attr"]["attrs"]["numeric"]
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

    # Save to Java, then reload to confirm that the metadata changes are persistent
    transformed.write.mode("overwrite").saveAsTable("transformed")
    df = spark.table("transformed")

    attrs = df.schema["vector"].metadata["ml_attr"]["attrs"]["numeric"]
    attr_names = [attr["name"] for attr in attrs]
    assert attr_names == ["column1hasstars", "column2multiplesymbols"]


def test_rename_vector_attributes_on_bucketized_feature(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        [[0.1, 0.7, 0.2, 0.5, 0.8, 0.2, 0.3]], schema=["namelast_jw"]
    )

    bucketizer = Bucketizer(
        inputCol="namelast_jw",
        outputCol="namelast_jw_buckets",
        splits=[0.0, 0.33, 0.67, 1.0],
    )
    rename_attrs = RenameVectorAttributes(
        inputCol="namelast_jw_buckets", strsToReplace=[","], replaceWith=""
    )
    transformed = rename_attrs.transform(bucketizer.transform(df))

    # Save to Java, then reload to confirm that the metadata changes are persistent
    transformed.write.mode("overwrite").saveAsTable("transformed")
    output_df = spark.table("transformed")

    # Bucketized vectors have different metadata
    values = output_df.schema["namelast_jw_buckets"].metadata["ml_attr"]["vals"]
    assert values == ["0.0 0.33", "0.33 0.67", "0.67 1.0"]
