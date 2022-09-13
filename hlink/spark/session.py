# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import os.path
from pyspark import SparkConf
from pyspark.sql import SparkSession
import hlink.spark
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)


class SparkConnection(object):
    """Handles initialization of spark session and connection to local cluster."""

    def __init__(self, derby_dir, warehouse_dir, tmp_dir, python, db_name):
        self.derby_dir = derby_dir
        self.warehouse_dir = warehouse_dir
        self.db_name = db_name
        self.tmp_dir = tmp_dir
        self.python = python

    def spark_conf(self, executor_cores, executor_memory, driver_memory, cores):
        spark_package_path = os.path.dirname(hlink.spark.__file__)
        jar_path = os.path.join(
            spark_package_path, "jars", "hlink_lib-assembly-1.0.jar"
        )
        os.environ["PYSPARK_PYTHON"] = self.python
        conf = (
            SparkConf()
            .set("spark.pyspark.python", self.python)
            .set("spark.local.dir", self.tmp_dir)
            .set("spark.sql.warehouse.dir", self.warehouse_dir)
            .set(
                "spark.driver.extraJavaOptions", f"-Dderby.system.home={self.derby_dir}"
            )
            .set("spark.executorEnv.SPARK_LOCAL_DIRS", self.tmp_dir)
            .set("spark.sql.legacy.allowUntypedScalaUDF", True)
            .setAppName("linking")
            # .set("spark.executor.cores", executor_cores) \
        )
        if executor_memory:
            conf.set("spark.executor.memory", executor_memory)
        if driver_memory:
            conf.set("spark.driver.memory", driver_memory)
        if cores:
            conf.set("spark.cores.max", cores)

        if os.path.isfile(jar_path):
            conf = conf.set("spark.jars", jar_path)
        return conf

    def local(self, cores=1, executor_memory="10G"):
        """Create a local 'cluster'."""
        # When the cluster is local, the executor and driver are the same machine. So set
        # driver_memory = executor_memory automatically.
        return self.connect(
            f"local[{cores}]", cores, executor_memory, executor_memory, cores
        )

    def connect(
        self,
        connection_string,
        executor_cores=None,
        executor_memory=None,
        driver_memory=None,
        cores=None,
    ):
        conf = self.spark_conf(
            executor_cores=executor_cores,
            executor_memory=executor_memory,
            driver_memory=driver_memory,
            cores=cores,
        )
        session = (
            SparkSession.builder.config(conf=conf)
            .enableHiveSupport()
            .master(connection_string)
            .getOrCreate()
        )
        session.sparkContext.setLogLevel("ERROR")

        if self.db_name not in [d.name for d in session.catalog.listDatabases()]:
            session.sql(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")
        session.catalog.setCurrentDatabase(self.db_name)
        session.sparkContext.setCheckpointDir(str(self.tmp_dir))
        self.__register_udfs(session)
        return session

    def __register_udfs(self, session):
        session.udf.registerJavaFunction("jw", "com.isrdi.udfs.JWCompare", DoubleType())
        session.udf.registerJavaFunction(
            "jw_max", "com.isrdi.udfs.MaxJWCompare", DoubleType()
        )
        session.udf.registerJavaFunction(
            "jw_rate", "com.isrdi.udfs.JWRate", DoubleType()
        )
        session.udf.registerJavaFunction(
            "rel_jw", "com.isrdi.udfs.JWRelatedRows", DoubleType()
        )
        session.udf.registerJavaFunction(
            "extra_children", "com.isrdi.udfs.ExtraChildren", DoubleType()
        )
        session.udf.registerJavaFunction(
            "hh_compare_rate", "com.isrdi.udfs.HHCompare", DoubleType()
        )
        session.udf.registerJavaFunction(
            "has_matching_element", "com.isrdi.udfs.HasMatchingElement", BooleanType()
        )
        session.udf.registerJavaFunction(
            "parseProbVector", "com.isrdi.udfs.ParseProbabilityVector", DoubleType()
        )
        session.udf.registerJavaFunction(
            "hh_rows_get_first_value",
            "com.isrdi.udfs.HHRowsGetFirstValue",
            StructType(
                [StructField("serial", LongType()), StructField("input", StringType())]
            ),
        )
        session.udf.registerJavaFunction(
            "extract_neighbors",
            "com.isrdi.udfs.ExtractNeighbors",
            ArrayType(StringType()),
        )
