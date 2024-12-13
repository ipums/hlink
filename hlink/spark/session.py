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

# SynapseML is a package which provides LightGBM-Spark integration for hlink.
# It's an optional dependency. When it is installed, we need to download an
# additional Scala library by setting some Spark configurations. When it's not
# installed, we avoid downloading the extra library since it won't be useful.
try:
    import synapse.ml  # noqa: F401
except ModuleNotFoundError:
    _synapse_ml_available = False
else:
    _synapse_ml_available = True


class SparkConnection:
    """Handles initialization of spark session and connection to local cluster."""

    def __init__(
        self,
        derby_dir,
        warehouse_dir,
        checkpoint_dir,
        tmp_dir,
        python,
        db_name,
        app_name="linking",
    ):
        self.derby_dir = derby_dir
        self.warehouse_dir = warehouse_dir
        self.checkpoint_dir = checkpoint_dir
        self.db_name = db_name
        self.tmp_dir = tmp_dir
        self.python = python
        self.app_name = app_name

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
            .setAppName(self.app_name)
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

        # A bit of a kludge. We set spark.jars.repositories here in the configuration,
        # but then we actually download the SynapseML Scala jar later in connect().
        # See the comment on the ADD JAR SQL statement in connect() for some more
        # context.
        #
        # SynapseML used to be named MMLSpark, thus the URL.
        if _synapse_ml_available:
            conf.set("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")

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
        session.sparkContext.setCheckpointDir(str(self.checkpoint_dir))
        self._register_udfs(session)

        # If the SynapseML Python package is available, include the Scala
        # package as well. Note that we have to pin to a particular version of
        # the Scala package here.
        #
        # Despite what the documentation for the spark.jars.packages config setting
        # says, this is the only way that I have found to include this jar for both
        # the driver and the executors. Setting spark.jars.packages caused errors
        # because the executors could not find the jar.
        if _synapse_ml_available:
            session.sql("ADD JAR ivy://com.microsoft.azure:synapseml_2.12:1.0.8")

        return session

    def _register_udfs(self, session):
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
