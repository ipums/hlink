# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import logging
import pyspark.sql.functions as f
from pyspark.sql.window import Window

from hlink.linking.util import spark_shuffle_partitions_heuristic

from hlink.linking.link_step import LinkStep


class LinkStepReportR2PercentLinked(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "report round 2 percent linked",
            input_table_names=[
                "prepped_df_a",
                "predicted_matches",
                "hh_predicted_matches",
            ],
            output_table_names=[],
        )

    def _run(self):
        """For households with anyone linked in round 1, report percent of remaining household members linked in round 2."""

        dataset_size1 = self.task.spark.table("prepped_df_a").count()
        dataset_size2 = self.task.spark.table("predicted_matches").count()
        dataset_size3 = self.task.spark.table("hh_predicted_matches").count()
        dataset_size_max = max(dataset_size1, dataset_size2, dataset_size3)
        num_partitions = spark_shuffle_partitions_heuristic(dataset_size_max)
        self.task.spark.sql(f"set spark.sql.shuffle.partitions={num_partitions}")
        logging.info(
            f"Dataset sizes are {dataset_size1}, {dataset_size2}, {dataset_size3}, so set Spark partitions to {num_partitions} for this step"
        )

        pdfa = self.task.spark.table("prepped_df_a").select("serialp", "histid")
        pm = (
            self.task.spark.table("predicted_matches")
            .select("histid_a")
            .withColumn("linked_round", f.lit(1))
        )
        hhpm = (
            self.task.spark.table("hh_predicted_matches")
            .select("histid_a")
            .withColumn("linked_round", f.lit(2))
        )
        linked_rnds = (
            pdfa.join(pm, pdfa["histid"] == pm["histid_a"], "left")
            .drop("histid_a")
            .join(hhpm, pdfa["histid"] == hhpm["histid_a"], "left")
            .drop("histid_a")
            .select(
                "serialp",
                "histid",
                f.when(~f.isnull(pm["linked_round"]), pm["linked_round"])
                .otherwise(hhpm["linked_round"])
                .alias("linked_round"),
            )
            .fillna(0)
        )

        linked_rnds.cache().createOrReplaceTempView("linked_rounds")

        window = Window.partitionBy(linked_rnds["serialp"])
        df = linked_rnds.withColumn("histid_ct_total", f.count("serialp").over(window))
        df0 = df.withColumn(
            "R1count", f.count(f.when(f.col("linked_round") == 1, True)).over(window)
        )
        df1 = df0.withColumn(
            "R2count", f.count(f.when(f.col("linked_round") == 2, True)).over(window)
        )

        dfu = df1.select("serialp", "histid_ct_total", "R1count", "R2count").distinct()
        df2 = dfu.withColumn("R1_pct", dfu["R1count"] / dfu["histid_ct_total"])
        df3 = df2.withColumn(
            "R2_pct", df2["R2count"] / (df2["histid_ct_total"] - df2["R1count"])
        )

        df3.cache().createOrReplaceTempView("counted_links")

        print(
            "Round 1 match rate: "
            + str(df3.agg({"R1_pct": "avg"}).collect()[0]["avg(R1_pct)"])
        )
        print(
            "Round 2 match rate of remaining HH members: "
            + str(df3.agg({"R2_pct": "avg"}).collect()[0]["avg(R2_pct)"])
        )

        self.task.spark.sql("set spark.sql.shuffle.partitions=200")
