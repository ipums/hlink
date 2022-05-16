# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
import pyspark.sql.functions as f
import os

from hlink.linking.link_step import LinkStep


class LinkStepExportCrosswalk(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "export crosswalk",
            input_table_names=[
                "raw_df_a",
                "raw_df_b",
                "predicted_matches",
                "hh_predicted_matches",
            ],
            output_table_names=[],
        )

    def _run(self):
        """Pull in key demographic data for linked individuals and export a fixed-width crosswalk file."""
        config = self.task.link_run.config

        pm = (
            self.task.spark.table("predicted_matches")
            .select("histid_a", "histid_b")
            .withColumn("linked_round", f.lit(1))
        )
        hhpm = (
            self.task.spark.table("hh_predicted_matches")
            .select("histid_a", "histid_b")
            .withColumn("linked_round", f.lit(2))
        )
        pm.unionByName(hhpm).write.mode("overwrite").saveAsTable(
            "all_predicted_matches"
        )

        raw_cols = ["histid", "serialp", "pernum", "age", "sex", "statefip", "bpl"]

        raw_cols_sql = "select "
        raw_cols_sql += ", ".join([f"raw_a.{col} as {col}_a" for col in raw_cols])
        raw_cols_sql += ", "
        raw_cols_sql += ", ".join([f"raw_b.{col} as {col}_b" for col in raw_cols])
        raw_cols_sql += "from all_predicted_matches left join raw_df_a raw_a on histid_a left join raw_df_b raw_b on histid_b"

        joined_predictions_with_demog = self.task.spark.sql(raw_cols_sql)
        joined_predictions_with_demog.write.mode("overwrite").saveAsTable(
            "joined_predictions"
        )
        jp = self.task.spark.table("joined_predictions")

        year_a = config["datasource_a"]["alias"]
        year_b = config["datasource_b"]["alias"]

        this_path = os.path.dirname(__file__)
        reports_path = os.path.join(this_path, "../../../output_data")
        folder_path = os.path.join(this_path, "../../../output_data/crosswalks")
        csv_path = os.path.join(
            this_path,
            f"../../../output_data/crosswalks/{year_a}_{year_b}_predicted_matches_crosswalk.csv",
        )
        if not os.path.exists(reports_path):
            os.mkdir(reports_path)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        jp.toPandas().to_csv(csv_path)

        # TODO: generate crosswalk output as fixed width instead of CSV (modify code from NHGIS)
        # column widths = {"histid": 36, "serialp": 8, "pernum": 4, "age": 3, "sex": 1, "statefip": 2, "bpl": 5}
