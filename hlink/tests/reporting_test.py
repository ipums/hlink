# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pandas as pd
import pyspark.sql.functions as f
import os


def test_report_r2_percent_linked(reporting, spark, reporting_test_data_r2_pct):

    pdfa_path, pm_path, hhpm_path = reporting_test_data_r2_pct

    reporting.spark.read.csv(pdfa_path, header=True).write.mode(
        "overwrite"
    ).saveAsTable("prepped_df_a")
    reporting.spark.read.csv(pm_path, header=True).write.mode("overwrite").saveAsTable(
        "predicted_matches"
    )
    reporting.spark.read.csv(hhpm_path, header=True).write.mode(
        "overwrite"
    ).saveAsTable("hh_predicted_matches")

    reporting.run_step(0)

    linked_rnds = reporting.spark.table("linked_rounds")
    calc_lr = linked_rnds.select("linked_round").rdd.flatMap(lambda x: x).collect()
    coded_lr = list(
        map(
            int,
            reporting.spark.table("prepped_df_a")
            .select("linked_round_hardcoded")
            .rdd.flatMap(lambda x: x)
            .collect(),
        )
    )

    assert calc_lr == coded_lr

    counted_links = reporting.spark.table("counted_links")
    assert (
        counted_links.select(f.mean("R1_pct")).rdd.flatMap(lambda x: x).collect()[0]
        == 0.425
    )
    assert (
        round(
            counted_links.select(f.mean("R2_pct"))
            .rdd.flatMap(lambda x: x)
            .collect()[0],
            5,
        )
        == 0.39583
    )


def test_report_representivity(
    reporting, spark, reporting_test_data_representivity, integration_conf
):

    rdf_path, pdf_path, pm_path, hhpm_path = reporting_test_data_representivity

    reporting.spark.read.csv(rdf_path, header=True).write.mode("overwrite").saveAsTable(
        "raw_df_a"
    )
    reporting.spark.read.csv(rdf_path, header=True).write.mode("overwrite").saveAsTable(
        "raw_df_b"
    )
    reporting.spark.read.csv(pdf_path, header=True).write.mode("overwrite").saveAsTable(
        "prepped_df_a"
    )
    reporting.spark.read.csv(pdf_path, header=True).write.mode("overwrite").saveAsTable(
        "prepped_df_b"
    )
    reporting.spark.read.csv(pm_path, header=True).write.mode("overwrite").saveAsTable(
        "predicted_matches"
    )
    reporting.spark.read.csv(hhpm_path, header=True).write.mode(
        "overwrite"
    ).saveAsTable("hh_predicted_matches")

    reporting.link_run.config = integration_conf
    reporting.run_step(1)

    sda = reporting.spark.table("source_data_a")
    assert all(
        elem
        in [
            "histid",
            "serialp",
            "sex",
            "age",
            "marst",
            "durmarr",
            "statefip",
            "sei",
            "linked_round_all",
            "race_div_100",
            "relate_div_100",
            "region",
            "bpl_clean",
            "namefrst_unstd",
            "namefrst_std",
            "namelast_clean",
            "region_of_residence",
        ]
        for elem in list(sda.columns)
    )

    fdir = os.path.dirname(__file__)
    df = pd.read_csv(
        os.path.join(fdir, "../../output_data/reports/representivity.csv"),
        index_col=["feature", "values"],
    )
    df_expected = pd.read_csv(
        os.path.join(fdir, "input_data/representivity.csv"),
        index_col=["feature", "values"],
    )
    assert df.equals(df_expected)
