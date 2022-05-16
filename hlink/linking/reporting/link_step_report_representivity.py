# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import csv
import os
from timeit import default_timer as timer
import pyspark.sql.functions as f
from pyspark.sql.window import Window

from hlink.linking.link_step import LinkStep


class LinkStepReportRepresentivity(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "report representivity",
            input_table_names=[
                "raw_df_a",
                "raw_df_b",
                "prepped_df_a",
                "prepped_df_b",
                "predicted_matches",
                "hh_predicted_matches",
            ],
            output_table_names=[],
        )

    def _run(self):
        """Report on representivity of linked data compared to source populations for 1900, 1910, linked round 1, and linked round 2."""
        spark = self.task.spark
        config = self.task.link_run.config

        # check to make sure columns are in data
        raw_cols_wishlist = {
            "histid",
            "serialp",
            "sex",
            "age",
            "marst",
            "durmarr",
            "sei",
        }
        raw_a_cols_present = set([x.lower() for x in spark.table("raw_df_a").columns])
        raw_b_cols_present = set([x.lower() for x in spark.table("raw_df_b").columns])
        raw_cols = list(raw_cols_wishlist & raw_a_cols_present & raw_b_cols_present)

        prepped_cols_wishlist = {
            "histid",
            "race_div_100",
            "relate_div_100",
            "region",
            "bpl_clean",
            "namefrst_unstd",
            "namefrst_std",
            "namelast_clean",
            "statefip",
        }
        pdfa_cols_present = set(
            [x.lower() for x in spark.table("prepped_df_a").columns]
        )
        pdfb_cols_present = set(
            [x.lower() for x in spark.table("prepped_df_b").columns]
        )
        prepped_cols = list(
            prepped_cols_wishlist & pdfa_cols_present & pdfb_cols_present
        )

        rdfa = spark.table("raw_df_a").select(raw_cols)
        rdfb = spark.table("raw_df_b").select(raw_cols)
        pdfa = spark.table("prepped_df_a").select(prepped_cols)
        pdfb = spark.table("prepped_df_b").select(prepped_cols)
        pm = (
            spark.table("predicted_matches")
            .select("histid_a", "histid_b")
            .withColumn("linked_round", f.lit(1))
        )
        hhpm = (
            spark.table("hh_predicted_matches")
            .select("histid_a", "histid_b")
            .withColumn("linked_round", f.lit(2))
        )

        source_data_a = (
            rdfa.join(pm, rdfa["histid"] == pm["histid_a"], "left")
            .drop("histid_a")
            .join(hhpm, rdfa["histid"] == hhpm["histid_a"], "left")
            .drop("histid_a")
            .select(
                "*",
                f.when(~f.isnull(pm["linked_round"]), pm["linked_round"])
                .when(~f.isnull(hhpm["linked_round"]), hhpm["linked_round"])
                .otherwise(0)
                .alias("linked_round_all"),
            )
            .drop("linked_round", "histid_b")
            .join(pdfa, "histid", "left")
        )

        source_data_b = (
            rdfb.join(pm, rdfb["histid"] == pm["histid_b"], "left")
            .drop("histid_b")
            .join(hhpm, rdfb["histid"] == hhpm["histid_b"], "left")
            .drop("histid_b")
            .select(
                "*",
                f.when(~f.isnull(pm["linked_round"]), pm["linked_round"])
                .when(~f.isnull(hhpm["linked_round"]), hhpm["linked_round"])
                .otherwise(0)
                .alias("linked_round_all"),
            )
            .drop("linked_round", "histid_a")
            .join(pdfb, "histid", "left")
        )

        source_data_a.createOrReplaceTempView("source_data_a_pre0")
        source_data_b.createOrReplaceTempView("source_data_b_pre0")

        sda_ct = source_data_a.count()
        sdb_ct = source_data_b.count()
        sda_r1_ct = source_data_a.filter("linked_round_all = 1").count()
        sda_r2_ct = source_data_a.filter("linked_round_all = 2").count()
        sdb_r1_ct = source_data_b.filter("linked_round_all = 1").count()
        sdb_r2_ct = source_data_b.filter("linked_round_all = 2").count()

        # Add the region of residence col
        input_col = "statefip"  # join key in core data
        output_col = "region_of_residence"
        col_to_join_on = "bpl"
        col_to_add = "region"
        this_path = os.path.dirname(__file__)
        region_dict = os.path.join(this_path, "../../tests/input_data/region.csv")
        null_filler = 99
        col_type = "integer"

        # open up csv file
        self.task.run_register_python(
            name="region_data",
            func=lambda: spark.read.csv(region_dict, header=True, inferSchema=True),
            persist=True,
        )

        # join the csv file to the dataframe (df_selected)
        source_data_a = self.task.run_register_sql(
            "source_data_a_pre1",
            template="attach_variable",
            t_ctx={
                "input_col": input_col,
                "output_col": output_col,
                "prepped_df": "source_data_a_pre0",
                "col_to_join_on": col_to_join_on,
                "col_to_add": col_to_add,
                "region_data": "region_data",
            },
            overwrite_preexisting_tables=True,
            persist=True,
        )
        source_data_b = self.task.run_register_sql(
            "source_data_b_pre1",
            template="attach_variable",
            t_ctx={
                "input_col": input_col,
                "output_col": output_col,
                "prepped_df": "source_data_b_pre0",
                "col_to_join_on": col_to_join_on,
                "col_to_add": col_to_add,
                "region_data": "region_data",
            },
            overwrite_preexisting_tables=True,
            persist=True,
        )
        source_data_a = source_data_a.fillna(null_filler, subset=[output_col])
        source_data_b = source_data_b.fillna(null_filler, subset=[output_col])

        source_data_a.withColumn(
            output_col, source_data_a[output_col].cast(col_type)
        ).write.mode("overwrite").saveAsTable("source_data_a")
        source_data_b.withColumn(
            output_col, source_data_b[output_col].cast(col_type)
        ).write.mode("overwrite").saveAsTable("source_data_b")

        source_data_a = spark.table("source_data_a")
        source_data_b = spark.table("source_data_b")

        reports_path = os.path.join(this_path, "../../../output_data")
        folder_path = os.path.join(this_path, "../../../output_data/reports")
        csv_path = os.path.join(
            this_path, "../../../output_data/reports/representivity.csv"
        )
        if not os.path.exists(reports_path):
            os.mkdir(reports_path)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        alias_source_a = config["datasource_a"]["alias"]
        alias_source_b = config["datasource_b"]["alias"]

        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            header = [
                "feature",
                "values",
                f"{alias_source_a} all count",
                f"{alias_source_a} all percent",
                f"{alias_source_a} round 1 count",
                f"{alias_source_a} round 1 percent",
                f"{alias_source_a} round 2 count",
                f"{alias_source_a} round 2 percent",
                f"{alias_source_b} all count",
                f"{alias_source_b} all percent",
                f"{alias_source_b} round 1 count",
                f"{alias_source_b} round 1 percent",
                f"{alias_source_b} round 2 count",
                f"{alias_source_b} round 2 percent",
            ]
            csvwriter.writerow(header)
            csvwriter.writerow(
                [
                    "Total count",
                    "",
                    sda_ct,
                    "",
                    sda_r1_ct,
                    "",
                    sda_r2_ct,
                    "",
                    sdb_ct,
                    "",
                    sdb_r1_ct,
                    "",
                    sdb_r2_ct,
                    "",
                ]
            )
            csvwriter.writerow([])

            def _groupby_cascade(
                feature,
                label=False,
                comp_type="other",
                interval=False,
                dni=False,
                gt_threshold=False,
                lt_threshold=False,
                second_feature=False,
                groupby_feat=False,
                csvwriter=csvwriter,
                source_data_a=source_data_a,
                source_data_b=source_data_b,
                sda_ct=sda_ct,
                sda_r1_ct=sda_r1_ct,
                sda_r2_ct=sda_r2_ct,
                sdb_ct=sdb_ct,
                sdb_r1_ct=sdb_r1_ct,
                sdb_r2_ct=sdb_r2_ct,
            ):
                start = timer()

                def _get_withColumn(sd):
                    if comp_type == "keep_low_by_group":
                        with_col = "rb"
                        sd = sd.withColumn(
                            with_col,
                            f.when(
                                sd[feature].cast("int") >= lt_threshold, "null"
                            ).otherwise(
                                sd[groupby_feat].cast("int")
                                - (sd[groupby_feat].cast("int") % interval)
                            ),
                        )

                    elif comp_type == "keep_high_by_group":
                        with_col = "rb"
                        if dni:
                            sd = sd.withColumn(
                                with_col,
                                f.when(
                                    (sd[feature].cast("int") > gt_threshold)
                                    & (sd[feature].cast("int") != dni),
                                    sd[groupby_feat].cast("int")
                                    - (sd[groupby_feat].cast("int") % interval),
                                ).otherwise("null"),
                            )

                        else:
                            sd = sd.withColumn(
                                with_col,
                                f.when(
                                    sd[feature].cast("int") > gt_threshold,
                                    sd[groupby_feat].cast("int")
                                    - (sd[groupby_feat].cast("int") % interval),
                                ).otherwise("null"),
                            )

                    elif comp_type == "not_equals":
                        with_col = "rb"
                        sd = sd.withColumn(
                            with_col,
                            (
                                sd[feature].cast("int")
                                != sd[second_feature].cast("int")
                            ).cast("int"),
                        )

                    elif comp_type == "not_equals_by_group":
                        with_col = "rb"
                        sd = sd.withColumn(
                            with_col,
                            f.when(
                                (
                                    sd[feature].cast("int")
                                    != sd[second_feature].cast("int")
                                ),
                                sd[groupby_feat].cast("int")
                                - (sd[groupby_feat].cast("int") % interval),
                            ).otherwise("null"),
                        )

                    elif comp_type == "groupby_then_bucketize_name_count":
                        with_col = "range"
                        w = Window.partitionBy(feature)
                        sd = sd.withColumn("n", f.count(feature).over(w))
                        sd = sd.withColumn(
                            with_col,
                            f.when(sd["n"] < 6, "1-5")
                            .when((sd["n"] >= 6) & (sd["n"] < 21), "6-20")
                            .when((sd["n"] >= 21) & (sd["n"] < 61), "21-60")
                            .otherwise("61-"),
                        )

                    elif interval and not gt_threshold:
                        with_col = "range"
                        sd = sd.withColumn(
                            with_col,
                            sd[feature].cast("int")
                            - (sd[feature].cast("int") % interval),
                        )

                    elif gt_threshold and not interval:
                        with_col = "bool"
                        if dni:
                            sd = sd.withColumn(
                                with_col,
                                f.when(
                                    sd[feature] != dni,
                                    (sd[feature].cast("int") > gt_threshold).cast(
                                        "int"
                                    ),
                                ).otherwise(0),
                            )

                        else:
                            sd = sd.withColumn(
                                with_col,
                                (sd[feature].cast("int") > gt_threshold).cast("int"),
                            )

                    elif lt_threshold and not interval:
                        with_col = "bool"
                        if dni:
                            sd = sd.withColumn(
                                with_col,
                                f.when(
                                    sd[feature] != dni,
                                    (sd[feature].cast("int") < lt_threshold).cast(
                                        "int"
                                    ),
                                ).otherwise(0),
                            )

                        else:
                            sd = sd.withColumn(
                                with_col,
                                (sd[feature].cast("int") < lt_threshold).cast("int"),
                            )

                    else:
                        with_col = feature
                    return sd, with_col

                if feature in source_data_a.columns:
                    source_data_a, with_col = _get_withColumn(source_data_a)
                    source_data_b, with_col = _get_withColumn(source_data_b)

                    data_sources = [
                        source_data_a,
                        source_data_a.filter("linked_round_all = 1"),
                        source_data_a.filter("linked_round_all = 2"),
                        source_data_b,
                        source_data_b.filter("linked_round_all = 1"),
                        source_data_b.filter("linked_round_all = 2"),
                    ]
                    data_outputs = ["all_a", "r1_a", "r2_a", "all_b", "r1_b", "r2_b"]
                    data = {}

                    for ds, do in zip(data_sources, data_outputs):
                        data[do] = _get_dict_from_window_rows(
                            ds.groupby(with_col).count().collect(), with_col
                        )
                    rows = {}
                    if label:
                        lb = label
                    else:
                        lb = feature
                    keys = sorted(set(data["all_a"].keys()) | set(data["all_b"].keys()))
                    dfs = ["r1_a", "r2_a", "all_b", "r1_b", "r2_b"]
                    cts = [sda_r1_ct, sda_r2_ct, sdb_ct, sdb_r1_ct, sdb_r2_ct]
                    for key in keys:
                        rows[key] = [
                            lb,
                            key,
                            data["all_a"].get(key, 0),
                            data["all_a"].get(key, 0) / sda_ct,
                        ]
                    for df, ct in zip(dfs, cts):
                        for key in keys:
                            rows[key] = rows[key] + [
                                data[df].get(key, 0),
                                data[df].get(key, 0) / ct,
                            ]
                    csvwriter.writerows(rows.values())
                    csvwriter.writerow([])
                    end = timer()
                    elapsed_time = round(end - start, 2)
                    print(f"Finished generating {lb}: {elapsed_time}s")
                else:
                    print(f"Not comparing {feature}: not present in source data.")

            def _serialp_children_over_10_window(
                label,
                csvwriter=csvwriter,
                source_data_a=source_data_a,
                source_data_b=source_data_b,
                sda_ct=sda_ct,
                sda_r1_ct=sda_r1_ct,
                sda_r2_ct=sda_r2_ct,
                sdb_ct=sdb_ct,
                sdb_r1_ct=sdb_r1_ct,
                sdb_r2_ct=sdb_r2_ct,
            ):
                start = timer()

                # use window to get count of persons in serialp household who have age > 10 and relate code 3XX for data a and b
                window = Window.partitionBy("serialp")

                source_data_a = source_data_a.withColumn(
                    "count_of_children_over_10",
                    f.sum(
                        (
                            (source_data_a["relate_div_100"].cast("int") == 3)
                            & (source_data_a["age"].cast("int") > 10)
                        ).cast("int")
                    ).over(window),
                )
                source_data_b = source_data_b.withColumn(
                    "count_of_children_over_10",
                    f.sum(
                        (
                            (source_data_b["relate_div_100"].cast("int") == 3)
                            & (source_data_b["age"].cast("int") > 10)
                        ).cast("int")
                    ).over(window),
                )

                # bucketize by count
                coc = "count_of_children_over_10"
                source_data_a = source_data_a.withColumn(
                    "child_count_bucketized",
                    f.when(source_data_a[coc] == 0, "0")
                    .when((source_data_a[coc] >= 1) & (source_data_a[coc] < 3), "1-2")
                    .when((source_data_a[coc] >= 3) & (source_data_a[coc] < 6), "3-5")
                    .otherwise("6-"),
                )
                source_data_b = source_data_b.withColumn(
                    "child_count_bucketized",
                    f.when(source_data_b[coc] == 0, "0")
                    .when((source_data_b[coc] >= 1) & (source_data_b[coc] < 3), "1-2")
                    .when((source_data_b[coc] >= 3) & (source_data_b[coc] < 6), "3-5")
                    .otherwise("6-"),
                )

                # for each datasource, get counts and percentages and write to CSV
                data_sources = [
                    source_data_a,
                    source_data_a.filter("linked_round_all = 1"),
                    source_data_a.filter("linked_round_all = 2"),
                    source_data_b,
                    source_data_b.filter("linked_round_all = 1"),
                    source_data_b.filter("linked_round_all = 2"),
                ]

                data_outputs = ["all_a", "r1_a", "r2_a", "all_b", "r1_b", "r2_b"]
                data = {}

                for ds, do in zip(data_sources, data_outputs):
                    data[do] = _get_dict_from_window_rows(
                        ds.groupby("child_count_bucketized").count().collect(),
                        "child_count_bucketized",
                    )
                rows = {}

                keys = sorted(set(data["all_a"].keys()) | set(data["all_b"].keys()))
                dfs = ["r1_a", "r2_a", "all_b", "r1_b", "r2_b"]
                cts = [sda_r1_ct, sda_r2_ct, sdb_ct, sdb_r1_ct, sdb_r2_ct]
                for key in keys:
                    rows[key] = [
                        label,
                        key,
                        data["all_a"].get(key, 0),
                        data["all_a"].get(key, 0) / sda_ct,
                    ]
                for df, ct in zip(dfs, cts):
                    for key in keys:
                        rows[key] = rows[key] + [
                            data[df].get(key, 0),
                            data[df].get(key, 0) / ct,
                        ]
                csvwriter.writerows(rows.values())
                csvwriter.writerow([])
                end = timer()
                elapsed_time = round(end - start, 2)
                print(f"Finished generating {label}: {elapsed_time}s")

            _serialp_children_over_10_window(
                label="presence of children over the age of 10 in the household"
            )

            # TODO: bucketize specific codes to text
            # TODO: window of household, are children over the age of 10 present

            _groupby_cascade(feature="sex")
            _groupby_cascade(feature="age", interval=10, label="age")
            _groupby_cascade(feature="race_div_100", label="race")
            _groupby_cascade(
                feature="relate_div_100", label="relationship to household head"
            )
            _groupby_cascade(feature="marst")
            _groupby_cascade(feature="marst", lt_threshold=3, label="married")
            _groupby_cascade(
                feature="marst",
                comp_type="keep_low_by_group",
                lt_threshold=3,
                groupby_feat="age",
                interval=10,
                label="married, by age",
            )
            _groupby_cascade(
                feature="durmarr",
                gt_threshold=9,
                dni=99,
                label="marriage duration at least ten years",
            )
            _groupby_cascade(
                feature="durmarr",
                comp_type="keep_high_by_group",
                gt_threshold=9,
                groupby_feat="age",
                interval=10,
                label="marriage duration at least 10 years, by age",
                dni=99,
            )
            _groupby_cascade(feature="region_of_residence", label="region of residence")
            _groupby_cascade(feature="region", label="region of birth")
            _groupby_cascade(feature="sei", interval=15, label="socioeconomic status")
            _groupby_cascade(
                feature="bpl_clean",
                second_feature="statefip",
                label="lifetime migrant",
                comp_type="not_equals",
            )
            _groupby_cascade(
                feature="bpl_clean",
                second_feature="statefip",
                groupby_feat="age",
                label="lifetime migrant by age",
                comp_type="not_equals_by_group",
                interval=10,
            )
            _groupby_cascade(
                feature="namefrst_unstd",
                label="namefrst_unstd commonality",
                comp_type="groupby_then_bucketize_name_count",
            )
            _groupby_cascade(
                feature="namefrst_std",
                label="namefrst_std commonality",
                comp_type="groupby_then_bucketize_name_count",
            )
            _groupby_cascade(
                feature="namelast_clean",
                label="namelast_clean commonality",
                comp_type="groupby_then_bucketize_name_count",
            )


def _get_dict_from_window_rows(collected_rows, new_col):
    new_dict = {}
    for row in collected_rows:
        new_dict[row[new_col]] = row["count"]
    return new_dict
