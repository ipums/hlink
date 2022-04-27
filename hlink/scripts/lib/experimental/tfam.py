# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from hlink.linking.link_task import LinkTask


def tfam(link_run, id_col, id_a, id_b):
    cols = [
        f"{id_col}",
        "serialp",
        "pernum",
        "relate",
        "namefrst",
        "namelast",
        "age",
        "birthyr",
        "sex",
        "race",
        "marst",
        "durmarr",
        "bpl",
        "nativity",
        "citizen",
        "mbpl",
        "fbpl",
        "statefip_p",
        "street_p",
        "countyicp_p",
        "region_p",
    ]

    if (
        link_run.get_table("training_data").exists()
        and link_run.get_table("training_features").exists()
    ):
        pass
    else:
        link_run.training.run_step(0)
        link_run.training.run_step(1)

    table_a, table_b = _prep_tfam_tables(
        link_run, "tfam_table", "training_data", cols, id_col
    )

    serialp_a = table_a.filter(f"{id_col} == '{id_a}'").take(1)[0]["serialp"]
    serialp_b = table_b.filter(f"{id_col} == '{id_b}'").take(1)[0]["serialp"]
    print("Family 1900:")
    table_a.where(f"serialp = '{serialp_a}'").orderBy("pernum").select(cols).show(
        truncate=False
    )
    print("Family 1910:")
    table_b.where(f"serialp = '{serialp_b}'").orderBy("pernum").select(cols).show(
        truncate=False
    )

    true_match = (
        link_run.spark.table("training_data")
        .where(f"histid_a = '{id_a}' AND histid_b != '{id_b}' AND match = 1")
        .take(1)
    )

    if len(true_match) != 0:
        print("Match family 1910:")
        histid_b_tm = true_match[0][f"{id_col}_b"]
        print(histid_b_tm)
        serialp_matched = table_b.filter(f"{id_col} == '{histid_b_tm}'").take(1)[0][
            "serialp"
        ]
        table_b.where(f"serialp = '{serialp_matched}'").orderBy("pernum").select(
            cols
        ).show(truncate=False)

    print("All hits:")

    tfam_table = link_run.get_table("tfam_hits")
    if tfam_table.exists():
        table_hits = tfam_table.df()
    else:
        sql = """
                    select pa.histid as histid_a, pb.histid as histid_b, pa.namefrst as namefrst_a, pa.namelast as namelast_a, pb.namefrst as namefrst_b, pb.namelast as namelast_b, pa.mbpl as mbpl_a, pb.mbpl as mbpl_b, pa.fbpl as fbpl_a, pb.fbpl as fbpl_b, pa.statefip_p as state_a, pb.statefip_p as state_b, pa.countyicp_p as county_a, pb.countyicp_p as county_b, tf.namefrst_jw, tf.namelast_jw
                    from training_features tf
                    left join raw_df_a pa on pa.histid = tf.histid_a
                    join raw_df_b pb on pb.histid = tf.histid_b
                    """
        table_hits = LinkTask(link_run).run_register_sql(
            name=tfam_table.name, sql=sql, persist=True
        )

    table_hits.where(f"{id_col}_a == '{id_a}'").orderBy(
        ["namelast_jw", "namefrst_jw"], ascending=False
    ).show(20, False)

    print("Features:")
    link_run.spark.table("training_features").where(
        f"histid_a = '{id_a}' AND histid_b = '{id_b}'"
    ).show(100, False)


def tfam_raw(link_run, id_col, id_a, id_b):
    cols = [
        f"{id_col}",
        "serial_p",
        "pernum",
        "relate",
        "namefrst",
        "namelast",
        "age",
        "birthyr",
        "sex",
        "race",
        "marst",
        "durmarr",
        "bpl",
        "nativity",
        "citizen",
        "mbpl",
        "fbpl",
        "statefip_p",
        "street_p",
        "countyicp_p",
        "region_p",
    ]

    table_a = link_run.spark.table("raw_df_a")
    table_b = link_run.spark.table("raw_df_b")

    serialp_a = table_a.filter(f"{id_col} == '{id_a}'").take(1)[0]["SERIAL_P"]
    serialp_b = table_b.filter(f"{id_col} == '{id_b}'").take(1)[0]["SERIAL_P"]
    print("Family 1900:")
    table_a.where(f"SERIAL_P = '{serialp_a}'").orderBy("PERNUM").select(cols).show(
        truncate=False
    )
    print("Family 1910:")
    table_b.where(f"SERIAL_P = '{serialp_b}'").orderBy("PERNUM").select(cols).show(
        truncate=False
    )


def hh_tfam(link_run, id_col, id_a, id_b):
    cols = [
        f"{id_col}",
        "serialp",
        "pernum",
        "relate",
        "namefrst",
        "namelast",
        "age",
        "birthyr",
        "sex",
        "race",
        "marst",
        "durmarr",
        "bpl",
        "nativity",
        "citizen",
        "mbpl",
        "fbpl",
        "statefip_p",
        "street_p",
        "countyicp_p",
        "region_p",
    ]

    table_a, table_b = _prep_tfam_tables(
        link_run, "hh_tfam_table", "hh_predicted_matches", cols, id_col
    )

    serialp_a = table_a.filter(f"{id_col} == '{id_a}'").take(1)[0]["serialp"]
    serialp_b = table_b.filter(f"{id_col} == '{id_b}'").take(1)[0]["serialp"]
    print("Family 1900:")
    table_a.where(f"serialp = '{serialp_a}'").orderBy("pernum").select(cols).show(
        truncate=False
    )
    print("Family 1910:")
    table_b.where(f"serialp = '{serialp_b}'").orderBy("pernum").select(cols).show(
        truncate=False
    )


def hh_tfam_2a(link_run, id_col, id_a1, id_a2, id_b):
    cols = [
        "serialp",
        "pernum",
        "relate",
        "age",
        "birthyr",
        "sex",
        "race",
        "marst",
        "durmarr",
        "bpl",
        "nativity",
        "citizen",
        "birthyr",
        "namelast",
        "namefrst",
        "mbpl",
        "fbpl",
        "statefip_p",
        "street_p",
        "countyicp_p",
        "region_p",
        "histid",
    ]

    table_a, table_b = _prep_tfam_tables(
        link_run, "hh_tfam_table", "hh_predicted_matches", cols, id_col
    )

    serialp_a1 = table_a.filter(f"{id_col} == '{id_a1}'").take(1)[0]["serialp"]
    serialp_a2 = table_a.filter(f"{id_col} == '{id_a2}'").take(1)[0]["serialp"]
    serialp_b = table_b.filter(f"{id_col} == '{id_b}'").take(1)[0]["serialp"]
    print("Family 1900 option 1:")
    table_a.where(f"serialp = '{serialp_a1}'").orderBy("pernum").select(cols).show(
        truncate=False
    )
    print("Family 1900 option 2:")
    table_b.where(f"serialp = '{serialp_a2}'").orderBy("pernum").select(cols).show(
        truncate=False
    )
    print("Family 1910:")
    table_b.where(f"serialp = '{serialp_b}'").orderBy("pernum").select(cols).show(
        truncate=False
    )


def hh_tfam_2b(link_run, id_col, id_a, id_b1, id_b2):
    cols = [
        f"{id_col}",
        "serialp",
        "pernum",
        "relate",
        "namefrst",
        "namelast",
        "age",
        "birthyr",
        "sex",
        "race",
        "marst",
        "durmarr",
        "bpl",
        "nativity",
        "citizen",
        "mbpl",
        "fbpl",
        "statefip_p",
        "street_p",
        "countyicp_p",
        "region_p",
    ]

    table_a, table_b = _prep_tfam_tables(
        link_run, "hh_tfam_table", "hh_predicted_matches", cols, id_col
    )

    serialp_a = table_a.filter(f"{id_col} == '{id_a}'").take(1)[0]["serialp"]
    serialp_b1 = table_b.filter(f"{id_col} == '{id_b1}'").take(1)[0]["serialp"]
    serialp_b2 = table_b.filter(f"{id_col} == '{id_b2}'").take(1)[0]["serialp"]
    print("Family 1900:")
    table_a.where(f"serialp = '{serialp_a}'").orderBy("pernum").select(cols).show(
        truncate=False
    )
    print("Family 1910 option 1:")
    table_b.where(f"serialp = '{serialp_b1}'").orderBy("pernum").select(cols).show(
        truncate=False
    )
    print("Family 1910 option 2:")
    table_b.where(f"serialp = '{serialp_b2}'").orderBy("pernum").select(cols).show(
        truncate=False
    )


def _prep_tfam_tables(link_run, table_name, source_table, cols, id_col):
    tables = []
    for a_or_b in ["a", "b"]:
        table = link_run.get_table(f"{table_name}_{a_or_b}")
        if table.exists():
            tables.append(table.df())
        else:
            tables.append(
                LinkTask(link_run).run_register_sql(
                    name=table.name,
                    template="tfam_tables",
                    t_ctx={
                        "a_or_b": a_or_b,
                        "cols": cols,
                        "id": id_col,
                        "source_table": f"{source_table}",
                    },
                    persist=True,
                )
            )
    return tables[0], tables[1]
