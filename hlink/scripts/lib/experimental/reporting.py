# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import subprocess
import pyspark.sql.functions as pyspark_funcs


def export_crosswalk(spark, output_path, variables, include_round):
    crosswalk_vars = ["histid_a", "histid_b"]

    if include_round:
        crosswalk_vars.append("round")

    if "histid" not in variables:
        variables.append("histid")

    # We are accessing these records in order to attach extra contextual
    # variables to the pairs of people in the predicted matches tables.
    raw_df_a = spark.table("raw_df_a")
    raw_df_b = spark.table("raw_df_b")

    # We are adding (unioning) together individual matches (predicted_matches) and
    # household matches (hh_predicted_matches)
    #
    # The addition of 'round' is to identify which matches came from which matching round.
    # We may or may not select and export it depending on the --include-round flag.
    predicted_matches = spark.table("predicted_matches").withColumn(
        "round", pyspark_funcs.lit(1)
    )

    hh_predicted_matches = spark.table("hh_predicted_matches").withColumn(
        "round", pyspark_funcs.lit(2)
    )

    for variable in variables:
        if variable not in [c.lower() for c in raw_df_a.columns]:
            print(f"Error: variable '{variable}' does not exist in raw_df_a.")
            return
        if variable not in [c.lower() for c in raw_df_b.columns]:
            print(f"Error: variable '{variable}' does not exist in raw_df_a.")
            return

    all_matches = predicted_matches.select(crosswalk_vars).unionByName(
        hh_predicted_matches.select(crosswalk_vars)
    )

    # Make distinct sets of variable names for the a and b datasets
    columns_a = [f"{variable} as {variable}_a" for variable in variables]
    columns_b = [f"{variable} as {variable}_b" for variable in variables]

    raw_df_a_selected = raw_df_a.selectExpr(columns_a)
    raw_df_b_selected = raw_df_b.selectExpr(columns_b)

    all_matches_with_selections = all_matches.join(raw_df_a_selected, "histid_a").join(
        raw_df_b_selected, "histid_b"
    )

    if "csv" in output_path.split("."):
        export_csv(all_matches_with_selections, output_path)
    else:
        export_fixed_width(variables, all_matches_with_selections, output_path)


def export_csv(all_matches_with_selections, output_path):
    output_tmp = output_path + ".tmp"
    all_matches_with_selections.write.csv(output_tmp, header=False)
    header = (
        '"' + '","'.join([col.name for col in all_matches_with_selections.schema]) + '"'
    )

    commands = [
        f"echo '{header}' > {output_path}",
        f"cat {output_tmp}/* >> {output_path} ",
        f"rm -rf {output_tmp}",
    ]

    for command in commands:
        subprocess.run(command, shell=True)


def export_fixed_width(variables, all_matches_with_selections, output_path):
    output_tmp = output_path + ".tmp"
    sizes = {
        "histid": 36,
        "serialp": 8,
        "pernum": 4,
        "age": 3,
        "sex": 1,
        "statefip_p": 2,
        "bpl": 5,
    }
    fw_columns_a = []
    fw_columns_b = []
    for variable in variables:
        size = sizes.get(variable, 15)
        fw_columns_a.append(
            [f"LPAD({variable}_a, {size}, ' ') as {variable}_a", size, f"{variable}_a"]
        )
    fw_columns_b.append(
        [f"LPAD({variable}_b, {size}, ' ') as {variable}_b", size, f"{variable}_b"]
    )
    all_column_selects = [c[0] for c in (fw_columns_a + fw_columns_b)]

    [print(f"{c[2]} - {c[1]}") for c in (fw_columns_a + fw_columns_b)]
    all_matches_fixed_width = all_matches_with_selections.selectExpr(all_column_selects)
    all_matches_fixed_width.selectExpr("CONCAT_WS('', *)").write.text(output_tmp)

    commands = [f"cat {output_tmp}/* >> {output_path} ", f"rm -rf {output_tmp}"]
    for command in commands:
        subprocess.run(command, shell=True)
