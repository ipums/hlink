# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.sql.functions import (
    array,
    collect_list,
    concat,
    count,
    expr,
    floor,
    length,
    lit,
    lower,
    regexp_replace,
    sort_array,
    soundex,
    split,
    struct,
    trim,
    udf,
    when,
)
from pyspark.sql.types import ArrayType, LongType, StringType
from pyspark.ml import Pipeline
from pyspark.sql import Window
from pyspark.ml.feature import NGram, RegexTokenizer, CountVectorizer, MinHashLSH


def generate_transforms(
    spark, df_selected, feature_selections, link_task, is_a, id_col
):
    not_skipped_feature_selections = [
        c
        for c in feature_selections
        if ("skip" not in c or not (c["skip"]))
        and ("post_agg_feature" not in c or not (c["post_agg_feature"]))
    ]
    post_agg_feature_selections = [
        c
        for c in feature_selections
        if ("post_agg_feature" in c) and c["post_agg_feature"]
    ]

    def parse_feature_selections(df_selected, feature_selection, is_a):
        transform = feature_selection["transform"]

        if not feature_selection.get("output_column", False):
            feature_selection["output_column"] = feature_selection["output_col"]

        if "checkpoint" in feature_selection and feature_selection["checkpoint"]:
            df_selected = df_selected.checkpoint()

        if "override_column_a" in feature_selection and is_a:
            override_name = feature_selection["override_column_a"]
            df_selected = df_selected.withColumn(
                feature_selection["output_column"], df_selected[override_name]
            )
            return df_selected

        elif "override_column_b" in feature_selection and not is_a:
            override_name = feature_selection["override_column_b"]
            df_selected = df_selected.withColumn(
                feature_selection["output_column"], df_selected[override_name]
            )
            return df_selected

        elif "set_value_column_a" in feature_selection and is_a:
            set_value = feature_selection["set_value_column_a"]
            df_selected = df_selected.withColumn(
                feature_selection["output_column"], lit(set_value)
            )
            return df_selected

        elif "set_value_column_b" in feature_selection and not is_a:
            set_value = feature_selection["set_value_column_b"]
            df_selected = df_selected.withColumn(
                feature_selection["output_column"], lit(set_value)
            )
            return df_selected

        elif transform == "bigrams":
            input_col = feature_selection["input_column"]
            output_col = feature_selection["output_column"]
            intermediate_col = input_col + "_tokens"
            unsorted_col = input_col + "_unsorted"
            if (
                "no_first_pad" in feature_selection
                and feature_selection["no_first_pad"]
            ):
                input_col_space = input_col
            else:
                input_col_space = input_col + "_space"
                df_selected = df_selected.withColumn(
                    input_col_space, concat(lit(" "), input_col)
                )
            tokenizer_a = RegexTokenizer(
                pattern="", inputCol=input_col_space, outputCol=intermediate_col
            )
            ngram_a = NGram(n=2, inputCol=intermediate_col, outputCol=output_col)
            pipeline = Pipeline(stages=[tokenizer_a, ngram_a])
            df_selected = pipeline.fit(df_selected).transform(df_selected)
            df_selected = df_selected.withColumn(unsorted_col, df_selected[output_col])
            df_selected = df_selected.withColumn(
                output_col, sort_array(df_selected[unsorted_col])
            )
            return df_selected

        elif transform == "sql_condition":
            cond = feature_selection["condition"]
            output_col = feature_selection["output_column"]
            df_selected = df_selected.withColumn(output_col, expr(cond))
            return df_selected

        elif transform == "array":
            col1, col2 = feature_selection["input_columns"]
            output_col = feature_selection["output_column"]
            df_selected = df_selected.withColumn(output_col, array(col1, col2))
            return df_selected

        elif transform == "union":
            col1, col2 = feature_selection["input_columns"]
            output_col = feature_selection["output_column"]

            def union_list(list_a, list_b):
                return list(set(list_a).union(set(list_b)))

            union_list_udf = udf(union_list, ArrayType(StringType()))
            df_selected = df_selected.withColumn(output_col, union_list_udf(col1, col2))
            return df_selected

        elif transform == "hash":
            input_col = feature_selection["input_column"]
            count_col = feature_selection["output_column"] + "_count"
            hash_array_col = feature_selection["output_column"]
            df_selected = df_selected.where(f"size({input_col}) > 0")
            count_vect = CountVectorizer(inputCol=input_col, outputCol=count_col)
            lsh = MinHashLSH(
                inputCol=count_col,
                outputCol=hash_array_col,
                numHashTables=feature_selection["number"],
                seed=445123,
            )
            # non_zero = udf(lambda v: v.numNonzeros() > 0, BooleanType())
            # hha_count_nonzero = hha_counts.where(non_zero(F.col("word_counts")))
            cv_model = count_vect.fit(df_selected)
            df_transformed = cv_model.transform(df_selected)
            lsh_model = lsh.fit(df_transformed)
            df_selected = lsh_model.transform(df_transformed)
            return df_selected

        elif transform == "soundex":
            input_col = feature_selection["input_column"]
            output_col = feature_selection["output_column"]
            df_selected = df_selected.withColumn(output_col, soundex(input_col))
            return df_selected

        elif transform == "neighbor_aggregate":
            return df_selected
            # df_selected.createOrReplaceTempView("prepped_df_tmp")
            # link_task.run_register_sql("hh_nbor_rank", t_ctx=feature_selection)
            # link_task.run_register_sql("hh_nbor", t_ctx=feature_selection)
            # df_selected = link_task.run_register_sql(
            #    None, template="attach_neighbor_col", t_ctx=feature_selection
            # )
            # spark.catalog.dropTempView("prepped_df_tmp")
            # spark.catalog.dropTempView("hh_nbor")
            # spark.catalog.dropTempView("hh_nbor_rank")

        elif transform == "attach_family_col":
            return df_selected

        elif transform == "related_individuals":
            df_selected.createOrReplaceTempView("prepped_df_tmp")
            df_selected = link_task.run_register_sql(
                None,
                template="attach_related_col",
                t_ctx={
                    "output_col": feature_selection["output_col"],
                    "input_col": feature_selection["input_col"],
                    "prepped_df": "prepped_df_tmp",
                    "family_id": feature_selection["family_id"],
                    "relate_col": feature_selection["relate_col"],
                    "top_code": feature_selection["top_code"],
                    "bottom_code": feature_selection["bottom_code"],
                    "id": id_col,
                },
            )
            spark.catalog.dropTempView("prepped_df_tmp")
            return df_selected

        elif transform == "related_individual_rows":
            return df_selected
        #            df_selected.createOrReplaceTempView("prepped_df_tmp")
        #            relate_filter = (
        #                feature_selection["filter_b"]
        #                if (not (is_a) and "filter_b" in feature_selection)
        #                else None
        #            )
        #            df_selected = link_task.run_register_sql(
        #                None,
        #                template="attach_related_cols_as_rows",
        #                t_ctx={
        #                    "output_col": feature_selection["output_col"],
        #                    "input_cols": feature_selection["input_cols"],
        #                    "prepped_df": "prepped_df_tmp",
        #                    "family_id": feature_selection["family_id"],
        #                    "relate_col": feature_selection["relate_col"],
        #                    "top_code": feature_selection["top_code"],
        #                    "bottom_code": feature_selection["bottom_code"],
        #                    "id": id_col,
        #                    "filter": relate_filter,
        #                },
        #            )
        #            spark.catalog.dropTempView("prepped_df_tmp")

        elif transform == "popularity":
            input_cols = feature_selection.get("input_cols", False)
            output_col = feature_selection["output_col"]

            # this should be a dictionary key:col_name, value:integer to be used for range
            range_col = feature_selection.get("range_col", False)
            range_val = feature_selection.get("range_val", False)

            if range_col and range_val:
                if input_cols:
                    window = (
                        Window.partitionBy([df_selected[col] for col in input_cols])
                        .orderBy(df_selected[range_col])
                        .rangeBetween(-range_val, range_val)
                    )
                else:
                    window = Window.orderBy(df_selected[range_col]).rangeBetween(
                        -range_val, range_val
                    )
            else:
                window = Window.partitionBy([df_selected[col] for col in input_cols])

            df_selected = df_selected.select(
                df_selected["*"], count(lit(1)).over(window).alias(output_col)
            )
            return df_selected

        elif transform == "power":
            input_col = feature_selection["input_col"]
            output_col = feature_selection["output_col"]
            exponent = feature_selection["exponent"]
            df_selected = df_selected.select(
                "*", pow(df_selected[input_col], exponent).alias(output_col)
            )
            return df_selected

        elif transform == "attach_variable":
            input_col = feature_selection["input_column"]  # join key in core data
            output_col = feature_selection[
                "output_column"
            ]  # desired alias for the added variable
            col_to_join_on = feature_selection["col_to_join_on"]  # join key in csv data
            col_to_add = feature_selection["col_to_add"]  # column to add from csv data
            region_dict = feature_selection["region_dict"]  # path to csv data file
            null_filler = feature_selection[
                "null_filler"
            ]  # value to replace null values
            col_type = feature_selection["col_type"]

            df_selected.createOrReplaceTempView("prepped_df_tmp")

            # open up csv file
            link_task.run_register_python(
                name="region_data",
                func=lambda: spark.read.csv(region_dict, header=True, inferSchema=True),
                # persist=True,
            )
            # self.spark.table("region_data").region.cast("int")

            # join the csv file to the dataframe (df_selected)
            df_selected = link_task.run_register_sql(
                None,
                template="attach_variable",
                t_ctx={
                    "input_col": input_col,
                    "output_col": output_col,
                    "prepped_df": "prepped_df_tmp",
                    "col_to_join_on": col_to_join_on,
                    "col_to_add": col_to_add,
                    "region_data": "region_data",
                },
            )
            df_selected = df_selected.fillna(null_filler, subset=[output_col])
            df_selected = df_selected.withColumn(
                output_col, df_selected[output_col].cast(col_type)
            )
            spark.catalog.dropTempView("prepped_df_tmp")
            return df_selected

        else:
            raise ValueError("Invalid transform type for {}".format(str(transform)))

    for feature_selection in not_skipped_feature_selections:
        df_selected = parse_feature_selections(df_selected, feature_selection, is_a)

    def get_transforms(name, is_a):
        to_process = []
        for f in not_skipped_feature_selections:
            if ("override_column_a" in f) and is_a:
                pass
            elif ("override_column_b" in f) and not is_a:
                pass
            elif ("set_value_column_a" in f) and is_a:
                pass
            elif ("set_value_column_b" in f) and not is_a:
                pass
            elif f["transform"] == name:
                to_process.append(f)

        return to_process

    hh_transforms = [
        get_transforms("attach_family_col", is_a),
        get_transforms("related_individual_rows", is_a),
        get_transforms("neighbor_aggregate", is_a),
    ]
    if any(hh_transforms):
        attach_ts, related_ts, neighbor_ts = hh_transforms
        if neighbor_ts:
            group_by = [
                neighbor_ts[0]["sort_column"],
                neighbor_ts[0]["neighborhood_column"],
            ]
        elif related_ts:
            group_by = [related_ts[0]["family_id"]]
        elif attach_ts:
            group_by = [attach_ts[0]["family_id"]]

        df_grouped = df_selected.groupBy(*group_by).agg(
            collect_list(struct("*")).alias("hh_rows")
        )
        neighbor_selects = []
        if neighbor_ts:
            for neighbor_t in neighbor_ts:
                serial_column = neighbor_t["sort_column"]
                window = f""" PARTITION BY {neighbor_t['neighborhood_column']}
                              ORDER BY {serial_column}
                              ROWS BETWEEN {neighbor_t['range']} PRECEDING
                                 AND {neighbor_t['range']} FOLLOWING"""
                output_column = neighbor_t["output_column"]
                output_column_tmp = neighbor_t["output_column"] + "_tmp"
                df_grouped = df_grouped.selectExpr(
                    "*",
                    f"collect_list(hh_rows_get_first_value(hh_rows, '{serial_column}', 'pernum', '{neighbor_t['input_column']}')) OVER ({window}) as {output_column_tmp}",
                )
                df_grouped = df_grouped.selectExpr(
                    "*",
                    f"extract_neighbors({output_column_tmp}, {serial_column}) as {output_column}",
                ).drop(f"{output_column_tmp}")
                neighbor_selects.append(output_column)
        if attach_ts:
            attach_hh_column = spark._jvm.com.isrdi.udfs.AttachHHColumn()
            attach_hh_column.createAttachUDF(
                spark._jsparkSession, df_grouped._jdf, attach_ts, "attach_hh_scala"
            )
            all_cols_but_hh_rows = list(set(df_grouped.columns) - set(["hh_rows"]))
            df_grouped_selects = all_cols_but_hh_rows + [
                "attach_hh_scala(hh_rows) as hh_rows"
            ]
            df_grouped = df_grouped.selectExpr(df_grouped_selects)
        if related_ts:
            attach_rel_rows = spark._jvm.com.isrdi.udfs.AttachRelatedRows()
            a_or_b = "a" if is_a else "b"
            attach_rel_rows.createAttachUDF(
                spark._jsparkSession,
                df_grouped._jdf,
                related_ts,
                id_col,
                a_or_b,
                "attach_rel_scala",
            )
            all_cols_but_hh_rows = list(set(df_grouped.columns) - set(["hh_rows"]))
            df_grouped_selects = all_cols_but_hh_rows + [
                "attach_rel_scala(hh_rows) as hh_rows"
            ]
            df_grouped = df_grouped.selectExpr(df_grouped_selects)
        explode_selects = neighbor_selects + ["explode(hh_rows) as tmp_row"]
        tmp_row_selects = neighbor_selects + ["tmp_row.*"]
        df_selected = df_grouped.selectExpr(*explode_selects).selectExpr(
            *tmp_row_selects
        )

    for feature_selection in post_agg_feature_selections:
        df_selected = parse_feature_selections(df_selected, feature_selection, is_a)
    return df_selected


#  These apply to the column mappings in the current config
def apply_transform(column_select, transform, is_a):
    """Given a dataframe select string return a new string having applied the given transform.
    column_select: A PySpark column type
    transform: The transform info from the current config
    is_a: Is running on dataset 'a' or 'b ?

    See the json_schema config file in config_schemas/config.json for definitions on each transform type.
    """
    transform_type = transform["type"]
    if transform_type == "add_to_a":
        if is_a:
            return column_select + transform["value"]
        else:
            return column_select
    if transform_type == "concat_to_a":
        if is_a:
            return concat(column_select, lit(transform["value"]))
        else:
            return column_select
    elif transform_type == "concat_to_b":
        if is_a:
            return column_select
        else:
            return concat(column_select, lit(transform["value"]))
    elif transform_type == "concat_two_cols":
        return concat(column_select, transform["column_to_append"])
    elif transform_type == "lowercase_strip":
        return lower(trim(column_select))
    elif transform_type == "rationalize_name_words":
        return regexp_replace(column_select, r"[^a-z?'\*\-]+", " ")
    elif transform_type == "remove_qmark_hyphen":
        return regexp_replace(column_select, r"[?\*\-]+", "")
    elif transform_type == "remove_punctuation":
        return regexp_replace(column_select, r"[?\-\\\/\"\':,.\[\]\{\}]+", "")
    elif transform_type == "replace_apostrophe":
        return regexp_replace(column_select, r"'+", " ")
    elif transform_type == "remove_alternate_names":
        return regexp_replace(column_select, r"(\w+)( or \w+)+", "$1")
    elif transform_type == "remove_suffixes":
        suffixes = "|".join(transform["values"])
        suffix_regex = r"\b(?: " + suffixes + r")\s*$"
        return regexp_replace(column_select, suffix_regex, "")
    elif transform_type == "remove_stop_words":
        words = "|".join(transform["values"])
        suffix_regex = r"\b(?:" + words + r")\b"
        return regexp_replace(column_select, suffix_regex, "")
    elif transform_type == "remove_prefixes":
        prefixes = "|".join(transform["values"])
        prefix_regex = "^(" + prefixes + ") "
        return regexp_replace(column_select, prefix_regex, "")
    elif transform_type == "condense_prefixes":
        prefixes = "|".join(transform["values"])
        prefix_regex = r"^(" + prefixes + ") "
        return regexp_replace(column_select, prefix_regex, r"$1")
    elif transform_type == "condense_strip_whitespace":
        return regexp_replace(trim(column_select), r"\s\s+", " ")
    elif transform_type == "remove_one_letter_names":
        return regexp_replace(column_select, r"^((?:\w )+)(\w+)", r"$2")
    elif transform_type == "split":
        return split(column_select, " ")
    elif transform_type == "length":
        return length(column_select)
    elif transform_type == "array_index":
        return column_select[transform["value"]]
    elif transform_type == "mapping":
        mapped_column = column_select
        if transform.get("values", False):
            print(
                "DEPRECATION WARNING: The 'mapping' transform no longer takes the 'values' parameter with a list of mappings in dictionaries; instead each mapping should be its own transform. Please change your config for future releases."
            )
            for mapping in transform["values"]:
                from_regexp = "|".join(["^" + str(f) + "$" for f in mapping["from"]])
                mapped_column = regexp_replace(
                    mapped_column, from_regexp, str(mapping["to"])
                )
        else:
            for key, value in transform["mappings"].items():
                from_regexp = "^" + str(key) + "$"
                mapped_column = regexp_replace(mapped_column, from_regexp, str(value))
        if transform.get("output_type", False) == "int":
            mapped_column = mapped_column.cast(LongType())
        return mapped_column
    elif transform_type == "swap_words":
        mapped_column = column_select
        for swap_from, swap_to in transform["values"].items():
            mapped_column = regexp_replace(
                mapped_column,
                r"(?:(?<=\s)|(?<=^))(" + swap_from + r")(?:(?=\s)|(?=$))",
                swap_to,
            )
        return mapped_column
    elif transform_type == "substring":
        if len(transform["values"]) == 2:
            sub_from = transform["values"][0]
            sub_length = transform["values"][1]
            return column_select.substr(sub_from, sub_length)
        else:
            raise ValueError(
                f"Length of substr transform should be 2. You gave: {transform}"
            )
    elif transform_type == "expand":
        expand_length = transform["value"]
        return array(
            [column_select + i for i in range(-expand_length, expand_length + 1)]
        )
    elif transform_type == "cast_as_int":
        return column_select.cast("int")
    elif transform_type == "divide_by_int":
        divisor = transform["value"]
        return column_select.cast("int") / divisor
    elif transform_type == "when_value":
        threshold = transform["value"]
        if_value = transform["if_value"]
        else_value = transform["else_value"]
        return when(column_select.cast("int") == threshold, if_value).otherwise(
            else_value
        )
    elif transform_type == "get_floor":
        return floor(column_select).cast("int")
    else:
        raise ValueError("Invalid transform type for {}".format(str(transform)))
