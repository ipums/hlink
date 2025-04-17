# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from typing import Any

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
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.ml.feature import NGram, RegexTokenizer, CountVectorizer, MinHashLSH


def _get_transforms(
    feature_selections: list[dict[str, Any]], name: str, is_a: bool
) -> list[dict[str, Any]]:
    """
    Filter the given list of feature selections for those that have the
    transform `name` and are active for the datasource indicated by `is_a`.

    feature_selections: the list of feature selections to filter
    name: the name of the transform to filter for, e.g. "neighbor_aggregate"
    is_a: whether this is for datasource A (True) or datasource B (False)
    """
    to_process = []
    for feature_selection in feature_selections:
        if ("override_column_a" in feature_selection) and is_a:
            pass
        elif ("override_column_b" in feature_selection) and not is_a:
            pass
        elif ("set_value_column_a" in feature_selection) and is_a:
            pass
        elif ("set_value_column_b" in feature_selection) and not is_a:
            pass
        elif feature_selection["transform"] == name:
            to_process.append(feature_selection)

    return to_process


def _parse_feature_selections(
    spark: SparkSession,
    link_task,
    df_selected: DataFrame,
    feature_selection: dict[str, Any],
    id_col: str,
    is_a: bool,
) -> DataFrame:
    """
    Parse the `feature_selection` and add it to `df_selected` as a new column.
    This looks at what type of transform the `feature_selection` is to
    determine how to compute it. Note that this function adds the new column to
    the return data frame lazily and does not `collect()` the data frame.

    spark: the Spark session
    link_task: the current link task
    df_selected: the data frame to use for computation
    feature_selection: the feature selection to compute, which may depend on
                       columns in `df_selected`
    id_col: the identifier column for the data frame
    is_a: whether this is for datasource A (True) or datasource B (False)
    """
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
        if "no_first_pad" in feature_selection and feature_selection["no_first_pad"]:
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
        input_cols = feature_selection["input_columns"]
        output_col = feature_selection["output_column"]
        df_selected = df_selected.withColumn(output_col, array(input_cols))
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
        null_filler = feature_selection["null_filler"]  # value to replace null values
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
        raise ValueError(f"Invalid transform type for {transform}")


def generate_transforms(
    spark: SparkSession,
    df_selected: DataFrame,
    feature_selections: list[dict[str, Any]],
    link_task,
    is_a: bool,
    id_col: str,
) -> DataFrame:
    """Generate feature selection columns and return the input dataframe with these new columns attached.

    Args:
    spark: the Spark session
    df_selected: the input Spark DataFrame
    feature_selections: a list of feature selections to compute
    link_task: the current LinkTask
    is_a: whether this is dataset A (True) or dataset B (False)
    id_col: the name of the identifier column in the input data frame
    """
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

    for feature_selection in not_skipped_feature_selections:
        df_selected = _parse_feature_selections(
            spark, link_task, df_selected, feature_selection, id_col, is_a
        )

    hh_transforms = [
        _get_transforms(not_skipped_feature_selections, "attach_family_col", is_a),
        _get_transforms(
            not_skipped_feature_selections, "related_individual_rows", is_a
        ),
        _get_transforms(not_skipped_feature_selections, "neighbor_aggregate", is_a),
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
        df_selected = _parse_feature_selections(
            spark, link_task, df_selected, feature_selection, id_col, is_a
        )
    return df_selected


#  These apply to the column mappings in the current config
def apply_transform(
    column_select: Column, transform: dict[str, Any], is_a: bool
) -> Column:
    """Return a new column that is the result of applying the given transform
    to the given input column (column_select). The is_a parameter controls the
    behavior of the transforms like "add_to_a" which act differently on
    datasets A and B.

    Args:
    column_select: a PySpark Column
    transform: the transform to apply
    is_a: whether this is dataset A (True) or B (False)
    """
    dataset = "a" if is_a else "b"
    context = {"dataset": dataset}
    transform_type = transform["type"]
    if transform_type == "add_to_a":
        return col_mapping_add_to_a(column_select, transform, context)
    if transform_type == "concat_to_a":
        return col_mapping_concat_to_a(column_select, transform, context)
    elif transform_type == "concat_to_b":
        return col_mapping_concat_to_b(column_select, transform, context)
    elif transform_type == "concat_two_cols":
        return col_mapping_concat_two_cols(column_select, transform, context)
    elif transform_type == "lowercase_strip":
        return col_mapping_lowercase_strip(column_select, transform, context)
    elif transform_type == "rationalize_name_words":
        return col_mapping_rationalize_name_words(column_select, transform, context)
    elif transform_type == "remove_qmark_hyphen":
        return col_mapping_remove_qmark_hyphen(column_select, transform, context)
    elif transform_type == "remove_punctuation":
        return col_mapping_remove_punctuation(column_select, transform, context)
    elif transform_type == "replace_apostrophe":
        return col_mapping_replace_apostrophe(column_select, transform, context)
    elif transform_type == "remove_alternate_names":
        return col_mapping_remove_alternate_names(column_select, transform, context)
    elif transform_type == "remove_suffixes":
        return col_mapping_remove_suffixes(column_select, transform, context)
    elif transform_type == "remove_stop_words":
        return col_mapping_remove_stop_words(column_select, transform, context)
    elif transform_type == "remove_prefixes":
        return col_mapping_remove_prefixes(column_select, transform, context)
    elif transform_type == "condense_prefixes":
        return col_mapping_condense_prefixes(column_select, transform, context)
    elif transform_type == "condense_strip_whitespace":
        return col_mapping_condense_strip_whitespace(column_select, transform, context)
    elif transform_type == "remove_one_letter_names":
        return col_mapping_remove_one_letter_names(column_select, transform, context)
    elif transform_type == "split":
        return col_mapping_split(column_select, transform, context)
    elif transform_type == "length":
        return col_mapping_length(column_select, transform, context)
    elif transform_type == "array_index":
        return col_mapping_array_index(column_select, transform, context)
    elif transform_type == "mapping":
        return col_mapping_mapping(column_select, transform, context)
    elif transform_type == "swap_words":
        return col_mapping_swap_words(column_select, transform, context)
    elif transform_type == "substring":
        return col_mapping_substring(column_select, transform, context)
    elif transform_type == "expand":
        return col_mapping_expand(column_select, transform, context)
    elif transform_type == "cast_as_int":
        return col_mapping_cast_as_int(column_select, transform, context)
    elif transform_type == "divide_by_int":
        return col_mapping_divide_by_int(column_select, transform, context)
    elif transform_type == "when_value":
        return col_mapping_when_value(column_select, transform, context)
    elif transform_type == "get_floor":
        return col_mapping_get_floor(column_select, transform, context)
    else:
        raise ValueError(f"Invalid transform type for {transform}")


def col_mapping_add_to_a(input_col, transform, context) -> Column:
    is_a = context["dataset"] == "a"
    if is_a:
        return input_col + transform["value"]
    else:
        return input_col


def col_mapping_concat_to_a(input_col, transform, context) -> Column:
    is_a = context["dataset"] == "a"
    if is_a:
        return concat(input_col, lit(transform["value"]))
    else:
        return input_col


def col_mapping_concat_to_b(input_col, transform, context) -> Column:
    is_a = context["dataset"] == "a"
    if is_a:
        return input_col
    else:
        return concat(input_col, lit(transform["value"]))


def col_mapping_concat_two_cols(input_col, transform, context) -> Column:
    return concat(input_col, transform["column_to_append"])


def col_mapping_lowercase_strip(input_col, transform, context) -> Column:
    return lower(trim(input_col))


def col_mapping_rationalize_name_words(input_col, transform, context) -> Column:
    return regexp_replace(input_col, r"[^a-z?'\*\-]+", " ")


def col_mapping_remove_qmark_hyphen(input_col, transform, context) -> Column:
    return regexp_replace(input_col, r"[?\*\-]+", "")


def col_mapping_remove_punctuation(input_col, transform, context) -> Column:
    return regexp_replace(input_col, r"[?\-\\\/\"\':,.\[\]\{\}]+", "")


def col_mapping_replace_apostrophe(input_col, transform, context) -> Column:
    return regexp_replace(input_col, r"'+", " ")


def col_mapping_remove_alternate_names(input_col, transform, context) -> Column:
    return regexp_replace(input_col, r"(\w+)( or \w+)+", "$1")


def col_mapping_remove_suffixes(input_col, transform, context) -> Column:
    suffixes = "|".join(transform["values"])
    suffix_regex = r"\b(?: " + suffixes + r")\s*$"
    return regexp_replace(input_col, suffix_regex, "")


def col_mapping_remove_stop_words(input_col, transform, context) -> Column:
    words = "|".join(transform["values"])
    suffix_regex = r"\b(?:" + words + r")\b"
    return regexp_replace(input_col, suffix_regex, "")


def col_mapping_remove_prefixes(input_col, transform, context) -> Column:
    prefixes = "|".join(transform["values"])
    prefix_regex = "^(" + prefixes + ") "
    return regexp_replace(input_col, prefix_regex, "")


def col_mapping_condense_prefixes(input_col, transform, context) -> Column:
    prefixes = "|".join(transform["values"])
    prefix_regex = r"^(" + prefixes + ") "
    return regexp_replace(input_col, prefix_regex, r"$1")


def col_mapping_condense_strip_whitespace(input_col, transform, context) -> Column:
    return regexp_replace(trim(input_col), r"\s\s+", " ")


def col_mapping_remove_one_letter_names(input_col, transform, context) -> Column:
    return regexp_replace(input_col, r"^((?:\w )+)(\w+)", r"$2")


def col_mapping_split(input_col, transform, context) -> Column:
    return split(input_col, " ")


def col_mapping_length(input_col, transform, context) -> Column:
    return length(input_col)


def col_mapping_array_index(input_col, transform, context) -> Column:
    return input_col[transform["value"]]


def col_mapping_mapping(input_col, transform, context) -> Column:
    mapped_column = input_col

    for key, value in transform["mappings"].items():
        from_regexp = f"^{key}$"
        mapped_column = regexp_replace(mapped_column, from_regexp, str(value))

    if transform.get("output_type", False) == "int":
        mapped_column = mapped_column.cast(LongType())

    return mapped_column


def col_mapping_swap_words(input_col, transform, context) -> Column:
    mapped_column = input_col
    for swap_from, swap_to in transform["values"].items():
        mapped_column = regexp_replace(
            mapped_column,
            r"(?:(?<=\s)|(?<=^))(" + swap_from + r")(?:(?=\s)|(?=$))",
            swap_to,
        )
    return mapped_column


def col_mapping_substring(input_col: Column, transform, context) -> Column:
    if len(transform["values"]) == 2:
        sub_from = transform["values"][0]
        sub_length = transform["values"][1]
        return input_col.substr(sub_from, sub_length)
    else:
        raise ValueError(
            f"Length of substr transform should be 2. You gave: {transform}"
        )


def col_mapping_expand(input_col: Column, transform, context) -> Column:
    expand_length = transform["value"]
    return array([input_col + i for i in range(-expand_length, expand_length + 1)])


def col_mapping_cast_as_int(input_col: Column, transform, context) -> Column:
    return input_col.cast("int")


def col_mapping_divide_by_int(
    input_col: Column, transform: dict[str, Any], context
) -> Column:
    divisor = transform["value"]
    return input_col.cast("int") / divisor


def col_mapping_when_value(
    input_col: Column, transform: dict[str, Any], context
) -> Column:
    threshold = transform["value"]
    if_value = transform["if_value"]
    else_value = transform["else_value"]
    return when(input_col.cast("int") == threshold, if_value).otherwise(else_value)


def col_mapping_get_floor(input_col: Column, transform, context) -> Column:
    return floor(input_col).cast("int")
