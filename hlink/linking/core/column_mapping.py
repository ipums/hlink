# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink
from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import (
    array,
    col,
    concat,
    floor,
    length,
    lit,
    lower,
    regexp_replace,
    split,
    trim,
    when,
)
from pyspark.sql.types import LongType


def select_column_mapping(
    column_mapping: dict[str, Any],
    df_selected: DataFrame,
    is_a: bool,
    column_selects: list[str],
) -> tuple[DataFrame, list[str]]:
    name = column_mapping["column_name"]
    if "override_column_a" in column_mapping and is_a:
        override_name = column_mapping["override_column_a"]
        column_select = col(override_name)
        if "override_transforms" in column_mapping:
            for transform in column_mapping["override_transforms"]:
                column_select = apply_transform(column_select, transform, is_a)
    elif "override_column_b" in column_mapping and not is_a:
        override_name = column_mapping["override_column_b"]
        column_select = col(override_name)
        if "override_transforms" in column_mapping:
            for transform in column_mapping["override_transforms"]:
                column_select = apply_transform(column_select, transform, is_a)
    elif "set_value_column_a" in column_mapping and is_a:
        value_to_set = column_mapping["set_value_column_a"]
        column_select = lit(value_to_set)
    elif "set_value_column_b" in column_mapping and not is_a:
        value_to_set = column_mapping["set_value_column_b"]
        column_select = lit(value_to_set)
    elif "transforms" in column_mapping:
        column_select = col(name)
        for transform in column_mapping["transforms"]:
            column_select = apply_transform(column_select, transform, is_a)
    else:
        column_select = col(name)

    alias = column_mapping["alias"] if "alias" in column_mapping else name

    column_selects.append(alias)
    return df_selected.withColumn(alias, column_select), column_selects


def _require_key(transform: dict[str, Any], key: str) -> Any:
    """
    Extract a key from a transform, or raise a helpful context-aware error if the
    key is not present.
    """
    try:
        return transform[key]
    except KeyError as e:
        transform_type = transform.get("type", "UNKNOWN")
        raise ValueError(
            f"""Missing required attribute '{key}' for column mapping transform type '{transform_type}'.\n\
            The full provided column mapping transform is\n\
            \n\
            {transform}"""
        ) from e


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
        return input_col + _require_key(transform, "value")
    else:
        return input_col


def col_mapping_concat_to_a(input_col, transform, context) -> Column:
    is_a = context["dataset"] == "a"
    if is_a:
        value = _require_key(transform, "value")
        return concat(input_col, lit(value))
    else:
        return input_col


def col_mapping_concat_to_b(input_col, transform, context) -> Column:
    is_a = context["dataset"] == "a"
    if is_a:
        return input_col
    else:
        value = _require_key(transform, "value")
        return concat(input_col, lit(value))


def col_mapping_concat_two_cols(input_col, transform, context) -> Column:
    column_to_append = _require_key(transform, "column_to_append")
    return concat(input_col, column_to_append)


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
    values = _require_key(transform, "values")
    suffixes = "|".join(values)
    suffix_regex = r"\b(?: " + suffixes + r")\s*$"
    return regexp_replace(input_col, suffix_regex, "")


def col_mapping_remove_stop_words(input_col, transform, context) -> Column:
    values = _require_key(transform, "values")
    words = "|".join(values)
    suffix_regex = r"\b(?:" + words + r")\b"
    return regexp_replace(input_col, suffix_regex, "")


def col_mapping_remove_prefixes(input_col, transform, context) -> Column:
    values = _require_key(transform, "values")
    prefixes = "|".join(values)
    prefix_regex = "^(" + prefixes + ") "
    return regexp_replace(input_col, prefix_regex, "")


def col_mapping_condense_prefixes(input_col, transform, context) -> Column:
    values = _require_key(transform, "values")
    prefixes = "|".join(values)
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
    value = _require_key(transform, "value")
    return input_col[value]


def col_mapping_mapping(input_col, transform, context) -> Column:
    mapped_column = input_col
    mappings = _require_key(transform, "mappings")

    for key, value in mappings.items():
        from_regexp = f"^{key}$"
        mapped_column = regexp_replace(mapped_column, from_regexp, str(value))

    if transform.get("output_type", False) == "int":
        mapped_column = mapped_column.cast(LongType())

    return mapped_column


def col_mapping_swap_words(input_col, transform, context) -> Column:
    mapped_column = input_col
    values = _require_key(transform, "values")
    for swap_from, swap_to in values.items():
        mapped_column = regexp_replace(
            mapped_column,
            r"(?:(?<=\s)|(?<=^))(" + swap_from + r")(?:(?=\s)|(?=$))",
            swap_to,
        )
    return mapped_column


def col_mapping_substring(input_col: Column, transform, context) -> Column:
    values = _require_key(transform, "values")
    if len(values) == 2:
        sub_from = values[0]
        sub_length = values[1]
        return input_col.substr(sub_from, sub_length)
    else:
        raise ValueError(
            f"Length of substr transform should be 2. You gave: {transform}"
        )


def col_mapping_expand(input_col: Column, transform, context) -> Column:
    expand_length = _require_key(transform, "value")
    return array([input_col + i for i in range(-expand_length, expand_length + 1)])


def col_mapping_cast_as_int(input_col: Column, transform, context) -> Column:
    return input_col.cast("int")


def col_mapping_divide_by_int(
    input_col: Column, transform: dict[str, Any], context
) -> Column:
    divisor = _require_key(transform, "value")
    return input_col.cast("int") / divisor


def col_mapping_when_value(
    input_col: Column, transform: dict[str, Any], context
) -> Column:
    threshold = _require_key(transform, "value")
    if_value = _require_key(transform, "if_value")
    else_value = _require_key(transform, "else_value")
    return when(input_col.cast("int") == threshold, if_value).otherwise(else_value)


def col_mapping_get_floor(input_col: Column, transform, context) -> Column:
    return floor(input_col).cast("int")
