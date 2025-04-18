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
    transforms = {
        "add_to_a": transform_add_to_a,
        "concat_to_a": transform_concat_to_a,
        "concat_to_b": transform_concat_to_b,
        "concat_two_cols": transform_concat_two_cols,
        "lowercase_strip": transform_lowercase_strip,
        "rationalize_name_words": transform_rationalize_name_words,
        "remove_qmark_hyphen": transform_remove_qmark_hyphen,
        "remove_punctuation": transform_remove_punctuation,
        "replace_apostrophe": transform_replace_apostrophe,
        "remove_alternate_names": transform_remove_alternate_names,
        "remove_suffixes": transform_remove_suffixes,
        "remove_stop_words": transform_remove_stop_words,
        "remove_prefixes": transform_remove_prefixes,
        "condense_prefixes": transform_condense_prefixes,
        "condense_strip_whitespace": transform_condense_strip_whitespace,
        "remove_one_letter_names": transform_remove_one_letter_names,
        "split": transform_split,
        "length": transform_length,
        "array_index": transform_array_index,
        "mapping": transform_mapping,
        "swap_words": transform_swap_words,
        "substring": transform_substring,
        "expand": transform_expand,
        "cast_as_int": transform_cast_as_int,
        "divide_by_int": transform_divide_by_int,
        "when_value": transform_when_value,
        "get_floor": transform_get_floor,
    }

    transform_func = transforms.get(transform_type)

    if transform_func is None:
        raise ValueError(f"Invalid transform type for {transform}")

    return transform_func(column_select, transform, context)


def transform_add_to_a(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    is_a = context["dataset"] == "a"
    if is_a:
        return input_col + _require_key(transform, "value")
    else:
        return input_col


def transform_concat_to_a(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    is_a = context["dataset"] == "a"
    if is_a:
        value = _require_key(transform, "value")
        return concat(input_col, lit(value))
    else:
        return input_col


def transform_concat_to_b(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    is_a = context["dataset"] == "a"
    if is_a:
        return input_col
    else:
        value = _require_key(transform, "value")
        return concat(input_col, lit(value))


def transform_concat_two_cols(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    column_to_append = _require_key(transform, "column_to_append")
    return concat(input_col, column_to_append)


def transform_lowercase_strip(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return lower(trim(input_col))


def transform_rationalize_name_words(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return regexp_replace(input_col, r"[^a-z?'\*\-]+", " ")


def transform_remove_qmark_hyphen(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return regexp_replace(input_col, r"[?\*\-]+", "")


def transform_remove_punctuation(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return regexp_replace(input_col, r"[?\-\\\/\"\':,.\[\]\{\}]+", "")


def transform_replace_apostrophe(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return regexp_replace(input_col, r"'+", " ")


def transform_remove_alternate_names(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return regexp_replace(input_col, r"(\w+)( or \w+)+", "$1")


def transform_remove_suffixes(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    values = _require_key(transform, "values")
    suffixes = "|".join(values)
    suffix_regex = r"\b(?: " + suffixes + r")\s*$"
    return regexp_replace(input_col, suffix_regex, "")


def transform_remove_stop_words(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    values = _require_key(transform, "values")
    words = "|".join(values)
    suffix_regex = r"\b(?:" + words + r")\b"
    return regexp_replace(input_col, suffix_regex, "")


def transform_remove_prefixes(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    values = _require_key(transform, "values")
    prefixes = "|".join(values)
    prefix_regex = "^(" + prefixes + ") "
    return regexp_replace(input_col, prefix_regex, "")


def transform_condense_prefixes(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    values = _require_key(transform, "values")
    prefixes = "|".join(values)
    prefix_regex = r"^(" + prefixes + ") "
    return regexp_replace(input_col, prefix_regex, r"$1")


def transform_condense_strip_whitespace(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return regexp_replace(trim(input_col), r"\s\s+", " ")


def transform_remove_one_letter_names(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return regexp_replace(input_col, r"^((?:\w )+)(\w+)", r"$2")


def transform_split(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return split(input_col, " ")


def transform_length(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return length(input_col)


def transform_array_index(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    value = _require_key(transform, "value")
    return input_col[value]


def transform_mapping(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    mapped_column = input_col
    mappings = _require_key(transform, "mappings")

    for key, value in mappings.items():
        from_regexp = f"^{key}$"
        mapped_column = regexp_replace(mapped_column, from_regexp, str(value))

    if transform.get("output_type", False) == "int":
        mapped_column = mapped_column.cast(LongType())

    return mapped_column


def transform_swap_words(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    mapped_column = input_col
    values = _require_key(transform, "values")
    for swap_from, swap_to in values.items():
        mapped_column = regexp_replace(
            mapped_column,
            r"(?:(?<=\s)|(?<=^))(" + swap_from + r")(?:(?=\s)|(?=$))",
            swap_to,
        )
    return mapped_column


def transform_substring(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    values = _require_key(transform, "values")
    if len(values) == 2:
        sub_from = values[0]
        sub_length = values[1]
        return input_col.substr(sub_from, sub_length)
    else:
        raise ValueError(
            f"Length of substr transform should be 2. You gave: {transform}"
        )


def transform_expand(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    expand_length = _require_key(transform, "value")
    return array([input_col + i for i in range(-expand_length, expand_length + 1)])


def transform_cast_as_int(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return input_col.cast("int")


def transform_divide_by_int(
    input_col: Column, transform: dict[str, Any], context
) -> Column:
    divisor = _require_key(transform, "value")
    return input_col.cast("int") / divisor


def transform_when_value(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    threshold = _require_key(transform, "value")
    if_value = _require_key(transform, "if_value")
    else_value = _require_key(transform, "else_value")
    return when(input_col.cast("int") == threshold, if_value).otherwise(else_value)


def transform_get_floor(
    input_col: Column, transform: dict[str, Any], context: dict[str, Any]
) -> Column:
    return floor(input_col).cast("int")
