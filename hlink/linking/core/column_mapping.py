# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.sql.functions import col, lit
import hlink.linking.core.transforms as transforms_core


def select_column_mapping(column_mapping, df_selected, is_a, column_selects):
    name = column_mapping["column_name"]
    if "override_column_a" in column_mapping and is_a:
        override_name = column_mapping["override_column_a"]
        column_select = col(override_name)
        if "override_transforms" in column_mapping:
            for transform in column_mapping["override_transforms"]:
                column_select = transforms_core.apply_transform(
                    column_select, transform, is_a
                )
    elif "override_column_b" in column_mapping and not is_a:
        override_name = column_mapping["override_column_b"]
        column_select = col(override_name)
        if "override_transforms" in column_mapping:
            for transform in column_mapping["override_transforms"]:
                column_select = transforms_core.apply_transform(
                    column_select, transform, is_a
                )
    elif "set_value_column_a" in column_mapping and is_a:
        value_to_set = column_mapping["set_value_column_a"]
        column_select = lit(value_to_set)
    elif "set_value_column_b" in column_mapping and not is_a:
        value_to_set = column_mapping["set_value_column_b"]
        column_select = lit(value_to_set)
    elif "transforms" in column_mapping:
        column_select = col(name)
        for transform in column_mapping["transforms"]:
            column_select = transforms_core.apply_transform(
                column_select, transform, is_a
            )
    else:
        column_select = col(name)

    alias = column_mapping["alias"] if "alias" in column_mapping else name

    column_selects.append(alias)
    return df_selected.withColumn(alias, column_select), column_selects
