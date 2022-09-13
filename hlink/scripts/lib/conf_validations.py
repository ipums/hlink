# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.sql.utils import AnalysisException
from os import path
import colorama


def print_checking(section: str):
    print(f"Checking {section}...", end=" ")


def print_ok():
    print(colorama.Fore.GREEN + "OK" + colorama.Style.RESET_ALL)


def analyze_conf(link_run):
    """Print an analysis of the configuration of the `link_run`."""
    colorama.init()

    try:
        print_checking("datasource_a")
        df_a = parse_datasource(link_run, "datasource_a")
        check_datasource(link_run.config, df_a, "A")
        print_ok()

        print_checking("datasource_b")
        df_b = parse_datasource(link_run, "datasource_b")
        check_datasource(link_run.config, df_b, "B")
        print_ok()

        print_checking("filters")
        check_filters(link_run.config, df_a, df_b)
        print_ok()

        print_checking("column_mappings")
        columns_available = check_column_mappings(link_run.config, df_a, df_b)
        print_ok()

        print_checking("substitution_columns")
        check_substitution_columns(link_run.config, columns_available)
        print_ok()

        print_checking("feature_selections")
        check_feature_selections(link_run.config, columns_available)
        print_ok()

        print_checking("blocking")
        check_blocking(link_run.config, columns_available)
        print_ok()

        print_checking("comparison_features")
        comp_features = check_comparison_features(link_run.config, columns_available)
        print_ok()

        print_checking("comparisons")
        check_comparisons(link_run.config, comp_features)
        print_ok()

        print_checking("pipeline_features")
        check_pipeline_features(link_run.config, comp_features)
        print_ok()

        print_checking("training")
        check_training(link_run.config, comp_features)
        print_ok()

        print_checking("hh_training")
        check_hh_training(link_run.config, comp_features)
        print_ok()
    finally:
        colorama.deinit()


def check_hh_training(config, comp_features):
    comp_features += ["jw_max_a", "jw_max_b"]
    hh_training = config.get("hh_training")
    if hh_training is None:
        return
    independent_vars = hh_training.get("independent_vars")
    if independent_vars is None:
        raise ValueError(
            "No independent_vars value specified in the [training] section."
        )
    for var in independent_vars:
        if var not in comp_features:
            raise ValueError(
                f"Within [training] the independent_var: '{var}' does not exist. Please add a specification as a [[comparison_feature]] or a [[pipeline_feature]]."
            )


def check_training(config, comp_features):
    comp_features += ["hits", "hits2", "exact_mult", "exact_all_mult"]
    training = config.get("training")
    if training is None:
        return
    independent_vars = training.get("independent_vars")
    if independent_vars is None:
        raise ValueError(
            "No independent_vars value specified in the [training] section."
        )
    for var in independent_vars:
        if var not in comp_features:
            raise ValueError(
                f"Within [training] the independent_var: '{var}' does not exist. Please add a specification as a [[comparison_feature]] or a [[pipeline_feature]]."
            )


def check_pipeline_features(config, comp_features):
    pipeline_features = config.get("pipeline_features")
    if pipeline_features is None:
        return
    for p in pipeline_features:
        input_column = p.get("input_column")
        if (input_column is not None) and (input_column not in comp_features):
            raise ValueError(
                f"Within [[pipeline_features]] the input_column: '{input_column}' is not available from a previous [[comparison_features]] or [[pipeline_features]] section. \n Available columns: \n {comp_features}"
            )
        input_columns = p.get("input_columns")
        if input_columns is not None:
            for c in input_columns:
                if c not in comp_features:
                    raise ValueError(
                        f"Within [[pipeline_features]] the input_column: '{c}' is not available from a previous [[comparison_features]] or [[pipeline_features]] section. \n Available columns: \n {comp_features}"
                    )
        output_column = p.get("output_column")
        if output_column is None:
            raise ValueError(
                "Within [[pipeline_features]] no 'output_column' specified for {p}."
            )
        comp_features.append(output_column)


def check_comparisons(config, comp_features):
    comparisons = config.get("comparisons")
    if comparisons is None:
        raise ValueError(
            "No [comparisons] section exists. Please add a [comparisons] section."
        )
    comp_a = comparisons.get("comp_a")
    comp_b = comparisons.get("comp_b")
    if comp_a is not None:
        feature_name = comp_a.get("feature_name")
        if (feature_name is not None) and (feature_name not in comp_features):
            raise ValueError(
                f"Within [comparisons] the feature_name '{feature_name}' is not available. Please add a corresponding feature in the [[comparison_features]] section. \n Available features: \n {comp_features}"
            )
    if comp_b is not None:
        feature_name = comp_b.get("feature_name")
        if (feature_name is not None) and (feature_name not in comp_features):
            raise ValueError(
                f"Within [comparisons] the feature_name '{feature_name}' is not available. Please add a corresponding feature in the [[comparison_features]] section. \n Available features: \n {comp_features}"
            )


def check_comparison_features(config, columns_available):
    comps = []
    comparison_features = config.get("comparison_features")
    if comparison_features is None:
        raise ValueError(
            "No [[comparison_features]] exist. Please add [[comparison_features]]."
        )
    for c in comparison_features:
        alias = c.get("alias")
        if alias is None:
            raise ValueError(
                f"No alias exists for a [[comparison_features]]: {c}. Please add an 'alias'."
            )
        column_name = c.get("column_name") or c.get("first_init_col")
        column_names = c.get("column_names") or c.get("mid_init_cols")
        if column_name is not None:
            if column_name not in columns_available:
                raise ValueError(
                    f"Within [[comparison_features]] the 'column_name' {column_name} is not available from a previous [[column_mappings]] or [[feature_selections]]: {c}"
                )
        if column_names is not None:
            for cname in column_names:
                if cname not in columns_available:
                    raise ValueError(
                        f"Within [[comparison_features]] the 'column_name' {cname} is not available from a previous [[column_mappings]] or [[feature_selections]]: {c}"
                    )
        comps.append(alias)
    return comps


def check_blocking(config, columns_available):
    blockings = config.get("blocking")
    if blockings is None:
        raise ValueError("No [[blocking]] exist. Please add blocking.")
    for b in blockings:
        column_name = b.get("derived_from") or b.get("column_name")
        if column_name is None:
            raise ValueError(f"Within [[blocking]] no column name is specified: {b}.")
        if column_name not in columns_available:
            raise ValueError(
                f"Within [[blocking]] the column_name of '{column_name}' is not available from an earlier [[column_mappings]] or [[feature_selections]]. \n Available columns: \n {columns_available}"
            )


def check_feature_selections(config, columns_available):
    feature_selections = config.get("feature_selections")
    if feature_selections is None:
        return
    for f in feature_selections:
        input_column = f.get("input_column")
        output_column = f.get("output_column") or f.get("output_col")
        other_col = f.get("other_col")
        if input_column is not None and input_column not in columns_available:
            raise ValueError(
                f"Within [[feature_selections]] the input_column: '{input_column}' is not created by an earlier [[column_mappings]] or [[feature_selections]]. \n Available Columns: \n {columns_available}."
            )
        if other_col is not None and other_col not in columns_available:
            raise ValueError(
                f"Within [[feature_selections]] the other_col: '{other_col}' is not created by an earlier [[column_mappings]] or [[feature_selections]]. \n Available Columns: \n {columns_available}."
            )
        if output_column is None:
            raise ValueError(
                f"No 'output_column' or 'output_col' value for [[feature_selections]]: {f}"
            )
        columns_available.append(output_column)


def check_substitution_columns(config, columns_available):
    substitution_columns = config.get("substitution_columns")
    if substitution_columns is None:
        return
    for s in substitution_columns:
        column_name = s.get("column_name")
        substitutions = s.get("substitutions")
        if column_name is None:
            raise ValueError("Within [[substitution_columns]] no 'column_name' exists.")
        if substitutions is None:
            raise ValueError(
                "Within [[substitution_columns]] no [[substitution_columns.substitutions]] exists."
            )
        for sub in substitutions:
            join_column = sub.get("join_column")
            join_value = sub.get("join_value")
            f = sub.get("substitution_file")
            if join_column is None or join_column not in columns_available:
                raise ValueError(
                    f"Within [[substitution_columns.substitutions]] the join_column '{join_column}' does not exist or is not available within columns specificed within [[column_mappings]]. \nList of available columns: \n {columns_available}"
                )
            if join_value is None:
                raise ValueError(
                    " Within [[substitution_columns.substitutions]] no 'join_value' exists."
                )
            if f is None or not path.exists(f):
                raise ValueError(
                    f" Within [[substitution_columns.substitutions]] no 'substitution_file' exists or does not point to an existing file: {f}"
                )


def check_column_mappings(config, df_a, df_b):
    column_mappings = config.get("column_mappings")
    if not column_mappings:
        raise ValueError("No [[column_mappings]] exist in the conf file.")
    columns_available = []
    for c in column_mappings:
        alias = c.get("alias")
        column_name = c.get("column_name")
        set_value_column_a = c.get("set_value_column_a")
        set_value_column_b = c.get("set_value_column_b")
        if not column_name:
            raise ValueError(
                f"The following [[column_mappings]] has no 'column_name' attribute: {c}"
            )
        if set_value_column_a is None:
            if column_name.lower() not in [c.lower() for c in df_a.columns]:
                if column_name not in columns_available:
                    raise ValueError(
                        f"Within a [[column_mappings]] the column_name: '{column_name}' does not exist in datasource_a and no previous [[column_mapping]] alias exists for it. \nColumn mapping: {c}. \nAvailable columns: \n {df_a.columns}"
                    )
        if set_value_column_b is None:
            if column_name.lower() not in [c.lower() for c in df_b.columns]:
                if column_name not in columns_available:
                    raise ValueError(
                        f"Within a [[column_mappings]] the column_name: '{column_name}' does not exist in datasource_b and no previous [[column_mapping]] alias exists for it. Column mapping: {c}. Available columns: \n {df_b.columns}"
                    )
        if alias:
            columns_available.append(alias)
        else:
            columns_available.append(column_name)
    return columns_available


def check_filters(config, df_a, df_b):
    filters = config.get("filter")
    if not filters:
        return
    for f in filters:
        expression = f.get("expression")
        if not expression:
            raise ValueError("A [[filter]] has no expression value in the config.")
        try:
            df_a.where(expression)
            df_b.where(expression)
        except AnalysisException as e:
            raise ValueError(
                f"Within a [[filter]] the expression '{expression}' is not valid. Spark gives the following error: {e}."
            )


def parse_datasource(link_run, section_name: str):
    datasource = link_run.config.get(section_name)

    if not datasource:
        raise ValueError(f"Section [{section_name}] does not exist in config.")

    parquet_file = datasource.get("parquet_file")
    file = datasource.get("file")

    if not parquet_file and not file:
        raise ValueError(
            f"Within [{section_name}] neither 'parquet_file' nor 'file' exist."
        )
    if parquet_file and file:
        raise ValueError(
            f"Within [{section_name}] both 'parquet_file' and 'file' exist."
        )

    # Now we know that either file or parquet_file was provided, but not both.
    if parquet_file:
        if not path.exists(parquet_file):
            raise ValueError(
                f"Within [{section_name}] path of parquet file {parquet_file} does not exist."
            )
        return link_run.spark.read.parquet(parquet_file)
    else:
        if not path.exists(file):
            raise ValueError(
                f"Within [{section_name}] path of file {file} does not exist."
            )
        _, file_extension = path.splitext(file)
        if file_extension == ".csv":
            return link_run.spark.read.csv(file, header=True)
        elif file_extension == ".parquet":
            return link_run.spark.read.parquet(file)
        else:
            raise ValueError(
                f"Within [{section_name}] file {file} is neither a CSV file nor a parquet file."
            )


def check_datasource(config, df, a_or_b):
    id_column = config["id_column"]
    input_columns = map((lambda s: s.lower()), df.columns)
    if id_column.lower() not in input_columns:
        raise ValueError(f"Datasource {a_or_b} is missing the id column '{id_column}'")
