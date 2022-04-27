# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import itertools


def create_feature_tables(
    link_task, t_ctx_def, advanced_comp_features, hh_comp_features, id_col, table_name
):
    """Creates the table which contains all the comparison features from a
       table that has all the column and feature selections.

    Parameters
    ----------
    link_task: LinkTask
        the link task that is currently being ran
    t_ctx_def: dictionary
        the dictionary of values to pass to the sql templates
    advanced_comp_features: list
        a list of the "advanced" features which
        require potential match aggregation
    id_col: string
        the id column
    table_name: string
        the name of the output table

    Returns
    -------
    The output table with the features.
    """

    has_adv_comp_features = len(advanced_comp_features) > 0
    has_hh_comp_features = len(hh_comp_features) > 0
    gen_comp_feat = has_adv_comp_features or has_hh_comp_features
    config = link_task.link_run.config

    tmp_training_features = link_task.run_register_sql(
        name=f"tmp_{table_name}",
        template="potential_matches_base_features",
        t_ctx=t_ctx_def,
        persist=gen_comp_feat,
    )

    if has_adv_comp_features and not has_hh_comp_features:
        return link_task.run_register_sql(
            table_name,
            template="aggregate_features",
            t_ctx={
                "id": config["id_column"],
                "potential_matches": f"tmp_{table_name}",
                "advanced_comp_features": advanced_comp_features,
            },
            persist=True,
        )

    elif has_hh_comp_features and not has_adv_comp_features:
        return link_task.run_register_sql(
            table_name,
            template="hh_aggregate_features",
            t_ctx={
                "id": config["id_column"],
                "hh_col": config[f"{link_task.training_conf}"].get("hh_col", "serialp"),
                "potential_matches": f"tmp_{table_name}",
                "hh_comp_features": hh_comp_features,
            },
            persist=True,
        )

    elif has_adv_comp_features and has_hh_comp_features:
        af = link_task.run_register_sql(
            table_name,
            template="aggregate_features",
            t_ctx={
                "id": config["id_column"],
                "potential_matches": f"tmp_{table_name}",
                "advanced_comp_features": advanced_comp_features,
            },
        )
        return link_task.run_register_sql(
            af,
            template="hh_aggregate_features",
            t_ctx={
                "id": config["id_column"],
                "potential_matches": f"tmp_{table_name}",
                "hh_comp_features": hh_comp_features,
            },
            persist=True,
        )

    else:
        return link_task.run_register_python(
            table_name, lambda: tmp_training_features, persist=True
        )


def get_features(config, independent_features, pregen_features=[]):
    """Splits apart the comparison features into comp_features,
        advanced_features, and dist_features.

    Parameters
    ----------
    config: dictionary
        the base configuration dictionary
    independent_features: list
        a list of the comparison features to split apart
    pregen_features: dictionary
        a list of features that have been pregenerated and should be skipped

    Returns
    -------
    A 3-tuple of the standard comparison features,
    the advanced comparison features, and the distance features.
    """

    aggregate_features = [
        "hits",
        "hits2",
        "exact_mult",
        "exact_all_mult",
        "exact_all_mult2",
    ]

    hh_aggregate_features = ["jw_max_a", "jw_max_b"]

    all_comp_features = config["comparison_features"]
    advanced_comp_features = [
        f for f in all_comp_features if f["alias"] in aggregate_features
    ] + [f for f in independent_features if f in aggregate_features]
    hh_comp_features = [
        f for f in all_comp_features if f["alias"] in hh_aggregate_features
    ] + [f for f in independent_features if f in hh_aggregate_features]
    derived_comp_features = [
        f
        for f in all_comp_features
        if f["alias"] not in aggregate_features
        and f["alias"] not in pregen_features
        and f["alias"] not in hh_aggregate_features
    ]
    derived_aliases = [f["alias"] for f in derived_comp_features]

    dist_features = [
        c
        for c in all_comp_features
        if c["comparison_type"] == "geo_distance" and c["alias"] not in pregen_features
    ]

    if len({"exact_mult"} & set(advanced_comp_features)) > 0 and (
        ("exact" not in derived_aliases)
    ):
        raise KeyError(
            'In order to calculate "exact_mult", "exact" needs to be added to the list of comparison features in your configuration.'
        )

    if len(
        set(["exact_all_mult", "exact_all_mult2"]) & set(advanced_comp_features)
    ) > 0 and (("exact_all" not in derived_aliases)):
        raise KeyError(
            'In order to calculate "exact_all_mult", or "exact_all_mult2", "exact_all" needs to be added to the list of comparison features in your configuration.'
        )

    comp_features = ",\n ".join(
        [
            generate_comparison_feature(f, config["id_column"], include_as=True)
            for f in derived_comp_features
        ]
    )
    return comp_features, advanced_comp_features, hh_comp_features, dist_features


def generate_comparison_feature(feature, id_col, include_as=False):
    """Returns an SQL expression for a given feature.

    Parameters
    ----------
    feature: dictionary
        a comparison feature from the config
    id_col: string
        the id column
    include_as: boolean
        if true, then the expression will include "as {alias}",
        where `alias` is the alias of the given feature

    Returns
    -------
    A string containing the sql expression.
    """
    comp_type = feature["comparison_type"]

    if comp_type == "sql_condition":
        expr = feature["condition"]

    elif comp_type == "maximum_jaro_winkler":
        columns = feature["column_names"]
        comps = ", ".join(
            [
                f"jw(nvl(a.{col1}, ''), nvl(b.{col2}, ''))"
                for col1, col2 in itertools.product(columns, columns)
            ]
        )
        expr = f"GREATEST({comps})"

    elif comp_type == "jaro_winkler":
        col = feature["column_name"]
        expr = f"jw(nvl(a.{col}, ''), nvl(b.{col}, ''))"

    elif comp_type == "jaro_winkler_street":
        col = feature["column_name"]
        boundary_col = feature["boundary"]
        expr = f"IF(a.{boundary_col} = b.{boundary_col}, jw(nvl(a.{col},''), nvl(b.{col}, '')), 0)"

    elif comp_type == "max_jaro_winkler":
        col = feature["column_name"]
        expr = f"jw_max(a.{col}, b.{col})"

    elif comp_type == "equals":
        col = feature["column_name"]
        expr = f"a.{col} IS NOT DISTINCT FROM b.{col}"

    elif comp_type == "f1_match":
        fi = feature["first_init_col"]
        mi0 = feature["mid_init_cols"][0]
        mi1 = feature["mid_init_cols"][1]
        expr = (
            f"CASE WHEN ("
            f"(a.{fi} IS NOT DISTINCT FROM b.{fi}) OR "
            f"(a.{fi} IS NOT DISTINCT FROM b.{mi0}) OR "
            f"(a.{fi} IS NOT DISTINCT FROM b.{mi1})"
            f") THEN 1 ELSE 2 END"
        )

    elif comp_type == "f2_match":
        fi = feature["first_init_col"]
        mi0 = feature["mid_init_cols"][0]
        mi1 = feature["mid_init_cols"][1]
        expr = (
            f"CASE WHEN ((a.{mi0} == '') OR (a.{mi0} IS NULL)) THEN 0 WHEN ("
            f"(a.{mi0} IS NOT DISTINCT FROM b.{fi}) OR "
            f"((a.{mi1} IS NOT NULL) AND (a.{mi1} IS NOT DISTINCT FROM b.{fi})) OR "
            f"(a.{mi0} IS NOT DISTINCT FROM b.{mi0}) OR "
            f"(a.{mi0} IS NOT DISTINCT FROM b.{mi1}) OR "
            f"((a.{mi1} IS NOT NULL) AND (a.{mi1} IS NOT DISTINCT FROM b.{mi0})) OR "
            f"((a.{mi1} IS NOT NULL) AND (a.{mi1} IS NOT DISTINCT FROM b.{mi1}))"
            f") THEN 1 ELSE 2 END"
        )

    elif comp_type == "not_equals":
        col = feature["column_name"]
        expr = f"a.{col} IS DISTINCT FROM b.{col}"

    elif comp_type == "equals_as_int":
        col = feature["column_name"]
        expr = f"CAST(a.{col} = b.{col} as INT)"

    elif comp_type == "all_equals":
        cols = feature["column_names"]
        all_equals = " AND ".join([f"a.{col} = b.{col}" for col in cols])
        expr = f"{all_equals}"

    elif comp_type == "not_zero_and_not_equals":
        col = feature["column_name"]
        expr = f"a.{col} is not null and b.{col} is not null and a.{col} != 0 AND b.{col} != 0 and a.{col} IS DISTINCT FROM b.{col}"

    elif comp_type == "or":
        expr_a = generate_comparison_feature(feature["comp_a"], id_col)
        expr_b = generate_comparison_feature(feature["comp_b"], id_col)
        if "comp_d" in feature:
            expr_d = generate_comparison_feature(feature["comp_d"], id_col)
            expr_c = generate_comparison_feature(feature["comp_c"], id_col)
            expr = f"{expr_a}  OR {expr_b} OR {expr_c} OR {expr_d}"
        elif "comp_c" in feature:
            expr_c = generate_comparison_feature(feature["comp_c"], id_col)
            expr = f"{expr_a} OR {expr_b} OR {expr_c}"
        else:
            expr = f"{expr_a} OR {expr_b}"

    elif comp_type == "and":
        expr_a = generate_comparison_feature(feature["comp_a"], id_col)
        expr_b = generate_comparison_feature(feature["comp_b"], id_col)
        if "comp_d" in feature:
            expr_d = generate_comparison_feature(feature["comp_d"], id_col)
            expr_c = generate_comparison_feature(feature["comp_c"], id_col)
            expr = f"{expr_a}  AND {expr_b} AND {expr_c} AND {expr_d}"
        elif "comp_c" in feature:
            expr_c = generate_comparison_feature(feature["comp_c"], id_col)
            expr = f"{expr_a} AND {expr_b} AND {expr_c}"
        else:
            expr = f"{expr_a} AND {expr_b}"

    elif comp_type == "times":
        expr_a = generate_comparison_feature(feature["comp_a"], id_col)
        expr_b = generate_comparison_feature(feature["comp_b"], id_col)
        expr = f"CAST({expr_a} as float) * CAST({expr_b} as float)"

    elif comp_type == "caution_comp_3":
        expr_a = generate_comparison_feature(feature["comp_a"], id_col)
        expr_b = generate_comparison_feature(feature["comp_b"], id_col)
        expr_c = generate_comparison_feature(feature["comp_c"], id_col)
        expr = f"({expr_a}  OR {expr_b}) AND {expr_c}"

    elif comp_type == "caution_comp_4":
        expr_a = generate_comparison_feature(feature["comp_a"], id_col)
        expr_b = generate_comparison_feature(feature["comp_b"], id_col)
        expr_c = generate_comparison_feature(feature["comp_c"], id_col)
        expr_d = generate_comparison_feature(feature["comp_d"], id_col)
        expr = f"({expr_a}  OR {expr_b} OR {expr_c}) AND {expr_d}"

    elif comp_type == "caution_comp_3_012":
        expr_a = generate_comparison_feature(feature["comp_a"], id_col)
        expr_b = generate_comparison_feature(feature["comp_b"], id_col)
        expr_c = generate_comparison_feature(feature["comp_c"], id_col)
        expr = (
            f"CASE WHEN CAST({expr_c} as string) == 'false' THEN 2 "
            f"WHEN ({expr_a} OR {expr_b}) AND {expr_c} THEN 1 "
            f"ELSE 0 END"
        )

    elif comp_type == "caution_comp_4_012":
        expr_a = generate_comparison_feature(feature["comp_a"], id_col)
        expr_b = generate_comparison_feature(feature["comp_b"], id_col)
        expr_c = generate_comparison_feature(feature["comp_c"], id_col)
        expr_d = generate_comparison_feature(feature["comp_d"], id_col)
        expr = (
            f"CASE WHEN CAST({expr_d} as string) == 'false' THEN 2 "
            f"WHEN (({expr_a}) OR ({expr_b}) OR ({expr_c})) AND ({expr_d}) THEN 1 "
            f"ELSE 0 END"
        )

    elif comp_type == "any_equals":
        col1, col2 = feature["column_names"]
        expr = f"""
        (
          ( a.{col1} = b.{col1} OR a.{col1} = b.{col2} )
          AND
          nvl(a.{col1}, '') != ''
        )
        OR
        (
          ( a.{col2} = b.{col1})
          AND
          nvl(a.{col2}, '') != ''
        )
        """

    elif comp_type == "either_are_1":
        col = feature["column_name"]
        expr = f"(a.{col} = 1 OR b.{col} = 1)"

    elif comp_type == "either_are_0":
        col = feature["column_name"]
        expr = f"(a.{col} = 0 OR b.{col} = 0)"

    elif comp_type == "second_gen_imm":
        col = feature["column_name"]
        expr = f"(a.{col} = 2 OR a.{col} = 3 OR a.{col} = 4)"

    elif comp_type == "rel_jaro_winkler":
        col = feature["column_name"]
        if "jw_threshold" in feature:
            jw_threshold = feature["jw_threshold"]
        else:
            jw_threshold = 0.8
            print(
                f"WARNING: No jw_threshold defined; Setting jw_threshold for rel_jaro_winkler comparison feature for {col} to {jw_threshold}"
            )
        if feature.get("age_threshold", False):
            age_threshold = feature["age_threshold"]
        else:
            age_threshold = 5
            print(
                f"WARNING: No age_threshold defined; Setting age_threshold for rel_jaro_winkler comparison feature for {col} to {age_threshold}"
            )
        histid = feature.get("histid_col", "histid")
        name = feature.get("name_col", "namefrst_std")
        byr = feature.get("birthyr_col", "birthyr")
        sex = feature.get("sex_col", "sex")
        expr = f"rel_jw(a.{col}, b.{col}, string({jw_threshold}), string({age_threshold}), map('name','{name}','byr','{byr}','sex','{sex}'))"

    elif comp_type == "extra_children":
        col = feature["column_name"]
        if "jw_threshold" in feature:
            jw_threshold = feature["jw_threshold"]
        else:
            jw_threshold = 0.8
            print(
                f"WARNING: No jw_threshold defined; Setting jw_threshold for rel_jaro_winkler comparison feature for {col} to {jw_threshold}"
            )
        if feature.get("age_threshold", False):
            age_threshold = feature["age_threshold"]
        else:
            age_threshold = 5
            print(
                f"WARNING: No age_threshold defined; Setting age_threshold for rel_jaro_winkler comparison feature for {col} to {age_threshold}"
            )
        year_b = feature.get("year_b", "year_b")
        relate = feature.get("relate_col", "relate")
        histid = id_col if id_col is not None else "histid"
        name = feature.get("name_col", "namefrst_std")
        byr = feature.get("birthyr_col", "birthyr")
        sex = feature.get("sex_col", "sex")
        expr = f"extra_children(a.{col}, b.{col}, string({year_b}), a.{relate}, b.{relate}, string({jw_threshold}), string({age_threshold}), map('histid', '{histid}', 'name','{name}','byr','{byr}','sex','{sex}'))"

    elif comp_type == "jaro_winkler_rate":
        col = feature["column_name"]
        if "jw_threshold" in feature:
            jw_threshold = feature["jw_threshold"]
        else:
            jw_threshold = 0.8
            print(
                f"WARNING: No jw_threshold defined; Setting jw_threshold for jaro_winkler_rate comparison feature for {col} to {jw_threshold}"
            )
        expr = f"jw_rate(a.{col}, b.{col}, string({jw_threshold}))"

    elif comp_type == "sum":
        col = feature["column_name"]
        expr = f"a.{col} + b.{col}"

    elif comp_type == "hh_compare_rate":
        col = feature["column_name"]
        expr = f"hh_compare_rate(a.{col}, b.{col})"

    elif comp_type == "length_b":
        col = feature["column_name"]
        expr = f"size(b.{col})"

    elif comp_type == "abs_diff":
        col = feature["column_name"]
        ne = feature.get("not_equals", False)
        if ne:
            expr = f"case when cast(b.{col} as INT) != {ne} and cast(a.{col} as INT) != {ne} then abs(CAST(b.{col} as INT) - CAST(a.{col} as INT)) else -1 end"
        else:
            expr = f"abs(CAST(b.{col} as INT) - CAST(a.{col} as INT))"

    elif comp_type == "b_minus_a":
        col = feature["column_name"]
        ne = feature.get("not_equals", False)
        if ne:
            expr = f"case when cast(b.{col} as INT) != {ne} and cast(a.{col} as INT) != {ne} then CAST(b.{col} as INT) - CAST(a.{col} as INT) else -1 end"
        else:
            expr = f"CAST(b.{col} as INT) - CAST(a.{col} as INT)"

    elif comp_type == "has_matching_element":
        col = feature["column_name"]
        expr = f"has_matching_element(a.{col}, b.{col})"
    elif comp_type == "geo_distance":
        distance_col = feature["distance_col"]
        dt = feature["table_name"]
        st = feature.get("secondary_table_name", False)
        if st:
            st_distance_col = feature["secondary_distance_col"]
            expr = f"IF({dt}.{distance_col} IS NOT NULL, {dt}.{distance_col}, {st}.{st_distance_col})"
        else:
            expr = f"{dt}.{distance_col}"

    elif comp_type == "fetch_a":
        col = feature["column_name"]
        expr = f"a.{col}"

    elif comp_type == "fetch_b":
        col = feature["column_name"]
        expr = f"b.{col}"

    elif comp_type == "fetch_td":
        col = feature["column_name"]
        expr = f"pm.{col}"

    elif comp_type == "new_marr":
        col = feature["column_name"]
        if "upper_threshold" not in feature:
            feature["upper_threshold"] = 10
            print(
                f"WARNING: No upper_threshold defined; Setting upper_threshold for new_marr comparison feature for {col} to 10"
            )
        expr = f"CAST(b.{col} as INT)"

    elif comp_type == "existing_marr":
        col = feature["column_name"]
        if "lower_threshold" not in feature:
            feature["lower_threshold"] = 10
            print(
                f"WARNING: No lower_threshold defined; Setting lower_threshold for existing_marr comparison feature for {col} to 10"
            )
        expr = f"CAST(b.{col} as INT)"

    elif comp_type == "parent_step_change":
        col = feature["column_name"]
        expr = f"(CAST(a.{col} as INT) > 0) IS DISTINCT FROM (CAST(b.{col} as INT) > 0)"

    elif comp_type == "present_both_years":
        col = feature["column_name"]
        expr = f"a.{col} IS NOT NULL AND a.{col} > 0 AND b.{col} IS NOT NULL AND b.{col} > 0"

    elif comp_type == "neither_are_null":
        col = feature["column_name"]
        expr = f"a.{col} IS NOT NULL AND b.{col} IS NOT NULL and a.{col} != '' and b.{col} != ''"

    elif comp_type == "present_and_matching_categorical":
        col = feature["column_name"]
        expr = f"IF(a.{col} IS NOT NULL AND b.{col} IS NOT NULL AND CAST(a.{col} as STRING) != '' and CAST(b.{col} as STRING) != '', IF(a.{col} IS DISTINCT FROM b.{col}, 1, 0), 2)"

    elif comp_type == "present_and_equal_categorical_in_universe":
        col = feature["column_name"]
        niu = feature["NIU"]
        expr = f"IF(a.{col} IS NOT NULL AND b.{col} IS NOT NULL AND a.{col} != {niu} AND b.{col} != {niu} AND CAST(a.{col} as string) != '' and CAST(b.{col} as string) != '', IF(a.{col} IS DISTINCT FROM b.{col}, 0, 1), 0)"

    elif comp_type == "present_and_not_equal":
        col = feature["column_name"]
        expr = f"IF(a.{col} IS NOT NULL AND b.{col} IS NOT NULL AND cast(a.{col} as string) != '' and cast(b.{col} as string) != '' and a.{col} > 0, IF(a.{col} IS DISTINCT FROM b.{col}, TRUE, FALSE), FALSE)"

    else:
        raise ValueError(f"No comparison type: {feature['comparison_type']}")

    if feature.get("power", False):
        exponent = feature["power"]
        expr = f"POWER(CAST({expr} as INT), {exponent})"

    if feature.get("threshold", False):
        threshold = feature["threshold"]
        expr = f"{expr} IS NOT NULL and {expr} >= {threshold}"
    elif feature.get("lower_threshold", False):
        lower_threshold = feature["lower_threshold"]
        expr = f"{expr} IS NOT NULL and {expr} >= {lower_threshold}"
    elif feature.get("upper_threshold", False):
        upper_threshold = feature["upper_threshold"]
        expr = f"{expr} IS NOT NULL and {expr} <= {upper_threshold}"
    elif feature.get("gt_threshold", False):
        gt_threshold = feature["gt_threshold"]
        expr = f"{expr} IS NOT NULL and {expr} > {gt_threshold}"
    elif feature.get("btwn_threshold", False):
        bt0 = feature["btwn_threshold"][0]
        bt1 = feature["btwn_threshold"][1]
        expr = f"{expr} IS NOT NULL and {expr} >= {bt0} and {expr} <= {bt1}"

    if feature.get("look_at_addl_var", False):
        addl_var = feature["addl_var"]
        check_val_expr = feature["check_val_expr"]
        else_val = feature["else_val"]
        datasource = feature["datasource"]
        expr = f"CASE WHEN {datasource}.{addl_var} {check_val_expr} then {expr} else {else_val} END"

    if include_as:
        full_expr = f"({expr})" + f" as {feature['alias']}"
    else:
        full_expr = expr

    return full_expr
