# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import hlink.linking.core.comparison_feature as comparison_feature_core


def get_comparison_leaves(comp):
    comp_leaves = []

    def __get_comp_leaf(comp, comp_leaves):

        if "comp_a" in comp:
            __get_comp_leaf(comp["comp_a"], comp_leaves)
            __get_comp_leaf(comp["comp_b"], comp_leaves)

        else:
            comp_leaves.append(comp)

    if "comp_a" in comp:
        __get_comp_leaf(comp["comp_a"], comp_leaves)
        __get_comp_leaf(comp["comp_b"], comp_leaves)

    elif "secondary" in comp:
        __get_comp_leaf(comp["threshold_a"], comp_leaves)
        __get_comp_leaf(comp["threshold_b"], comp_leaves)

    else:
        __get_comp_leaf(comp, comp_leaves)

    return comp_leaves


def generate_comparisons(comp, features, id_col):
    """Creates the comparison SQL clause given a comparison and a list of comparison features.

    Parameters
    ----------
    comp: dictionary
        the config dictionary containing the comparison definition
    features: dictionary
        the config dictionary containing the comparison features
    id_col: string
        the id column

    Returns
    -------
    A string of the sql clause to be used for comparison
    filtering after blocking.
    """
    if comp != {}:
        if "comp_a" in comp:
            comp_a_clause = generate_comparisons(comp["comp_a"], features, id_col)
            comp_b_clause = generate_comparisons(comp["comp_b"], features, id_col)
            if comp["operator"] == "AND":
                return f"""
                ({comp_a_clause} AND {comp_b_clause})
                """
            elif comp["operator"] == "OR":
                return f"""
                ({comp_a_clause} OR {comp_b_clause})
                """
        elif "secondary" in comp:
            comp_a = comp["threshold_a"]
            comp_a_clause = f"{comp_a['feature_name']} >= {comp_a['threshold']}"
            comp_b = comp["threshold_b"]
            comp_b_clause = f"{comp_b['feature_name']} >= {comp_b['threshold']}"
            if comp["operator"] == "AND":
                return f"({comp_a_clause} AND {comp_b_clause})"

        else:
            if "column_name" in comp:
                col = comp["column_name"]
            else:
                col = comparison_feature_core.generate_comparison_feature(
                    [f for f in features if f["alias"] == comp["feature_name"]][0],
                    id_col,
                )
            if "comparison_type" in comp:
                comp_type = comp["comparison_type"]
                if comp_type == "threshold":
                    if comp.get("threshold_expr", False):
                        return f"{col} {comp['threshold_expr']}"
                    else:
                        return f"{col} >= {comp['threshold']}"
            else:
                return f"{col}"
    else:
        return ""
