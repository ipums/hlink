# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink


def register_dist_tables_and_create_sql(link_task, dist_features):
    """Given a list of distance table comparison features,
        registers the required distance tables and returns
        the distance table join sql statements.

    Parameters
    ----------
    link_task: LinkTask
        the LinkTask used to register the distance tables
    dist_features: list
        a list of comparison features that use distance tables

    Returns
    -------
    A list of the sql join clauses to be used to join in the
    distance tables when creating the comparison features.
    """
    tables_loaded = []
    join_clauses = []
    for feature in dist_features:
        st = feature.get("secondary_table_name", False)
        dt = feature["table_name"]
        if dt not in tables_loaded:
            link_task.run_register_python(
                f"{dt}",
                lambda: link_task.spark.read.csv(
                    f"{feature['distances_file']}", header=True, inferSchema=True
                ),
                persist=True,
            )
            tables_loaded.append(dt)
        if st:
            if st not in tables_loaded:
                link_task.run_register_python(
                    f"{st}",
                    lambda: link_task.spark.read.csv(
                        f"{feature['secondary_distances_file']}",
                        header=True,
                        inferSchema=True,
                    ),
                    persist=True,
                )
                tables_loaded.append(st)
        if feature["key_count"] == 1:
            join_clause = __key_count_1(
                dt, feature["column_name"], feature["loc_a"], feature["loc_b"]
            )
            if join_clause not in join_clauses:
                join_clauses.append(join_clause)
        elif feature["key_count"] == 2:
            join_clause = __key_count_2(
                dt,
                feature["source_column_a"],
                feature["source_column_b"],
                feature["loc_a_0"],
                feature["loc_a_1"],
                feature["loc_b_0"],
                feature["loc_b_1"],
            )

            if join_clause not in join_clauses:
                join_clauses.append(join_clause)
        if st:
            if feature["secondary_key_count"] == 1:
                join_clause = __key_count_1(
                    st,
                    feature["secondary_source_column"],
                    feature["secondary_loc_a"],
                    feature["secondary_loc_b"],
                )

                if join_clause not in join_clauses:
                    join_clauses.append(join_clause)
            elif feature["secondary_key_count"] == 2:
                join_clause = (
                    st,
                    feature["secondary_source_column_a"],
                    feature["secondary_source_column_b"],
                    feature["secondary_loc_a_0"],
                    feature["secondary_loc_a_1"],
                    feature["secondary_loc_b_0"],
                    feature["secondary_loc_b_1"],
                )
                if join_clause not in join_clauses:
                    join_clauses.append(join_clause)
    return join_clauses, tables_loaded


def __key_count_1(table, column, loc_a, loc_b):
    join_clause = (
        f"LEFT JOIN {table} "
        f"ON a.{column} = {table}.{loc_a} "
        f"AND b.{column} = {table}.{loc_b}"
    )
    return join_clause


def __key_count_2(table, column_a, column_b, loc_a_0, loc_a_1, loc_b_0, loc_b_1):
    join_clause = (
        f"LEFT JOIN {table} "
        f"ON a.{column_a} = {table}.{loc_a_0} "
        f"AND a.{column_b} = {table}.{loc_b_0} "
        f"AND b.{column_a} = {table}.{loc_a_1} "
        f"AND b.{column_b} = {table}.{loc_b_1}"
    )
    return join_clause


def get_broadcast_hint(tables_loaded):
    tables = ", ".join(tables_loaded)
    broadcast_hints = f"/*+ BROADCAST({tables}) */"
    return broadcast_hints
