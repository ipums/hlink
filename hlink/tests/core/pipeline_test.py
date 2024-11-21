import hlink.linking.core.pipeline as pipeline_core


def test_categorical_comparison_features():
    """Catches a bug where comparison features marked as categorical = false
    were still included as categorical. See Issue #81.
    """
    ind_vars = ["birthyr", "m_frstname_jw", "agecat"]
    cols_to_pass = ["age"]
    pipeline_features = []
    comparison_features = [
        {
            "alias": "age",
            "column_names": ["birthyr"],
            "comparison_type": "sql_condition",
            "condition": "1900 - a.birthyr",
        },
        {
            "alias": "agecat",
            "column_names": ["birthyr"],
            "comparison_type": "sql_condition",
            "condition": "1900 - a.birthyr",
            "categorical": True,
        },
        {
            "alias": "m_frstname_jw",
            "column_names": ["m_frstname"],
            "comparison_type": "jaro_winkler",
            "categorical": False,
        },
    ]

    cat_comp_features, _ = pipeline_core._calc_categorical_features(
        ind_vars, cols_to_pass, comparison_features, pipeline_features
    )

    assert set(cat_comp_features) == {"agecat"}
