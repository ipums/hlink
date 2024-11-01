import pytest

from hlink.linking.core.comparison import generate_comparisons, get_comparison_leaves


def test_get_comparison_leaves_base_case() -> None:
    """
    A comparison with no sub-comparisons (comp_a and comp_b) is itself the only leaf.
    """
    comparison = {
        "comparison_type": "threshold",
        "feature_name": "namefrst_jw",
        "threshold": 0.79,
    }
    leaves = get_comparison_leaves(comparison)
    assert leaves == [comparison]


@pytest.mark.parametrize("operator", ["AND", "OR"])
def test_get_comparison_leaves_one_level(operator: str) -> None:
    """
    When there are comp_a and comp_b subcomparisons, get_comparison_leaves()
    extracts them as the leaves.
    """
    comparison_a = {
        "comparison_type": "threshold",
        "feature_name": "namefrst_jw",
        "threshold": 0.79,
    }
    comparison_b = {
        "comparison_type": "threshold",
        "feature_name": "namelast_jw",
        "threshold": 0.84,
    }
    comparisons = {
        "operator": operator,
        "comp_a": comparison_a,
        "comp_b": comparison_b,
    }
    leaves = get_comparison_leaves(comparisons)
    assert leaves == [comparison_a, comparison_b]


@pytest.mark.parametrize("operator1", ["AND", "OR"])
@pytest.mark.parametrize("operator2", ["AND", "OR"])
def test_get_comparison_leaves_nested(operator1: str, operator2: str) -> None:
    """
    get_comparison_leaves() recurses through the tree to find leaves when there
    are multiple nested levels.
    """
    comparison_a = {
        "comparison_type": "threshold",
        "feature_name": "namefrst_jw",
        "threshold": 0.79,
    }
    comparison_b_a = {
        "comparison_type": "threshold",
        "feature_name": "namelast_jw",
        "threshold": 0.84,
    }
    comparison_b_b = {
        "comparison_type": "threshold",
        "feature_name": "marst_flag",
        "threshold_expr": ">0.5",
    }

    comparisons = {
        "operator": operator1,
        "comp_a": comparison_a,
        "comp_b": {
            "operator": operator2,
            "comp_a": comparison_b_a,
            "comp_b": comparison_b_b,
        },
    }

    leaves = get_comparison_leaves(comparisons)
    assert leaves == [comparison_a, comparison_b_a, comparison_b_b]


def test_generate_comparisons_empty_input() -> None:
    """
    generate_comparisons() returns an empty string for empty input.
    """
    comparisons = {}
    features = []
    id_col = ""
    assert generate_comparisons(comparisons, features, id_col) == ""


def test_generate_comparisons_base_case() -> None:
    """
    When there are no nested comp_a and comp_b comparisons, generate_comparisons()
    generates SQL just for the given comparison.
    """
    comparison = {
        "comparison_type": "threshold",
        "feature_name": "namefrst_jw",
        "threshold": 0.79,
    }
    features = [
        {
            "alias": "namefrst_jw",
            "column_name": "namefrst",
            "comparison_type": "jaro_winkler",
        }
    ]
    sql = generate_comparisons(comparison, features, "id")
    assert sql == "jw(nvl(a.namefrst, ''), nvl(b.namefrst, '')) >= 0.79"
