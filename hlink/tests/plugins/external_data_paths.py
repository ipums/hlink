# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import hlink.tests
import pytest
import os


@pytest.fixture(scope="module")
def handle_null_path(spark):
    """Create a fixture with the path to the region codes file"""

    path = "input_data/handle_null.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def region_code_path(spark):
    """Create a fixture with the path to the region codes file"""

    path = "input_data/regioncode.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def substitutions_womens_names_path(spark):
    """Create a fixture with the path to Jonas's file for name substitutions for sex = 2"""
    path = "input_data/female.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def birthyr_replace_path(spark):
    """Create a fixture with the path to Jonas's file for name substitutions for sex = 2"""
    path = "input_data/birthyr_replace.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def substitutions_mens_names_path(spark):
    """Create a fixture with the path to Jonas's file for name substitutions for sex = 1"""
    path = "input_data/male.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def substitutions_street_abbrevs_path(spark):
    """Create a fixture with the path to Jonas's file for name substitutions for sex = 1"""
    path = "input_data/street_abbrevs_most_common.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def test_street_names_data_path(spark):
    """Create a fixture with the path to Jonas's file for name substitutions for sex = 1"""
    path = "input_data/test_street_names_data.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def state_dist_path(spark):
    """Create a fixture with the path to the distances lookup file"""

    path = "input_data/statedist.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def county_dist_path(spark):
    """Create a fixture with the path to the distances lookup file"""

    path = "input_data/county_distances.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def ext_path_preprocessing_popularity(spark):
    """Create a fixture with the path to the test potential_matches csv file"""

    path = "input_data/popularity.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def potential_matches_path(spark):
    """Create a fixture with the path to the test potential_matches csv file"""

    path = "input_data/potential_matches.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def potential_matches_path_ids_only(spark):
    """Create a fixture with the path to the test potential_matches csv file"""

    path = "input_data/potential_matches_ids_only.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def potential_matches_agg_path(spark):
    """Create a fixture with the path to the test potential_matches csv file"""

    path = "input_data/potential_matches_agg.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def nativity_datasources(spark):
    """Create a fixture with the path to the test training data file"""

    path_a = "input_data/nativity_test_data_a.csv"
    path_b = "input_data/nativity_test_data_b.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path_a = os.path.join(package_path, path_a)
    full_path_b = os.path.join(package_path, path_b)

    return full_path_a, full_path_b


@pytest.fixture(scope="module")
def training_data_path(spark):
    """Create a fixture with the path to the test training data file"""

    path = "input_data/training_data.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def training_data_doubled_path(spark):
    """Create a fixture with the path to the test training data file"""

    path = "input_data/training_data_doubled.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def threshold_ratio_data_path(spark):
    """Create a fixture with the path to the test training data file"""

    path = "input_data/threshold_ratio_test.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def threshold_ratio_data_path_2(spark):
    """Create a fixture with the path to the test training data file"""

    path = "input_data/threshold_ratio_test_data_2.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def hh_matching_stubs(spark):
    """Create a fixture with the path to the test training data file"""

    path_a = "input_data/hh_year_a.csv"
    path_b = "input_data/hh_year_b.csv"
    path_matches = "input_data/scored_matches_household_test.csv"
    path_pred_matches = "input_data/predicted_matches_test.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path_a = os.path.join(package_path, path_a)
    full_path_b = os.path.join(package_path, path_b)
    full_path_matches = os.path.join(package_path, path_matches)
    full_path_pred_matches = os.path.join(package_path, path_pred_matches)

    return full_path_a, full_path_b, full_path_matches, full_path_pred_matches


@pytest.fixture(scope="module")
def hh_integration_test_data(spark):
    """Create a fixture with the path to the test training data file"""

    path_a = "input_data/hh_matching_a.csv"
    path_b = "input_data/hh_matching_b.csv"
    path_matches = "input_data/matched_men.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path_a = os.path.join(package_path, path_a)
    full_path_b = os.path.join(package_path, path_b)
    full_path_matches = os.path.join(package_path, path_matches)

    return full_path_a, full_path_b, full_path_matches


@pytest.fixture(scope="module")
def scored_matches_test_data(spark):
    """Create a fixture with the path to the test training data file"""

    path_matches = "input_data/scored_matches_test_data.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path_matches = os.path.join(package_path, path_matches)

    return full_path_matches


@pytest.fixture(scope="module")
def hh_agg_features_test_data(spark):
    """Create a fixture with the path to the test training data file"""

    path_a = "input_data/ha_source.csv"
    path_b = "input_data/hb_source.csv"
    path_pms = "input_data/hhpm_agg_test.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path_a = os.path.join(package_path, path_a)
    full_path_b = os.path.join(package_path, path_b)
    full_path_pms = os.path.join(package_path, path_pms)

    return full_path_a, full_path_b, full_path_pms


@pytest.fixture(scope="module")
def hh_training_data_path(spark):
    """Create a fixture with the path to the test HH training data file"""

    td_path = "input_data/new_hh_test_td.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path_td = os.path.join(package_path, td_path)

    return full_path_td


@pytest.fixture(scope="module")
def training_validation_path(spark):
    """Create a fixture with the path to the test training data file"""

    path = "validation_data/training_all.parquet"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_path = os.path.join(package_path, path)

    return full_path


@pytest.fixture(scope="module")
def reporting_test_data_r2_pct(spark):
    """Create a fixture with the path to the test training data file"""

    pdfa_path = "input_data/reporting_prepped_df_a.csv"
    pm_path = "input_data/reporting_predicted_matches.csv"
    hhpm_path = "input_data/reporting_hh_predicted_matches.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_pdfa_path = os.path.join(package_path, pdfa_path)
    full_pm_path = os.path.join(package_path, pm_path)
    full_hhpm_path = os.path.join(package_path, hhpm_path)

    return full_pdfa_path, full_pm_path, full_hhpm_path


@pytest.fixture(scope="module")
def reporting_test_data_representivity(spark):
    """Create a fixture with the path to the test training data file"""

    rdf_path = "input_data/raw_df_reporting.csv"
    pdf_path = "input_data/prepped_df_reporting.csv"
    pm_path = "input_data/predicted_matches_reporting.csv"
    hhpm_path = "input_data/hh_predicted_matches_reporting.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_rdf_path = os.path.join(package_path, rdf_path)
    full_pdf_path = os.path.join(package_path, pdf_path)
    full_pm_path = os.path.join(package_path, pm_path)
    full_hhpm_path = os.path.join(package_path, hhpm_path)

    return full_rdf_path, full_pdf_path, full_pm_path, full_hhpm_path


@pytest.fixture(scope="module")
def test_data_rel_rows_age(spark):
    """Create a fixture with the path to the test training data file"""

    raw_a = "input_data/rel_rows_test_a.csv"
    raw_b = "input_data/rel_rows_test_b.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_a_path = os.path.join(package_path, raw_a)
    full_b_path = os.path.join(package_path, raw_b)

    return full_a_path, full_b_path


@pytest.fixture(scope="module")
def test_data_blocking_double_comparison(spark):
    """Create a fixture with the path to the test training data file"""

    raw_a = "input_data/jw_blocking_test_a.csv"
    raw_b = "input_data/jw_blocking_test_b.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_a_path = os.path.join(package_path, raw_a)
    full_b_path = os.path.join(package_path, raw_b)

    return full_a_path, full_b_path


@pytest.fixture(scope="module")
def crosswalk_input_paths(spark):
    """Create a fixture with the path to the test training data file"""

    raw_df_a = "input_data/crosswalk/raw_df_a.csv"
    raw_df_b = "input_data/crosswalk/raw_df_b.csv"
    predicted_matches = "input_data/crosswalk/predicted_matches.csv"
    hh_predicted_matches = "input_data/crosswalk/hh_predicted_matches.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    full_raw_df_a_path = os.path.join(package_path, raw_df_a)
    full_raw_df_b_path = os.path.join(package_path, raw_df_b)
    full_predicted_matches_path = os.path.join(package_path, predicted_matches)
    full_hh_predicted_matches_path = os.path.join(package_path, hh_predicted_matches)

    return (
        full_raw_df_a_path,
        full_raw_df_b_path,
        full_predicted_matches_path,
        full_hh_predicted_matches_path,
    )


@pytest.fixture(scope="module")
def crosswalk_validation_path(spark):
    package_path = os.path.dirname(hlink.tests.__file__)
    return os.path.join(package_path, "validation_data/crosswalks/crosswalk.csv")


@pytest.fixture(scope="module")
def crosswalk_with_round_validation_path(spark):
    package_path = os.path.dirname(hlink.tests.__file__)
    return os.path.join(
        package_path, "validation_data/crosswalks/crosswalk_with_round.csv"
    )
