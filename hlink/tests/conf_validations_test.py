import os
import pytest

from hlink.configs.load_config import load_conf_file
from hlink.scripts.lib.conf_validations import analyze_conf
from hlink.linking.link_run import LinkRun


@pytest.mark.parametrize(
    "conf_name,error_msg",
    [
        ("missing_datasource_a", r"Section \[datasource_a\] does not exist in config"),
        ("missing_datasource_b", r"Section \[datasource_b\] does not exist in config"),
        ("no_id_column_a", "Datasource A is missing the id column 'ID'"),
        ("no_id_column_b", "Datasource B is missing the id column 'ID'"),
    ],
)
def test_invalid_conf(conf_dir_path, spark, conf_name, error_msg):
    conf_file = os.path.join(conf_dir_path, conf_name)
    config = load_conf_file(conf_file)
    link_run = LinkRun(spark, config)

    with pytest.raises(ValueError, match=error_msg):
        analyze_conf(link_run)
