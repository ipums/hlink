# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from hlink.configs.load_config import load_conf_file
import os.path
import pytest


@pytest.mark.quickcheck
def test_load_conf_file_json(package_path):
    conf_path = os.path.join(package_path, "conf")
    conf_file = os.path.join(conf_path, "test")
    conf = load_conf_file(conf_file)
    assert conf["id_column"] == "id"


@pytest.mark.quickcheck
def test_load_conf_file_toml(package_path):
    conf_path = os.path.join(package_path, "conf")
    conf_file = os.path.join(conf_path, "test1")
    conf = load_conf_file(conf_file)
    assert conf["id_column"] == "id-toml"


@pytest.mark.quickcheck
def test_load_conf_file_nested(package_path):
    running_path = package_path.rpartition("hlink/tests")[0]
    conf_name = "hlink_config/config/test_conf_flag_run"
    conf_file = os.path.join(running_path, conf_name)
    conf = load_conf_file(conf_file)
    assert conf["id_column"] == "id_conf_flag"
