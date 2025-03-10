# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pathlib import Path

import pytest

from hlink.configs.load_config import load_conf_file
from hlink.errors import UsageError


@pytest.mark.parametrize("file_name", ["test", "test.json"])
def test_load_conf_file_json(conf_dir_path: str, file_name: str) -> None:
    conf_file = Path(conf_dir_path) / file_name
    path, conf = load_conf_file(str(conf_file))
    assert conf["id_column"] == "id"
    assert path == conf_file.with_suffix(".json")


@pytest.mark.parametrize("file_name", ["test1", "test1.toml"])
def test_load_conf_file_toml(conf_dir_path: str, file_name: str) -> None:
    conf_file = Path(conf_dir_path) / file_name
    path, conf = load_conf_file(str(conf_file))
    assert conf["id_column"] == "id-toml"
    assert path == conf_file.with_suffix(".toml")


def test_load_conf_file_json2(conf_dir_path: str) -> None:
    conf_file = Path(conf_dir_path) / "test_conf_flag_run"
    path, conf = load_conf_file(str(conf_file))
    assert conf["id_column"] == "id_conf_flag"
    assert path == conf_file.with_suffix(".json")


def test_load_conf_file_does_not_exist(tmp_path: Path) -> None:
    conf_file = tmp_path / "notthere"
    with pytest.raises(
        FileNotFoundError, match="Couldn't find any of these three files:"
    ):
        load_conf_file(str(conf_file))


def test_load_conf_file_unrecognized_extension(tmp_path: Path) -> None:
    conf_file = tmp_path / "test.yaml"
    conf_file.touch()
    with pytest.raises(
        UsageError,
        match="The file .+ exists, but it doesn't have a '.toml' or '.json' extension",
    ):
        load_conf_file(str(conf_file))


def test_load_conf_file_json_legacy_parser(conf_dir_path: str) -> None:
    """
    The use_legacy_toml_parser argument does not affect json parsing.
    """
    conf_file = Path(conf_dir_path) / "test.json"
    _, conf = load_conf_file(str(conf_file), use_legacy_toml_parser=True)
    assert conf["id_column"] == "id"


def test_load_conf_file_toml_legacy_parser(conf_dir_path: str) -> None:
    conf_file = Path(conf_dir_path) / "test1.toml"
    _, conf = load_conf_file(str(conf_file), use_legacy_toml_parser=True)
    assert conf["id_column"] == "id-toml"
