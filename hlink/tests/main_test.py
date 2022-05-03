# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import pytest
import json
import toml
from pathlib import Path

from hlink.scripts.main import load_conf
from hlink.errors import UsageError

users = ("jesse", "woody")


@pytest.fixture()
def global_conf(tmp_path):
    """The contents of the test global config as a dictionary."""
    global_conf = {}
    global_conf["users_dir"] = str(tmp_path / "users_dir")
    global_conf["users_dir_fast"] = str(tmp_path / "users_dir_fast")
    global_conf["python"] = "python"

    return global_conf


@pytest.fixture()
def set_up_global_conf_file(monkeypatch, tmp_path, global_conf):
    """Create the global config file and set the HLINK_CONF environment variable.

    The contents of the global config file are the same as the `global_conf` fixture
    dictionary.
    """
    file = tmp_path / "global_config_file.json"

    with open(file, "w") as f:
        json.dump(global_conf, f)

    monkeypatch.setenv("HLINK_CONF", str(file))


def get_conf_dir(global_conf, user):
    """Given the global config and user, return the path to the user's config directory."""
    return Path(global_conf["users_dir"]) / user / "confs"


@pytest.mark.parametrize("conf_file", ("my_conf", "my_conf.toml", "my_conf.json"))
@pytest.mark.parametrize("user", users)
def test_load_conf_does_not_exist_no_env(monkeypatch, tmp_path, conf_file, user):
    monkeypatch.delenv("HLINK_CONF", raising=False)

    filename = str(tmp_path / conf_file)
    toml_filename = filename + ".toml"
    json_filename = filename + ".json"

    error_msg = f"Couldn't find any of these three files: {filename}, {toml_filename}, {json_filename}"
    with pytest.raises(FileNotFoundError, match=error_msg):
        load_conf(filename, user)


@pytest.mark.quickcheck
@pytest.mark.parametrize("conf_file", ("my_conf.json",))
@pytest.mark.parametrize("user", users)
def test_load_conf_json_exists_no_env(monkeypatch, tmp_path, conf_file, user):
    monkeypatch.delenv("HLINK_CONF", raising=False)
    monkeypatch.chdir(tmp_path)
    filename = str(tmp_path / conf_file)

    contents = {}
    with open(filename, "w") as f:
        json.dump(contents, f)

    conf = load_conf(filename, user)
    assert conf["conf_path"] == filename


@pytest.mark.parametrize("conf_name", ("my_conf", "my_conf.json", "my_conf.toml"))
@pytest.mark.parametrize("user", users)
def test_load_conf_json_exists_ext_added_no_env(monkeypatch, tmp_path, conf_name, user):
    monkeypatch.delenv("HLINK_CONF", raising=False)
    monkeypatch.chdir(tmp_path)
    filename = str(tmp_path / conf_name) + ".json"

    contents = {}
    with open(filename, "w") as f:
        json.dump(contents, f)

    conf = load_conf(str(tmp_path / conf_name), user)
    assert conf["conf_path"] == filename


@pytest.mark.quickcheck
@pytest.mark.parametrize("conf_file", ("my_conf.toml",))
@pytest.mark.parametrize("user", users)
def test_load_conf_toml_exists_no_env(monkeypatch, tmp_path, conf_file, user):
    monkeypatch.delenv("HLINK_CONF", raising=False)
    monkeypatch.chdir(tmp_path)
    filename = str(tmp_path / conf_file)

    contents = {}
    with open(filename, "w") as f:
        toml.dump(contents, f)

    conf = load_conf(filename, user)
    assert conf["conf_path"] == filename


@pytest.mark.parametrize("conf_name", ("my_conf", "my_conf.json", "my_conf.toml"))
@pytest.mark.parametrize("user", users)
def test_load_conf_toml_exists_ext_added_no_env(monkeypatch, tmp_path, conf_name, user):
    monkeypatch.delenv("HLINK_CONF", raising=False)
    monkeypatch.chdir(tmp_path)
    filename = str(tmp_path / conf_name) + ".toml"

    contents = {}
    with open(filename, "w") as f:
        toml.dump(contents, f)

    conf = load_conf(str(tmp_path / conf_name), user)
    assert conf["conf_path"] == filename


@pytest.mark.parametrize("conf_name", ("my_conf", "testing.txt", "what.yaml"))
@pytest.mark.parametrize("user", users)
def test_load_conf_unrecognized_ext_env(
    monkeypatch, tmp_path, set_up_global_conf_file, global_conf, conf_name, user
):
    monkeypatch.chdir(tmp_path)

    conf_dir = get_conf_dir(global_conf, user)
    conf_dir.mkdir(parents=True)
    file = conf_dir / conf_name
    file.touch()

    error_msg = (
        f"The file {file} exists, but it doesn't have a '.toml' or '.json' extension."
    )
    with pytest.raises(UsageError, match=error_msg):
        load_conf(str(file), user)


def test_load_conf_keys_set_no_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HLINK_CONF", raising=False)
    monkeypatch.chdir(tmp_path)
    filename = str(tmp_path / "keys_test.json")
    contents = {"key1": "value1", "rock": "stone", "how": "about that"}

    with open(filename, "w") as f:
        json.dump(contents, f)

    conf = load_conf(filename, "test")

    for (key, value) in contents.items():
        assert conf[key] == value

    # Check for extra keys added by load_conf()
    assert "conf_path" in conf
    assert "derby_dir" in conf
    assert "warehouse_dir" in conf
    assert "spark_tmp_dir" in conf
    assert "log_file" in conf
    assert "python" in conf


@pytest.mark.parametrize("global_conf", ("my_global_conf.json", "test.json"))
def test_load_conf_global_conf_does_not_exist_env(monkeypatch, tmp_path, global_conf):
    global_path = str(tmp_path / global_conf)
    monkeypatch.setenv("HLINK_CONF", global_path)

    with pytest.raises(FileNotFoundError):
        load_conf("notthere.toml", "test")


@pytest.mark.parametrize("conf_file", ("my_conf", "my_conf.json", "my_conf.toml"))
@pytest.mark.parametrize("user", users)
def test_load_conf_does_not_exist_env(
    monkeypatch, tmp_path, set_up_global_conf_file, global_conf, conf_file, user
):
    monkeypatch.chdir(tmp_path)

    conf_dir = get_conf_dir(global_conf, user)
    filename = str(conf_dir / conf_file)
    toml_filename = filename + ".toml"
    json_filename = filename + ".json"

    error_msg = f"Couldn't find any of these three files: {filename}, {toml_filename}, {json_filename}"
    with pytest.raises(FileNotFoundError, match=error_msg):
        load_conf(conf_file, user)


@pytest.mark.quickcheck
@pytest.mark.parametrize("conf_file", ("my_conf.json",))
@pytest.mark.parametrize("user", users)
def test_load_conf_json_exists_in_conf_dir_env(
    monkeypatch, tmp_path, set_up_global_conf_file, global_conf, conf_file, user
):
    monkeypatch.chdir(tmp_path)
    conf_dir = get_conf_dir(global_conf, user)
    conf_dir.mkdir(parents=True)

    file = conf_dir / conf_file
    contents = {}

    with open(file, "w") as f:
        json.dump(contents, f)

    conf = load_conf(conf_file, user)
    assert conf["conf_path"] == str(file)


@pytest.mark.quickcheck
@pytest.mark.parametrize("conf_file", ("my_conf.toml",))
@pytest.mark.parametrize("user", users)
def test_load_conf_toml_exists_in_conf_dir_env(
    monkeypatch, tmp_path, set_up_global_conf_file, global_conf, conf_file, user
):
    monkeypatch.chdir(tmp_path)
    conf_dir = get_conf_dir(global_conf, user)
    conf_dir.mkdir(parents=True)

    file = conf_dir / conf_file
    contents = {}

    with open(file, "w") as f:
        toml.dump(contents, f)

    conf = load_conf(conf_file, user)
    assert conf["conf_path"] == str(file)


@pytest.mark.parametrize("conf_name", ("my_conf", "test", "testingtesting123.txt"))
@pytest.mark.parametrize("user", users)
def test_load_conf_json_exists_in_conf_dir_ext_added_env(
    monkeypatch, tmp_path, set_up_global_conf_file, global_conf, conf_name, user
):
    monkeypatch.chdir(tmp_path)
    conf_dir = get_conf_dir(global_conf, user)
    conf_dir.mkdir(parents=True)

    conf_file = conf_name + ".json"
    file = conf_dir / conf_file
    contents = {}

    with open(file, "w") as f:
        json.dump(contents, f)

    conf = load_conf(conf_name, user)
    assert conf["conf_path"] == str(file)


@pytest.mark.parametrize("conf_name", ("my_conf", "test", "testingtesting123.txt"))
@pytest.mark.parametrize("user", users)
def test_load_conf_toml_exists_in_conf_dir_ext_added_env(
    monkeypatch, tmp_path, set_up_global_conf_file, global_conf, conf_name, user
):
    monkeypatch.chdir(tmp_path)
    conf_dir = get_conf_dir(global_conf, user)
    conf_dir.mkdir(parents=True)

    conf_file = conf_name + ".toml"
    file = conf_dir / conf_file
    contents = {}

    with open(file, "w") as f:
        toml.dump(contents, f)

    conf = load_conf(conf_name, user)
    assert conf["conf_path"] == str(file)


@pytest.mark.parametrize("conf_name", ("my_conf", "testing.txt", "what.yaml"))
@pytest.mark.parametrize("user", users)
def test_load_conf_unrecognized_ext_no_env(monkeypatch, tmp_path, conf_name, user):
    monkeypatch.delenv("HLINK_CONF", raising=False)
    monkeypatch.chdir(tmp_path)

    file = tmp_path / conf_name
    file.touch()

    error_msg = f"The file {conf_name} exists, but it doesn't have a '.toml' or '.json' extension."
    with pytest.raises(UsageError, match=error_msg):
        load_conf(conf_name, user)


def test_load_conf_keys_set_env(
    monkeypatch, tmp_path, set_up_global_conf_file, global_conf
):
    monkeypatch.chdir(tmp_path)
    user = "test"
    conf_dir = get_conf_dir(global_conf, user)
    conf_dir.mkdir(parents=True)
    file = conf_dir / "keys_test.json"
    filename = str(file)

    contents = {"key1": "value1", "rock": "stone", "how": "about that"}

    with open(file, "w") as f:
        json.dump(contents, f)

    conf = load_conf(filename, user)

    for (key, value) in contents.items():
        assert conf[key] == value

    # Check for extra keys added by load_conf()
    assert "conf_path" in conf
    assert "derby_dir" in conf
    assert "warehouse_dir" in conf
    assert "spark_tmp_dir" in conf
    assert "log_file" in conf
    assert "python" in conf
