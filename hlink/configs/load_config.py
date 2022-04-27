# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pathlib import Path
import json
import toml

from hlink.errors import UsageError


def load_conf_file(conf_name):
    """Flexibly load a config file.

    Given a path `conf_name`, look for a file at that path. If that file
    exists and has a '.toml' extension or a '.json' extension, load it and
    return its contents. If it doesn't exist, look for a file with the same
    name with a '.toml' extension added and load it if it exists. Then do the
    same for a file with a '.json' extension added.

    After successfully loading a config file, store the absolute path where the
    config file was found as the value of the "conf_path" key in the returned
    config dictionary.

    Args:
        conf_name (str): the file to look for

    Returns:
        dict: the contents of the config file

    Raises:
        FileNotFoundError: if none of the three checked files exist
        UsageError: if the file at path `conf_name` exists, but it doesn't have a '.toml' or '.json' extension
    """
    candidate_files = [
        Path(conf_name),
        Path(conf_name + ".toml"),
        Path(conf_name + ".json"),
    ]

    existing_files = filter((lambda file: file.exists()), candidate_files)

    for file in existing_files:
        if file.suffix == ".toml":
            with open(file) as f:
                conf = toml.load(f)
                conf["conf_path"] = str(file.resolve())
                return conf

        if file.suffix == ".json":
            with open(file) as f:
                conf = json.load(f)
                conf["conf_path"] = str(file.resolve())
                return conf

        raise UsageError(
            f"The file {file} exists, but it doesn't have a '.toml' or '.json' extension."
        )

    candidate_files_str = ", ".join(map(str, candidate_files))
    raise FileNotFoundError(
        f"Couldn't find any of these three files: {candidate_files_str}"
    )
