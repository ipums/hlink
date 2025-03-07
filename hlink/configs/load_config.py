# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pathlib import Path
from typing import Any
import json
import toml
import tomli

from hlink.errors import UsageError


def load_conf_file(
    conf_name: str, *, use_legacy_toml_parser: bool = False
) -> tuple[Path, dict[str, Any]]:
    """Flexibly load a config file.

    Given a path `conf_name`, look for a file at that path. If that file
    exists and has a '.toml' extension or a '.json' extension, load it and
    return its contents. If it doesn't exist, look for a file with the same
    name with a '.toml' extension added and load it if it exists. Then do the
    same for a file with a '.json' extension added.

    `use_legacy_toml_parser` tells this function to use the legacy TOML library
    which hlink used to use instead of the current default. This is provided
    for backwards compatibility. Some previously written config files may
    depend on bugs in the legacy TOML library, making it hard to migrate to the
    new TOML v1.0 compliant parser. It is strongly recommended that new code
    and config files use the default parser. Old code and config files should
    also try to migrate to the default parser when possible.

    Args:
        conf_name: the file to look for
        use_legacy_toml_parser: (Not Recommended) Use the legacy, buggy TOML
        parser instead of the default parser.

    Returns:
        a tuple (absolute path to the config file, contents of the config file)

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
            # Legacy support for using the "toml" library instead of "tomli".
            #
            # Eventually we should remove use_legacy_toml_parser and just use
            # tomli or Python's standard library tomllib, which is available in
            # Python 3.11+.
            if use_legacy_toml_parser:
                with open(file) as f:
                    conf = toml.load(f)
                    return file.absolute(), conf
            else:
                with open(file, "rb") as f:
                    conf = tomli.load(f)
                    return file.absolute(), conf

        if file.suffix == ".json":
            with open(file) as f:
                conf = json.load(f)
                return file.absolute(), conf

        raise UsageError(
            f"The file {file} exists, but it doesn't have a '.toml' or '.json' extension."
        )

    candidate_files_str = ", ".join(map(str, candidate_files))
    raise FileNotFoundError(
        f"Couldn't find any of these three files: {candidate_files_str}"
    )
