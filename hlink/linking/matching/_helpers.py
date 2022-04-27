# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink


def get_blocking(conf):
    if "blocking" in conf:
        return conf["blocking"]
    else:
        print(
            "DEPRECATION WARNING: The config value 'blocking_steps' has been renamed to 'blocking' and is now just a single array of objects."
        )
        return conf["blocking_steps"][0]
