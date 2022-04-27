# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import sys
import logging
import traceback


def report_and_log_error(message: str, err: Exception):
    print(f"An error occured: {message}")
    i = sys.exc_info()
    print(f"ERROR type: {type(err)}")
    print(f"ERROR message: {i[1]}")
    print("See log for details.")
    print("")
    # Perhaps for a verbose mode:
    # traceback.print_exception("",err,i[2])
    multi_line = "\n==========\n"

    logging.error(
        str(i[0])
        + " : "
        + str(i[1])
        + multi_line
        + str.join("", traceback.format_exception(type(err), err, i[2]))
        + multi_line
    )
