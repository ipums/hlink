# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import logging


def test_step_log(spark, preprocessing_conf, main, caplog):
    caplog.set_level(logging.INFO)
    main.do_set_link_task("preprocessing")
    main.do_run_step("0")
    print(caplog.records[0])
    assert "Finished Preprocessing - step 0: register raw dataframes in" in caplog.text
