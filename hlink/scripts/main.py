# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import argparse
import getpass
import logging
import os
from pathlib import Path
import json
import importlib.metadata
import readline
import sys
from timeit import default_timer as timer
import traceback
from typing import Any
import uuid

from pyspark.sql import SparkSession

from hlink.spark.session import SparkConnection
from hlink.configs.load_config import load_conf_file
from hlink.errors import SparkError, UsageError
from hlink.scripts.lib.util import report_and_log_error
from hlink.linking.link_run import LinkRun
from hlink.scripts.main_loop import Main
from hlink.scripts.lib.conf_validations import analyze_conf
from hlink.scripts.lib.table_ops import drop_all_tables

HLINK_DIR = Path("./hlink_config")
logger = logging.getLogger(__name__)


def cli():
    """Called by the hlink script."""
    if "--version" in sys.argv:
        version = importlib.metadata.version("hlink")
        print(f"Hlink version: {version}")
        return
    args = _parse_args()

    try:
        if args.conf:
            conf_path, run_conf = load_conf_file(args.conf)
            print(f"*** Using config file {conf_path}")
        else:
            raise Exception(
                "ERROR: You must specify a config file to use by including either the --run or --conf flag in your program call."
            )
    except UsageError:
        print("Exception setting up config")
        i = sys.exc_info()
        print(i[1])
        sys.exit(1)
    except Exception as err:
        i = sys.exc_info()
        print(i[0])
        print(i[1])
        # traceback.print_tb(i[2])
        traceback.print_exception("", err, None)
        sys.exit(1)

    run_name = conf_path.stem
    _setup_logging(conf_path, run_name)

    logger.info("Initializing Spark")
    spark_init_start = timer()
    spark = _get_spark(run_name, args)
    spark_init_end = timer()
    spark_init_time = round(spark_init_end - spark_init_start, 2)
    logger.info(f"Initialized Spark in {spark_init_time}s")

    history_file = os.path.expanduser("~/.history_hlink")
    _read_history_file(history_file)

    try:
        if args.execute_tasks:
            main = Main(
                LinkRun(spark, run_conf, use_preexisting_tables=False),
                run_name=run_name,
            )
            main.preloop()

            task_list = " ".join(args.execute_tasks)

            main.do_run_all_steps(task_list)
        elif args.execute_command:
            main = Main(
                LinkRun(spark, run_conf, use_preexisting_tables=False),
                start_task=args.task,
                run_name=run_name,
            )
            main.preloop()
            command = " ".join(args.execute_command)
            print(f"Running Command: {command}")
            main.onecmd(command)
        else:
            _cli_loop(spark, args, run_conf, run_name)
        readline.write_history_file(history_file)
        spark.stop()
    except RuntimeError as err:
        report_and_log_error("Runtime Error", err)
        sys.exit(1)

    except SparkError as err:
        report_and_log_error("Spark Error", err)
        sys.exit(1)

    except Exception as err:
        report_and_log_error("Unclassified Error", err)
        sys.exit(1)


def _parse_args():
    parser = argparse.ArgumentParser(description="Hierarchical linking program.")
    parser.add_argument(
        "--user", help="run as a specific user", default=getpass.getuser()
    )
    parser.add_argument(
        "--cores", help="the max number of cores to use", default=4, type=int
    )
    parser.add_argument(
        "--executor_memory", help="the memory per executor to use", default="10G"
    )
    parser.add_argument(
        "--task", help="The initial task to begin processing.", default="preprocessing"
    )
    parser.add_argument(
        "--execute_tasks",
        help="Execute a series of tasks then exit the program.",
        nargs="+",
    )
    parser.add_argument(
        "--execute_command",
        help="Execute a single command then exit the program.",
        nargs="+",
    )
    parser.add_argument(
        "--conf",
        "--run",
        help="Specify a filepath where your config file for the run is located.",
    )
    parser.add_argument(
        "--clean",
        help="Drop any preexisting Spark tables when hlink starts up.",
        action="store_true",
    )

    return parser.parse_args()


def _get_spark(run_name: str, args: argparse.Namespace) -> SparkSession:
    derby_dir = HLINK_DIR / "derby" / run_name
    warehouse_dir = HLINK_DIR / "warehouse" / run_name
    checkpoint_dir = HLINK_DIR / "checkpoint" / run_name
    tmp_dir = HLINK_DIR / "tmp" / run_name
    python = sys.executable

    spark_connection = SparkConnection(
        derby_dir=derby_dir,
        warehouse_dir=warehouse_dir,
        checkpoint_dir=checkpoint_dir,
        tmp_dir=tmp_dir,
        python=python,
        db_name="linking",
    )
    spark = spark_connection.local(
        cores=args.cores, executor_memory=args.executor_memory
    )
    return spark


def _read_history_file(history_file):
    if not os.path.exists(history_file):
        with open(history_file, "a"):
            os.utime(history_file, (1330712280, 1330712292))
    readline.read_history_file(history_file)


def _cli_loop(spark, args, run_conf, run_name):
    if args.clean:
        print("Dropping preexisting tables")
        drop_all_tables(LinkRun(spark, run_conf))

    try:
        print("Analyzing config file")
        analyze_conf(LinkRun(spark, run_conf))
        logger.info("Analyzed config file, no errors found")
    except ValueError as err:
        logger.error(
            "Analysis found an error in the config file. See below for details."
        )
        report_and_log_error("", err)

    while True:
        main = Main(LinkRun(spark, run_conf), start_task=args.task, run_name=run_name)
        try:
            main.cmdloop()
            if main.lastcmd == "reload":
                logger.info("Reloading config file")
                conf_path, run_conf = load_conf_file(args.conf)
                print(f"*** Using config file {conf_path}")
            else:
                break
        except Exception as err:
            report_and_log_error("", err)


def _setup_logging(conf_path, run_name):
    log_dir = HLINK_DIR / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    user = getpass.getuser()
    session_id = uuid.uuid4().hex
    hlink_version = importlib.metadata.version("hlink")

    log_file = log_dir / f"{run_name}-{session_id}.log"

    format_string = "%(levelname)s %(asctime)s -- %(message)s"
    print(f"*** Hlink log: {log_file.absolute()}")

    logging.basicConfig(filename=log_file, level=logging.INFO, format=format_string)

    logger.info(f"New session {session_id} by user {user}")
    logger.info(f"Configured with {conf_path}")
    logger.info(f"Using hlink version {hlink_version}")
    logger.info(
        "-------------------------------------------------------------------------------------"
    )
