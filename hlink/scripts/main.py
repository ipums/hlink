# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import argparse
import getpass
from importlib import reload
import logging
import os
from pathlib import Path
import json
import pkg_resources
import readline
import sys
import traceback
import uuid
from timeit import default_timer as timer

from hlink.spark.session import SparkConnection
from hlink.configs.load_config import load_conf_file
from hlink.errors import SparkError, UsageError
from hlink.scripts.lib.util import report_and_log_error
from hlink.linking.link_run import LinkRun
from hlink.scripts.main_loop import Main
from hlink.scripts.lib.conf_validations import analyze_conf
from hlink.scripts.lib.table_ops import drop_all_tables


def load_conf(conf_name, user):
    """Load and return the hlink config dictionary.

    Add the following attributes to the config dictionary:
    "derby_dir", "warehouse_dir", "spark_tmp_dir", "log_file", "python", "conf_path"
    """
    if "HLINK_CONF" not in os.environ:
        global_conf = None
    else:
        global_conf_file = os.environ["HLINK_CONF"]
        with open(global_conf_file) as f:
            global_conf = json.load(f)

    run_name = Path(conf_name).stem

    if global_conf is None:
        current_dir = Path.cwd()
        hlink_dir = current_dir / "hlink_config"
        base_derby_dir = hlink_dir / "derby"
        base_warehouse_dir = hlink_dir / "warehouse"
        base_spark_tmp_dir = hlink_dir / "spark_tmp_dir"
        conf = load_conf_file(conf_name)

        conf["derby_dir"] = base_derby_dir / run_name
        conf["warehouse_dir"] = base_warehouse_dir / run_name
        conf["spark_tmp_dir"] = base_spark_tmp_dir / run_name
        conf["log_file"] = hlink_dir / "run.log"
        conf["python"] = sys.executable
    else:
        user_dir = Path(global_conf["users_dir"]) / user
        user_dir_fast = Path(global_conf["users_dir_fast"]) / user
        conf_dir = user_dir / "confs"
        conf_path = conf_dir / conf_name
        conf = load_conf_file(str(conf_path))

        conf["derby_dir"] = user_dir / "derby" / run_name
        conf["warehouse_dir"] = user_dir_fast / "warehouse" / run_name
        conf["spark_tmp_dir"] = user_dir_fast / "tmp" / run_name
        conf["log_file"] = user_dir / "hlink.log"
        conf["python"] = global_conf["python"]

    print(f"*** Using config file {conf['conf_path']}")
    return conf


def cli():
    """Called by the hlink script. Referenced in setup.py."""
    if "--version" in sys.argv:
        version = pkg_resources.get_distribution("hlink").version
        print(f"Hlink version: {version}")
        return
    args = _parse_args()

    try:
        if args.conf:
            run_conf = load_conf(args.conf, args.user)
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

    _setup_logging(run_conf)

    logging.info("Initializing Spark")
    spark_init_start = timer()
    spark = _get_spark(run_conf, args)
    spark_init_end = timer()
    spark_init_time = round(spark_init_end - spark_init_start, 2)
    logging.info(f"Initialized Spark in {spark_init_time}s")

    history_file = os.path.expanduser("~/.history_hlink")
    _read_history_file(history_file)

    try:
        if args.execute_tasks:
            main = Main(LinkRun(spark, run_conf, use_preexisting_tables=False))
            main.preloop()

            task_list = " ".join(args.execute_tasks)

            main.do_run_all_steps(task_list)
        elif args.execute_command:
            main = Main(
                LinkRun(spark, run_conf, use_preexisting_tables=False),
                start_task=args.task,
            )
            main.preloop()
            command = " ".join(args.execute_command)
            print(f"Running Command: {command}")
            main.onecmd(command)
        else:
            _cli_loop(spark, args, run_conf)
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


def _get_spark(run_conf, args):
    spark_connection = SparkConnection(
        run_conf["derby_dir"],
        run_conf["warehouse_dir"],
        run_conf["spark_tmp_dir"],
        run_conf["python"],
        "linking",
    )
    spark = spark_connection.local(
        cores=args.cores, executor_memory=args.executor_memory
    )
    return spark


def _read_history_file(history_file):
    if not (os.path.exists(history_file)):
        with open(history_file, "a"):
            os.utime(history_file, (1330712280, 1330712292))
    readline.read_history_file(history_file)


def _cli_loop(spark, args, run_conf):
    if args.clean:
        print("Dropping preexisting tables")
        drop_all_tables(LinkRun(spark, run_conf))

    try:
        print("Analyzing config file")
        analyze_conf(LinkRun(spark, run_conf))
        logging.info("Analyzed config file, no errors found")
    except ValueError as err:
        logging.error(
            "Analysis found an error in the config file. See below for details."
        )
        report_and_log_error("", err)

    while True:
        main = Main(LinkRun(spark, run_conf), start_task=args.task)
        try:
            main.cmdloop()
            if main.lastcmd == "reload":
                _reload_modules()
                # Reload modules twice in order to fix import problem
                # with the _*.py files in the linking modules
                _reload_modules()
                run_conf = load_conf(args.conf, args.user)
            else:
                break
        except Exception as err:
            report_and_log_error("", err)


def _reload_modules():
    no_reloads = []
    mods_to_reload_raw = [name for name, mod in sys.modules.items()]
    # We need to order the modules to reload the _*.py files in the
    # linking modules before loading the __init__.py files.
    mods_to_reload_ordered = sorted(mods_to_reload_raw)[::-1]
    for name in mods_to_reload_ordered:
        if name.startswith("hlink") and name not in no_reloads:
            reload(sys.modules[name])

    # Here we should reset the classes in link_run.link_task_choices with
    # the newly reloaded classes.


def _setup_logging(conf):
    log_file = Path(conf["log_file"])
    log_file.parent.mkdir(exist_ok=True, parents=True)

    user = getpass.getuser()
    session_id = uuid.uuid4().hex
    hlink_version = pkg_resources.get_distribution("hlink").version

    # format_string = f"%(levelname)s %(asctime)s {user} {session_id} %(message)s -- {conf['conf_path']}"
    format_string = "%(levelname)s %(asctime)s -- %(message)s"
    print(f"*** Hlink log: {log_file}")

    logging.basicConfig(filename=log_file, level=logging.INFO, format=format_string)

    logging.info("")
    logging.info(
        "-------------------------------------------------------------------------------------"
    )
    logging.info(f"   New session {session_id} by user {user}")
    logging.info(f"   Configured with {conf['conf_path']}")
    logging.info(f"   Using hlink version {hlink_version}")
    logging.info(
        "-------------------------------------------------------------------------------------"
    )
    logging.info("")
