# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from cmd import Cmd
import logging
from typing import Optional
import functools
from timeit import default_timer as timer

from hlink.errors import UsageError
from hlink.linking.link_run import link_task_choices

import hlink.scripts.lib.experimental.tfam as x_tfam
import hlink.scripts.lib.experimental.reporting as x_reporting
import hlink.scripts.lib.table_ops as table_ops
import hlink.scripts.lib.io as io
import hlink.scripts.lib.linking_ops as linking_ops
import hlink.scripts.lib.conf_validations as conf_validations


def split_and_check_args(expected_count):
    """A parametrized decorator to make handling arguments easier for `Main` methods.

    The decorator splits the string `args` and checks the count of split args against `expected_count`.
    If the count is allowed, it passes the split args along to the wrapped function. Otherwise,
    it immediately returns None. It uses `Main.check_arg_count()` to check the split args.
    The docstring of the decorated method is used as the `command_docs` argument to `Main.check_arg_count()`.

    Decorated methods should take `self` and `split_args` as their arguments.
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, args):
            split_args = args.split()
            if self.check_arg_count(split_args, expected_count, f.__doc__):
                return
            return f(self, split_args)

        return wrapper

    return decorator


class Main(Cmd):
    """Main program which handles user input. See https://docs.python.org/3/library/cmd.html for more information."""

    prompt = "hlink $ "
    intro = "Welcome to hlink. Type ? to list commands and q to quit.\n"

    def __init__(
        self,
        link_run,
        start_task: Optional[str] = None,
    ):
        self.link_run = link_run
        self.spark = self.link_run.spark

        if start_task is None:
            self.current_link_task = self.link_run.preprocessing
        else:
            self.current_link_task = self.link_run.get_task(start_task)

        super().__init__()

    def preloop(self):
        self.reload_auto_complete_cache()

    def reload_auto_complete_cache(self):
        self.table_names = [t.name for t in self.spark.catalog.listTables()]

    def precmd(self, line):
        return line

    # These are meant to be flags / switches, not long options with arguments  following them
    def extract_flags_from_args(self, applicable_flags, split_args):
        """Separates the flags from the regular arguments, checks that flags
        passed in match the applicable flags for the command.
        arg1: list of applicable flags
        arg2: Pre-split list of arguments including flags given by user."""

        flags = [a for a in split_args if "--" in a]
        non_flag_args = [a for a in split_args if "--" not in a]

        unsupported_flags = set(flags) - set(applicable_flags)
        if unsupported_flags:
            raise UsageError(
                f"The flags {unsupported_flags} aren't supported by this command. Supported flags are {applicable_flags}"
            )

        return flags, non_flag_args

    def check_arg_count(self, split_args, expected_count, command_docs):
        """Checks the number of arguments submitted against the expected number(s).

        Args:
            split_args (List[str]): the arguments submitted
            expected_count (int | List[int]): the expected number of arguments, or a list of allowed numbers of arguments
            command_docs (str): the help documentation for the calling command

        Returns:
            bool: True if the argument count is incorrect, False if it is correct
        """
        num_args = len(split_args)
        # Special case: there are actually 0 args if the only provided arg is the empty string
        if num_args == 1 and split_args[0] == "":
            num_args = 0

        expected_counts = (
            expected_count if type(expected_count) == list else [expected_count]
        )
        expected_counts_str = " or ".join(map(str, expected_counts))

        arg_form = "argument" if expected_counts == [1] else "arguments"

        if num_args not in expected_counts:
            print("Argument error!")
            print(
                f"This command takes {expected_counts_str} {arg_form} and you gave {num_args}."
            )
            print("See the command description below:")
            print(command_docs)
            return True

        return False

    def emptyline(self):
        return False

    @split_and_check_args(0)
    def do_q(self, split_args):
        """Quits the program.
        Usage: q"""
        return True

    @split_and_check_args(0)
    def do_reload(self, split_args):
        """Hot reload modules.
        Usage: reload"""
        return "reload"

    @split_and_check_args(1)
    def do_set_link_task(self, split_args):
        """Set the linking task to run steps for.
        Arg 1: task
        To retrieve a list of valid linking tasks, use the command 'get_tasks'."""
        link_task = split_args[0]
        if link_task in link_task_choices:
            self.current_link_task = self.link_run.get_task(link_task)
            print(f"Set to: {self.current_link_task}")
        else:
            choices = ", \n\t".join(link_task_choices.keys())
            print(f"Invalid choice. \nValid choices are: \n\t{choices}")

    def complete_set_link_task(self, text, line, begidx, endidx):
        return [t for t in link_task_choices if t.startswith(text)]

    @split_and_check_args(0)
    def do_set_preexisting_tables(self, split_args):
        """Toggle the preexisting tables flag. Steps will essentially be skipped when their output already exists.
        Default setting is True (uses pre-existing tables.)
        Usage: set_preexisting_tables"""
        self.link_run.use_preexisting_tables = not self.link_run.use_preexisting_tables
        print(f"Use preexisting tables: {self.link_run.use_preexisting_tables}")

    @split_and_check_args(0)
    def do_set_print_sql(self, split_args):
        """Toggle the print sql flag.
        Default setting is False.
        Usage: set_print_sql"""
        self.link_run.print_sql = not self.link_run.print_sql
        print(f"Print sql: {self.link_run.print_sql}")

    @split_and_check_args(0)
    def do_get_settings(self, split_args):
        """Show the current settings which can be toggled.
        Current settings displayed include:
        - use pre-existing tables
        - print SQL
        - current linking task.
        Usage: get_settings"""
        print(f"Use preexisting tables: {self.link_run.use_preexisting_tables}")
        print(f"Print sql: {self.link_run.print_sql}")
        print(f"Current link task: {self.current_link_task}")

    @split_and_check_args(0)
    def do_ipython(self, split_args):
        """Open an ipython shell.
        Usage: ipython"""
        import IPython

        IPython.embed()

    @split_and_check_args(0)
    def do_get_tasks(self, split_args):
        """Get all of the available linking tasks.
        Usage: get_tasks
        Hint: Specify the current linking task using the 'set_link_task' command."""
        linking_ops.show_tasks(self.current_link_task, self.link_run, link_task_choices)

    @split_and_check_args(0)
    def do_get_steps(self, split_args):
        """Get all of the steps for the current linking task.
        Usage: get_steps
        Hint: Specify the current linking task using the 'set_link_task' command."""
        linking_ops.show_step_info(self.current_link_task, self.link_run)

    @split_and_check_args(1)
    def do_run_step(self, split_args):
        """Run the specified household linking step.
        Arg 1: step number (an integer)
        Hint: Use the command 'get_steps' to fetch a list of all the household linking steps."""
        print(f"Link task: {self.current_link_task}")
        step_num = int(split_args[0])
        self.current_link_task.run_step(step_num)

    def do_run_all_steps(self, args):
        """Run all of the linking steps within the given tasks, in the order given. If no tasks are given, run all the steps for the current task.
        ArgN (Optional): Link tasks to run all steps for."""
        split_args = args.split()
        if len(split_args) > 0:
            for link_task in split_args:
                if link_task not in link_task_choices:
                    print(
                        f"Argument error! \nThis function takes a list of link tasks as arguments. \n Argument {link_task} is not a link task. See method description below:"
                    )
                    print(self.do_run_all_steps.__doc__)
                    return
            for link_task in split_args:
                task_inst = self.link_run.get_task(link_task)
                print(f"Running task: {task_inst}")
                task_inst.run_all_steps()
                print()
        else:
            print(f"Running task: {self.current_link_task}")
            self.current_link_task.run_all_steps()
            print()

    def complete_run_all_steps(self, text, line, begidx, endidx):
        return [t for t in link_task_choices if t.startswith(text)]

    @split_and_check_args(0)
    def do_analyze(self, split_args):
        """Print an analysis of dependencies in the config."""
        conf_validations.analyze_conf(self.link_run)

    @split_and_check_args(1)
    def do_count(self, split_args):
        """Prints the count of rows in a table.
        Arg 1: table"""
        table_name = split_args[0]
        table_ops.show_table_row_count(self.spark, table_name)

    def complete_count(self, text, line, begidx, endidx):
        return self.check_table_names(text)

    @split_and_check_args(1)
    def do_desc(self, split_args):
        """Prints the columns of a table.
        Arg 1: table"""
        table_name = split_args[0]
        table_ops.show_table_columns(self.spark, table_name)

    def complete_desc(self, text, line, begidx, endidx):
        return self.check_table_names(text)

    def do_list(self, args):
        """List tables that have been registered.
        Usage: list"""
        split_args = args.split()
        list_all = any(x in ["a", "all"] for x in split_args)
        table_ops.list_tables(self.link_run, list_all=list_all)

    @split_and_check_args(1)
    def do_show(self, split_args):
        """Prints the first 10 lines of a table.
        Arg 1: table"""
        table_name = split_args[0]
        table_ops.show_table(self.spark, table_name, limit=10)

    def complete_show(self, text, line, begidx, endidx):
        return self.check_table_names(text)

    @split_and_check_args(1)
    def do_showf(self, split_args):
        """Prints the first 10 lines of a table, without truncating data.
        Arg 1: table"""
        table_name = split_args[0]
        table_ops.show_table(self.spark, table_name, limit=10, truncate=False)

    def complete_showf(self, text, line, begidx, endidx):
        return self.check_table_names(text)

    @split_and_check_args(1)
    def do_drop(self, split_args):
        """Delete a table.
        Arg 1: table"""
        table_name = split_args[0]
        table_ops.drop_table(self.link_run, table_name)

    def complete_drop(self, text, line, begidx, endidx):
        return self.check_table_names(text)

    @split_and_check_args(0)
    def do_drop_all(self, split_args):
        """Delete all tables.
        Usage: drop_all"""
        table_ops.drop_all_tables(self.link_run)

    @split_and_check_args(0)
    def do_drop_all_temp(self, split_args):
        """Delete all temporary tables.
        Usage: drop_all_temp"""
        self.link_run.drop_temp_tables()

    @split_and_check_args(0)
    def do_drop_all_prc(self, split_args):
        """Delete all precision recall curve tables.
        Usage: drop_all_prc"""
        table_ops.drop_prc_tables(self.link_run)

    @split_and_check_args(2)
    def do_x_summary(self, split_args):
        """Prints a summary of a variable in a table.
        [!] This command is experimental.
        Arg 1: table
        Arg 2: variable"""
        table_name, col_name = split_args
        table_ops.show_column_summary(self.spark, table_name, col_name)

    def complete_x_summary(self, text, line, begidx, endidx):
        return self.check_table_names(text)

    @split_and_check_args(2)
    def do_x_tab(self, split_args):
        """Prints tabulation of a variable.
        [!] This command is experimental.
        Arg 1: table
        Arg 2: var_name"""
        table_name, col_name = split_args
        table_ops.show_column_tab(self.spark, table_name, col_name)

    def complete_x_tab(self, text, line, begidx, endidx):
        return self.check_table_names(text)

    @split_and_check_args(1)
    def do_x_persist(self, split_args):
        """Takes a temporary table and makes it permanent.
        [!] This command is experimental.
        Arg 1: table to persist"""
        table_ops.persist_table(self.spark, split_args[0])

    def complete_x_persist(self, text, line, begidx, endidx):
        return self.check_table_names(text)

    def do_x_sql(self, args):
        """Runs arbitrary sql. Drop to ipython for multiline queries.
        [!] This command is experimental.
        Args: SQL query"""
        split_args = args.split()
        if len(split_args) == 0:
            print(
                "Argument error! \nThis function takes a SQL query as an argument. \nSee method description below:"
            )
            print(self.do_x_sql.__doc__)
            return
        table_ops.run_and_show_sql(self.spark, args)

    def do_x_sqlf(self, args):
        """Runs arbitrary sql without truncating. Drop to ipython for multiline queries.
        [!] This command is experimental.
        Args: SQL query"""
        split_args = args.split()
        if len(split_args) == 0:
            print(
                "Argument error! \nThis function takes a SQL query as an argument. \nSee method description below:"
            )
            print(self.do_x_sqlf.__doc__)
            return
        table_ops.run_and_show_sql(self.spark, args, truncate=False)

    @split_and_check_args(4)
    def do_x_union(self, split_args):
        """Creates a new table from the union of two previous tables.
        [!] This command is experimental.
        Arg 1: first table
        Arg 2: second table
        Arg 3: output name
        Arg 4: mark column"""
        table1_name, table2_name, output_table_name, mark_col_name = split_args
        table_ops.take_table_union(
            self.spark, table1_name, table2_name, output_table_name, mark_col_name
        )

    def complete_x_union(self, text, line, begidx, endidx):
        return self.check_table_names(text)

    @split_and_check_args([2, 3])
    def do_csv(self, split_args):
        """Writes a table out to csv.
        Arg 1: table name
        Arg 2: path
        Arg 3 (optional): # of partitions"""
        table_name = split_args[0]
        output_path = split_args[1]
        num_args = len(split_args)
        num_partitions = int(split_args[2]) if num_args == 3 else None

        io.write_table_to_csv(
            self.link_run.spark, table_name, output_path, num_partitions
        )

    def complete_csv(self, text, line, begidx, endidx):
        return self.check_table_names(text)

    @split_and_check_args(1)
    def do_borrow_tables(self, split_args):
        """Register tables from another hlink installation. Takes an absolute path to a Spark warehouse directory and a job name e.g.
        borrow_tables /mnt/nas/spark/linking/ccd/warehouse/full_count_1900_1910"""
        borrow_tables_from = split_args[0] + "/linking.db"
        print(f"Trying to borrow tables in {borrow_tables_from}")
        print("")

        io.borrow_spark_tables(self.spark, borrow_tables_from)

        print("")
        print("Type 'list' to show all the available tables.")
        print("")

    def check_table_names(self, check):
        return [t for t in self.table_names if t.startswith(check)]

    @split_and_check_args(2)
    def do_x_load(self, split_args):
        """Loads in an external datasource to the database as a table.
        [!] This command is experimental.
        Arg 1: input_path
        Arg 2: table_name"""
        input_path, table_name = split_args
        io.load_external_table(self.spark, input_path, table_name)

    @split_and_check_args(2)
    def do_x_parquet_from_csv(self, split_args):
        """Reads a csv and creates a parquet file.
        [!] This command is experimental.
        Arg 1: input_path
        Arg 2: output_path"""
        csv_path, parquet_path = split_args
        io.read_csv_and_write_parquet(self.spark, csv_path, parquet_path)

    def do_x_crosswalk(self, args):
        """Export a crosswalk of all predicted matches for round 1 and round 2 linking.
        [!] This command is experimental.
        Arg 1: output path
        Arg 2: comma seperated list of variables to export
        Usage: crosswalk [output_path] [list_of_variables]
        Example: 'crosswalk /mypath histid,serial,pernum,sex,age,bpl'"""
        flags, split_args = self.extract_flags_from_args(
            ["--include-rounds"], args.split()
        )
        include_round = "--include-rounds" in flags

        # Flags have been removed from the split_args already, so that
        # the correct number gets checked.
        if self.check_arg_count(split_args, 2, self.do_x_crosswalk.__doc__):
            return

        output_path, variables_string = split_args
        variables = list([v.lower() for v in variables_string.split(",")])

        if include_round:
            print("Including round numbers in exported data")
        else:
            print(
                "Not including rounds in export, to include them use the --include-rounds flag."
            )

        x_reporting.export_crosswalk(self.spark, output_path, variables, include_round)

    @split_and_check_args(2)
    def do_x_tfam(self, split_args):
        """Show the family of a training match.
        [!] This command is experimental.
        Arg 1: id_a
        Arg 2: id_b"""
        id_col = self.link_run.config["id_column"]
        id_a, id_b = split_args

        x_tfam.tfam(self.link_run, id_col, id_a, id_b)

    @split_and_check_args(2)
    def do_x_tfam_raw(self, split_args):
        """Show the family of a potential match.
        [!] This command is experimental.
        Arg 1: id_a
        Arg 2: id_b"""
        start = timer()

        id_col = self.link_run.config["id_column"]
        id_a, id_b = split_args

        x_tfam.tfam_raw(self.link_run, id_col, id_a, id_b)

        end = timer()
        elapsed_time = round(end - start, 2)

        print(f"Time: {elapsed_time}s")
        logging.info(f"Finished: hh_tfam display - {elapsed_time}")

    @split_and_check_args(2)
    def do_x_hh_tfam(self, split_args):
        """Show the family of a training match.
        [!] This command is experimental.
        Arg 1: id_a
        Arg 2: id_b"""
        start = timer()

        id_col = self.link_run.config["id_column"]
        id_a, id_b = split_args

        x_tfam.hh_tfam(self.link_run, id_col, id_a, id_b)

        end = timer()
        elapsed_time = round(end - start, 2)

        print(f"Time: {elapsed_time}s")
        logging.info(f"Finished: hh_tfam display - {elapsed_time}")

    @split_and_check_args(3)
    def do_x_hh_tfam_2a(self, split_args):
        """Show the family of a training match.
        [!] This command is experimental.
        Arg 1: id_a option 1
        Arg 2: id_a option 2
        Arg 3: id_b"""
        id_col = self.link_run.config["id_column"]
        id_a1, id_a2, id_b = split_args

        x_tfam.hh_tfam_2a(self.link_run, id_col, id_a1, id_a2, id_b)

    @split_and_check_args(3)
    def do_x_hh_tfam_2b(self, split_args):
        """Show the family of a training match.
        [!] This command is experimental.
        Arg 1: id_a
        Arg 2: id_b option 1
        Arg 3: id_b option 2"""
        id_col = self.link_run.config["id_column"]
        id_a, id_b1, id_b2 = split_args

        x_tfam.hh_tfam_2b(self.link_run, id_col, id_a, id_b1, id_b2)
