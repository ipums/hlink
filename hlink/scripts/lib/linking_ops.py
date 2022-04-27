# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from colorama import init as colorama_init, deinit as colorama_deinit, Fore, Style


def show_step_info(link_task, link_run):
    """Show step information for the given `link_task`."""
    colorama_init()
    print(Fore.CYAN + f"Link task: {link_task}" + Style.RESET_ALL)

    steps = link_task.get_steps()
    tables = link_run.known_tables

    for (i, step) in enumerate(steps):
        print(Fore.GREEN + f"step {i}: {step}" + Style.RESET_ALL)

        print("\tTables used:")
        for input_table_name in step.input_table_names:
            print(f"\t\t{tables[input_table_name]}")

        if len(step.input_model_names) > 0:
            print("\tModels loaded:")
            for input_model_name in step.input_model_names:
                print(Fore.MAGENTA + f"\t\t{input_model_name}" + Style.RESET_ALL)

        print("\tTables created:")
        for output_table_name in step.output_table_names:
            print(f"\t\t{tables[output_table_name]}")

        if len(step.output_model_names) > 0:
            print("\tModels saved:")
            for output_model_name in step.output_model_names:
                print(Fore.MAGENTA + f"\t\t{output_model_name}" + Style.RESET_ALL)

    colorama_deinit()


def show_tasks(current_task, link_run, link_task_choices):
    """Show information about the available link tasks for the link run.

    Args:
        current_task (LinkTask): the currently active link task
        link_run (LinkRun)
        link_task_choices (Dict[str, LinkTask]): a dict mapping string names to link tasks
    """
    colorama_init()
    print(Fore.CYAN + f"Current link task: {current_task}")

    print("Linking task choices are: " + Style.RESET_ALL)
    for link_task in link_task_choices:
        task_inst = link_run.get_task(link_task)
        print(Fore.GREEN + f"{link_task} :: {task_inst}" + Style.RESET_ALL)

        input_tables = set()
        output_tables = set()
        for step in task_inst.get_steps():
            input_tables.update(set(step.input_table_names))
            output_tables.update(set(step.output_table_names))

        input_tables = input_tables - output_tables

        if len(input_tables) == 0:
            print("\tRequires no preexisting tables.")
        else:
            print("\tRequires tables: " + str(input_tables))
        if len(output_tables) == 0:
            print("\tProduces no persistent tables.")
        else:
            print("\tProduces tables: " + str(output_tables))

    colorama_deinit()
