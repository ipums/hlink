# Running hlink

## Using hlink as a Library

hlink can be used as a Python library for scripting linking jobs. It provides some high-level classes and
functions for interacting with Spark, handling configuration, and running linking tasks and steps.

The main class in the library is `LinkRun`, which represents a complete linking job. It provides access
to each of the link tasks and their steps. Here is an example script that uses `LinkRun` to do some linking.
Below we go into more detail on each of the important aspects of the script.

```python
from hlink.linking.link_run import LinkRun
from hlink.spark.factory import SparkFactory
from hlink.configs.load_config import load_conf_file

# First we create a SparkSession with all default configuration settings.
factory = SparkFactory()
spark = factory.create()

# Now let's load in our config file.
config = load_conf_file("./my_conf")

lr = LinkRun(spark, config)

# Get some information about each of the steps in the
# preprocessing task.
prep_steps = lr.preprocessing.get_steps()
for (i, step) in enumerate(prep_steps):
    print(f"Step {i}:", step)
    print("Required input tables:", step.input_table_names)
    print("Generated output tables:", step.output_table_names)

# Run all of the steps in the preprocessing task.
lr.preprocessing.run_all_steps()

# Run the first two steps in the matching task.
lr.matching.run_step(0)
lr.matching.run_step(1)

# Get the potential_matches table.
matches = lr.get_table("potential_matches")

assert matches.exists()

# Get the Spark DataFrame for the potential_matches table.
matches_df = matches.df()
```

Each link task can be accessed through the `LinkRun` as an attribute like `lr.preprocessing` or `lr.hh_model_exploration`.
Link steps for each task can be run with `task.run_all_steps()` or `task.run_step(i)`. The easiest way to
access Spark tables is through `lr.get_table()`. This method returns an `hlink.linking.table.Table` object, which provides
an interface to easily check if the table exists, get its Spark DataFrame, or drop it.

To create a `LinkRun`, we need to set up a `pyspark.sql.SparkSession` object. The most convenient way to do this is through
the `hlink.spark.factory.SparkFactory` class. `SparkFactory` defines many default configuration values which can be adjusted as needed.

```
from hlink.spark.factory import SparkFactory

factory = SparkFactory()
spark = factory.set_local().set_num_cores(8).set_executor_memory("5G").create()
```

We'll also need to load in a config to get the `LinkRun` up and running. A config is
a dictionary with string keys, often read in from a TOML or JSON file. The
`hlink.configs.load_config.load_conf_file` function is helpful for reading in config files,
as are the `json` and `toml` python modules. For more information on writing config files,
please see the [Configuration](config) page.

In the `examples/tutorial` directory there is an example script that uses hlink as a library to
link people between two datasets. The example includes a working config file.

## Interactive Mode

In addition to a library, hlink provides a command-line interface, which can be started
with the `hlink` command.

### Starting the program

The program takes as input a TOML or JSON configuration file, described in the [Configuration](config) page.  Parameters described in the config include paths to input data files, paths to training data files, instructions for generating machine learning features, and model parameters.  The configuration enables reproducible runs that should produce the same results on the same input data.

All input flags can be printed to the console by running `hlink --help`.

```
cpu ~$ hlink --help
usage: hlink [-h] [--user USER] [--cores CORES]
             [--executor_memory EXECUTOR_MEMORY] [--task TASK]
             [--execute_tasks EXECUTE_TASKS [EXECUTE_TASKS ...]]
             [--execute_command EXECUTE_COMMAND [EXECUTE_COMMAND ...]]
             [--conf CONF]

Hierarchical linking program.

optional arguments:
  -h, --help            show this help message and exit
  --user USER           run as a specific user
  --cores CORES         the max number of cores to use on
  --executor_memory EXECUTOR_MEMORY
                        the memory per executor to use
  --task TASK           The initial task to begin processing.
  --execute_tasks EXECUTE_TASKS [EXECUTE_TASKS ...]
                        Execute a series of tasks then exit the program.
  --execute_command EXCUTE_COMMAND [EXECUTE_COMMAND ...]
                        Execute a single command then exit the program.
  --conf CONF, --run CONF
                        Specify a filepath where your config file for the run
                        is located.
```

To run the program in interactive mode using a configuration file at a specified path, say `./fullcount_1870_1880.toml`, run a command following this pattern:

```bash
hlink --conf=./full_count_1870_1880.toml
```

After the program has started, you will see a prompt that looks like this:

```
hlink $
```

Type `help` or `?` and hit enter to see a list of commands; type `help <command>` to see the help text of a specific command.
Commands that start with "x_" are experimental. They may be unstable or missing some documentation.

```
hlink $ ?

Documented commands (type help <topic>):
========================================
analyze        get_steps      set_preexisting_tables  x_persist
borrow_tables  get_tasks      set_print_sql           x_sql
count          help           show                    x_sqlf
csv            ipython        showf                   x_summary
desc           list           x_crosswalk             x_tab
drop           q              x_hh_tfam               x_tfam
drop_all       reload         x_hh_tfam_2a            x_tfam_raw
drop_all_prc   run_all_steps  x_hh_tfam_2b            x_union
drop_all_temp  run_step       x_load
get_settings   set_link_task  x_parquet_from_csv
```

### Running Linking Tasks and Steps

The program is organized into a hierarchy of tasks and steps. The five major tasks are `preprocessing`, `training`, `matching`, `hh_training`, and `hh_matching`, and within each task are multiple steps.
To see all linking tasks, run the command `get_tasks`.  You should see something like this:

```
hlink $ get_tasks
Current link task: Preprocessing
Linking task choices are: 
preprocessing :: Preprocessing
        Requires no preexisting tables.
        Produces tables: {'prepped_df_a', 'prepped_df_b', 'raw_df_b', 'raw_df_a'}
training :: Training
        Requires tables: {'prepped_df_a', 'prepped_df_b'}
        Produces tables: {'training_data', 'training_features'}
matching :: Matching
        Requires tables: {'prepped_df_a', 'prepped_df_b'}
        Produces tables: {'scored_potential_matches', 'potential_matches_prepped', 'potential_matches', 'exploded_df_b', 'exploded_df_a', 'predicted_matches'}
hh_training :: Household Training
        Requires tables: {'prepped_df_a', 'prepped_df_b'}
        Produces tables: {'hh_training_features', 'hh_training_data'}
hh_matching :: Household Matching
        Requires tables: {'prepped_df_a', 'predicted_matches', 'prepped_df_b'}
        Produces tables: {'hh_predicted_matches', 'hh_scored_potential_matches', 'hh_potential_matches', 'hh_blocked_matches', 'hh_potential_matchs_prepped'}
model_exploration :: Model Exploration
        Requires tables: {'prepped_df_a', 'prepped_df_b'}
        Produces tables: {'model_eval_training_vectorized', 'model_eval_training_data', 'model_eval_repeat_FPs', 'model_eval_training_features', 'model_eval_training_results', 'model_eval_repeat_FNs'}
hh_model_exploration :: Household Model Exploration
        Requires tables: {'prepped_df_a', 'prepped_df_b'}
        Produces tables: {'hh_model_eval_training_vectorized', 'hh_model_eval_repeat_FPs', 'hh_model_eval_repeat_FNs', 'hh_model_eval_training_results', 'hh_model_eval_training_features', 'hh_model_eval_training_data'}
reporting :: Reporting
        Requires tables: {'prepped_df_a', 'hh_predicted_matches', 'prepped_df_b', 'predicted_matches', 'raw_df_b', 'raw_df_a'}
        Produces no persistent tables.
```

Each linking task will interact with Spark tables within the program. To see a list of tables run the command `list`. To also see hidden intermediate tables, run `list all`. If you have just started the program for the first time, you should see no tables created yet:

```
hlink $ list
+--------+---------+-----------+
|database|tableName|isTemporary|
+--------+---------+-----------+
+--------+---------+-----------+
```

To see information about the steps of the task you are currently on, run `get_steps`. You should see something that looks like this:

```text
Link task: Preprocessing
step 0: register raw dataframes
        Tables used:
        Tables created:
                Table 'raw_df_a' <- Preprocessing: Raw data read in from datasource A
                Table 'raw_df_b' <- Preprocessing: Raw data read in from datasource B
step 1: prepare dataframes
        Tables used:
                Table 'raw_df_a' <- Preprocessing: Raw data read in from datasource A
                Table 'raw_df_b' <- Preprocessing: Raw data read in from datasource B
        Tables created:
                Table 'prepped_df_a' <- Preprocessing: Preprocessed data from source A with selected columns and features
                Table 'prepped_df_b' <- Preprocessing: Preprocessed data from source B with selected columns and features
```

To change your current link task, run `set_link_task <task_name>`, where `<task_name>` is the name of the link task. 

Once you are sure that you are on the right task, you can use the `run_step <num>` command to run a step. For example if you run `run_step 0` you should see something like this:

```
hlink $ run_step 0
Link task: Preprocessing
Running step 0: register raw dataframes
Finished step 0: register raw dataframes in 5.85s 
```

After the step is complete, you can run `list` to see what tables it created:

```
hlink $ list
+--------+---------+-----------+-------------------------------------------------+
|database|tableName|isTemporary|description                                      |
+--------+---------+-----------+-------------------------------------------------+
|linking |raw_df_a |false      |Preprocessing: Raw data read in from datasource A|
|linking |raw_df_b |false      |Preprocessing: Raw data read in from datasource B|
+--------+---------+-----------+-------------------------------------------------+
```

To run all steps in a task, use the `run_all_steps <tasks>` command, where `<tasks>` is a list of tasks you want to run all the steps for. By default this command will run all the steps for the current task.

### Example interactive mode workflow 

1) Create a config file and put it in your hlink config directory. 
   For example:
   ```
   /path/to/conf/full_count_1870_1880.toml
   ```

2) Launch the hlink program in interactive mode:
    ```bash
    hlink --conf=/path/to/conf/full_count_1870_1880
   ```
3) Run the tasks you want to complete:
   ```
    hlink $ run_all_steps preprocessing training matching
    ```
4) List the created tables: 
    ```
   hlink $ list 
   ```
5) Export the results: 
    ```
   hlink $ csv predicted_matches /my/output/file.csv
   ```
