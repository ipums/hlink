# Link Tasks

## Preprocessing

### Overview

Read in raw data and prepare it for linking. This task may include a variety of
transformations on the data, such as stripping out whitespace and normalizing strings
that have common abbreviations. The same transformations are applied to both input
datasets.

### Task steps

* Step 0: Read raw data in from Parquet or CSV Files. Register the raw dataframes with the program. 
* Step 1: Prepare the dataframes for linking. Perform substitutions, transformations, and column mappings as requested.

### Related Configuration Sections

* The [`datasource_a` and `datasource_b`](config.html#data-sources) sections specify where to find the input data.
* [`column_mappings`](column_mapping_transforms.html), [`feature_selections`](feature_selection_transforms.html),
and [`substitution_columns`](substitutions.html) may all be used to define transformations on the input data.
* The [`filter`](config.html#filter) section may be used to filter some records out of the input data
as they are read in.

## Training and Household Training

### Overview

Train a machine learning model to use for classification of potential links.

### Task steps

The steps in each of these tasks are the same:
* Step 0: Ingest the training data from a .csv file.
* Step 1: Create comparison features.
* Step 2: Train and save the model.

## Matching

### Overview

Run the linking algorithm, generating a table with potential matches between individuals in the two datasets.
This is the core of hlink's work and may take the longest of all of the tasks. To reduce
the total number of comparisons needed during matching, blocking may be performed
to separate records into buckets before matching.

### Task steps

* Step 0: Perform blocking, separating records into different buckets to reduce the total number
of comparisons needed during matching. Some columns may be "exploded" here if needed.
* Step 1: Run the matching algorithm, outputting potential matches to the `potential_matches` table.
* Step 2: Score the potential matches with the trained model. This step will be automatically skipped if machine learning is not being used.

### Related Configuration Sections

* The [`potential_matches_universe`](config.html#potential-matches-universe) section may be used to
provide a universe for matches in the form of a SQL condition. Only records that satisfy the
condition are eligible for matching.
* [`blocking`](config.html#blocking) specifies how to block the input records into separate buckets
before matching.

## Household Matching

### Overview

Generate a table with potential matches between households in the two datasets.

### Task steps

* Step 0: Block on households.
* Step 1: Filter households based on `hh_comparisons` configuration settings. 
* Step 2: Score the potential matches with the trained model. This step will be automatically skipped if machine learning is not being used.

## Model Exploration and Household Model Exploration

### Overview

There are two dedicated linking tasks for model exploration.  `model_exploration` uses configuration settings from the Training section of the config file.  `hh_model_exploration` uses configuration settings from the Household Training section of the config file. See documentation of the [`[training]`](config.html#training-and-models) and [`[hh_training]`](config.html#household-training-and-models) config sections for more details. 

### Task steps
The steps in each of these tasks are the same:
 * Step 0: Ingest the specified training data file specified in the config with the  `dataset` tag.
 * Step 1: Create training features on the training data, or use those in the training data file (specified in the respective config section with the `use_training_data_features` flag).
 * Step 2: Run `n_training_iterations` number of train-test splits on each of the models in the config `model_parameters`.

## Reporting

### Overview

Report on characteristics of the linked data.

### Task steps

* Step 0: For households with anyone linked in round 1, report the percent of remaining household members linked in round 2.
* Step 1: Report on the representivity of linked data compared to source populations.
* Step 2: Pull in key demographic data for linked individuals and export a fixed-width crosswalk file.
