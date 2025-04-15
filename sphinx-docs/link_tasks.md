# Link Tasks

## Preprocessing

### Overview

Read in raw data and prepare it for linking. This task may include a variety of
transformations on the data, such as stripping out whitespace and normalizing strings
that have common abbreviations. The same transformations are applied to both input
datasets.

### Task steps

* Step 0: Read raw data in from Parquet or CSV files. Register the raw dataframes with the program.
* Step 1: Prepare the dataframes for linking. Perform substitutions, transformations, and column mappings as requested.

### Related Configuration Sections

* The [`datasource_a` and `datasource_b`](config.html#data-sources) sections specify where to find the input data.
* [`column_mappings`](column_mappings.html#column-mappings),
[`feature_selections`](feature_selection_transforms.html#feature-selection-transforms),
and [`substitution_columns`](substitutions.html#substitutions) may all be used to define transformations on the input data.
* The [`filter`](config.html#filter) section may be used to filter some records out of the input data
as they are read in.

## Training and Household Training

### Overview

Train a machine learning model to use for classification of potential links. This
requires training data, which is read in in the first step. Comparison features
are generated for the training data, and then the model is trained on the data
and saved for use in the Matching task. The last step optionally saves some metadata
like feature importances or coefficients for the model to help with introspection.

### Task steps

The first three steps in each of these tasks are the same:
* Step 0: Ingest the training data from a CSV file.
* Step 1: Create comparison features.
* Step 2: Train and save the model.
* Step 3: Save the coefficients or feature importances of the model for inspection.
  This step is skipped by default. To enable it, set the `training.feature_importances`
  and/or the `hh_training.feature_importances` config attribute(s) to true in your config file.

### Related Configuration Sections

* The [`training`](config.html#training-and-models) section is the most important
for Training and provides configuration attributes for many aspects of the task.
For Household Training, use the [`hh_training`](config.html#household-training-and-models)
section instead.
* [`comparison_features`](config.html#comparison-features) and
[`pipeline_features`](pipeline_features.html#pipeline-generated-features) are
both generated in order to train the model. These sections are also used extensively
by the Matching task.

## Matching

### Overview

Run the linking algorithm, generating a table with potential matches between records in the two datasets.
This is the core of hlink's work and may take the longest of all of the tasks. Universe
definition and blocking reduce the number of comparisons needed when
determining potential matches, which can drastically improve the runtime of Matching.

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
before matching. Two records are eligible to match with one another only if they
are grouped into the same blocking bucket.
* [`comparison_features`](config.html#comparison-features) support computing features
on each record. These features may be passed to a machine learning model through the
[`training`](config.html#training-and-models) section and/or passed to deterministic
rules with the [`comparisons`](config.html#comparisons) section. There are many
different [comparison types](comparison_features) available for use with
`comparison_features`.
* [`pipeline_features`](pipeline_features.html#pipeline-generated-features) are machine learning transformations
useful for reshaping and interacting data before they are fed to the machine learning
model.

## Household Matching

### Overview

Generate a table with potential matches between households in the two datasets.

### Task steps

* Step 0: Block on households.
* Step 1: Filter households based on `hh_comparisons` configuration settings.
* Step 2: Score the potential matches with the trained model. This step will be automatically skipped if machine learning is not being used.

### Related Configuration Sections

* [`comparison_features`](config.html#comparison-features) and [`pipeline_features`](pipeline_features.html#pipeline-generated-features) are used as they are in the Matching task.
* [`hh_comparisons`](config.html#household-comparisons) correspond to `comparisons` in the Matching task and may be thought of as "post-blocking filters". Only potential matches that pass these comparisons will be eligible for being scored as matches.
* [`hh_training`](config.html#household-training-and-models) corresponds to `training` in Matching.

## Model Exploration and Household Model Exploration

### Overview

Evaluate the performance of different types of models and different parameter combinations
on training data. These tasks are highly configurable and are typically not part of a full
linking run. Instead, they are usually run ahead of time, and then the best-performing
model is chosen and used for the full linking run.

### Task steps
The steps in each of these tasks are the same:
 * Step 0: Ingest the training data file specified in the config with the `dataset` attribute.
 * Step 1: Create training features on the training data. If the `use_training_data_features`
   attribute is provided in the respective training config section, then instead read
   features from the training data file.
 * Step 2: Run `n_training_iterations` number of train-test splits on each of the
   models in the config `model_parameters`.

### Related Configuration Sections

* [`training`](config.html#training-and-models) is used extensively by Model Exploration,
  and [`hh_training`](config.html#household-training-and-models) is used extensively
  by Household Model Exploration.
* [`comparison_features`](config.html#comparison-features) and
  [`pipeline_features`](pipeline_features.html#pipeline-generated-features) are
  used to generate features that are passed as input to the trained models.

## Reporting

### Overview

Report on characteristics of the linked data. This task is experimental and focused
primarily on demographic census data. At the moment, it does not allow very much
configuration.

### Task steps

* Step 0: For households with anyone linked in Matching, report the percent of remaining household members linked in Household Matching.
* Step 1: Report on the representivity of linked data compared to source populations.
* Step 2: Pull in key demographic data for linked individuals and export a fixed-width crosswalk file.

### Related Configuration Sections

* The `alias` attributes are read from both [`datasource_a`](config.html#data-sources) and [`datasource_b`](config.html#data-sources). The step uses them to construct the output reports.
