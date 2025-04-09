# Configuration
1. [Basic Example Config File](#basic-config-file)
2. [Advanced Example Config File](#advanced-config-file)
3. [Top-Level Configs](#top-level-configs)
4. [Data Sources](#data-sources)
5. [Filter](#filter)
6. [Column Mappings](#column-mappings)
7. [Substitution Columns](#substitution-columns)
8. [Feature Selections](#feature-selections)
9. [Potential Matches Universe](#potential-matches-universe)
10. [Blocking](#blocking)
11. [Comparisons](#comparisons)
12. [Household Comparisons](#household-comparisons)
13. [Comparison Features](#comparison-features)
14. [Pipeline-Generated Features](#pipeline-generated-features)
15. [Training and Model Exploration](#training-and-model-exploration)
16. [Household Training and Model Exploration](#household-training-and-model-exploration)
17. [Household Matching](#household-matching)

## Basic Config File

The config file tells the hlink program what to link and how to link it. A description of the different sections of
a configuration file are below. For reference, here is an example of a relatively basic config file. This config file
is used by the `examples/tutorial/tutorial.py` script for linking, and there is a more detailed discussion of the config
file in the README in `examples/tutorial`.

Note that this config is written in TOML, but hlink is also able to work with JSON config files.

```
id_column = "id"

[datasource_a]
alias = "a"
file = "data/A.csv"

[datasource_b]
alias = "b"
file = "data/B.csv"

[[column_mappings]]
column_name = "NAMEFRST"
transforms = [
    {type = "lowercase_strip"}
]

[[column_mappings]]
column_name = "NAMELAST"
transforms = [
    {type = "lowercase_strip"}
]

[[column_mappings]]
column_name = "AGE"
transforms = [
    {type = "add_to_a", value = 10}
]

[[column_mappings]]
column_name = "SEX"

[[blocking]]
column_name = "SEX"

[[blocking]]
column_name = "AGE_2"
dataset = "a"
derived_from = "AGE"
expand_length = 2
explode = true

[[comparison_features]]
alias = "NAMEFRST_JW"
column_name = "NAMEFRST"
comparison_type = "jaro_winkler"

[[comparison_features]]
alias = "NAMELAST_JW"
column_name = "NAMELAST"
comparison_type = "jaro_winkler"

[comparisons]
operator = "AND"

[comparisons.comp_a]
comparison_type = "threshold"
feature_name = "NAMEFRST_JW"
threshold = 0.79

[comparisons.comp_b]
comparison_type = "threshold"
feature_name = "NAMELAST_JW"
threshold = 0.84
```

## Advanced Config File

Here is an example of a more complex config file that makes use of more of hlink's features.
It uses machine learning to probabilistically link the two datasets.

```
id_column = "histid"
drop_data_from_scored_matches = false

# --------- DATASOURCES --------------
[datasource_a]
alias = "us1900"
file = "/path/to/us1900m_usa.P.parquet"

[datasource_b]
alias = "us1910"
file = "/path/to/us1910m_usa.P.parquet"

# --------- FILTERS --------------

[[filter]]
expression = "NAMELAST is not null and NAMELAST != ''"

[[filter]]
training_data_subset = true
datasource = "a"

[[filter]]
expression = "age >= 5"
datasource = "b"

# --------- COLUMN MAPPINGS --------------

[[column_mappings]]
column_name = "serialp"

[[column_mappings]]
column_name = "sex"

[[column_mappings]]
column_name = "age"

[[column_mappings]]
column_name = "namelast"

[[column_mappings]]
alias = "namefrst_clean"
column_name = "namefrst"
transforms = [
  { type = "lowercase_strip" },
  { type = "rationalize_name_words" },
  { type = "remove_qmark_hyphen"},
  { type = "replace_apostrophe"},
  { type = "remove_suffixes",  values = ["jr", "sr", "ii", "iii"] },
  { type = "remove_alternate_names"},
  { type = "condense_strip_whitespace"},
]

[[column_mappings]]
alias = "namefrst_split"
column_name = "namefrst_clean"
transforms = [ { type = "split" } ]

[[column_mappings]]
alias = "namefrst_std"
column_name = "namefrst_split"
transforms = [
  { type = "array_index", value = 0 }
]

[[column_mappings]]
alias = "bpl_orig"
column_name = "bpl"
transforms = [
  { type = "divide_by_int", value = 100 },
  { type = "get_floor" }
]

[[column_mappings]]
alias = "statefip"
column_name = "statefip_h"

[[column_mappings]]
column_name = "birthyr"
alias = "clean_birthyr"
[[column_mappings.transforms]]
type = "mapping"
mappings = {9999 = "", 1999 = ""}
output_type = "int"

[[column_mappings]]
alias = "relate_div_100"
column_name = "relate"
transforms = [
  { type = "divide_by_int", value = 100 },
  { type = "get_floor" }
]

# --------- SUBSTITUTIONS --------------

[[substitution_columns]]
column_name = "namefrst_std"

[[substitution_columns.substitutions]]
join_column = "sex"
join_value = "1"
substitution_file = "/path/to/name_std/male.csv"

[[substitution_columns.substitutions]]
join_column = "sex"
join_value = "2"
substitution_file = "/path/to/name_std/female.csv"

# --------- FEATURE SELECTIONS --------------

[[feature_selections]]
input_column = "clean_birthyr"
output_column = "replaced_birthyr"
condition = "case when clean_birthyr is null or clean_birthyr == '' then year - age else clean_birthyr end"
transform = "sql_condition"

[[feature_selections]]
input_column = "namelast"
output_column = "namelast_bigrams"
transform = "bigrams"

[[feature_selections]]
input_column = "bpl_orig"
output_column = "bpl_clean"
condition = "case when bpl_str == 'washington' and bpl2_str=='washington' then 53 when (bpl_str is null or bpl_str == '') and bpl2_str=='washington' then 53 when bpl_str == 'washington' and (bpl2_str=='' or bpl2_str is null) then 53 else bpl_orig end"
transform = "sql_condition"

[[feature_selections]]
input_column = "bpl_clean"
output_column = "region"
transform = "attach_variable"
region_dict = "/path/to/region.csv"
col_to_join_on = "bpl"
col_to_add = "region"
null_filler = 99
col_type = "float"

# --------- POTENTIAL MATCHES UNIVERSE -------------

[[potential_matches_universe]]
expression = "sex == 1"

# --------- BLOCKING --------------

[[blocking]]
column_name = "sex"

[[blocking]]
column_name = "birthyr_3"
dataset = "a"
derived_from = "replaced_birthyr"
expand_length = 3
explode = true

[[blocking]]
column_name = "namelast_bigrams"
explode = true

# --------- COMPARISONS --------------

[comparisons]
operator = "AND"

[comparisons.comp_a]
comparison_type = "threshold"
feature_name = "namefrst_std_jw"
threshold = 0.8

[comparisons.comp_b]
comparison_type = "threshold"
feature_name = "namelast_jw"
threshold = 0.75

# --------- HOUSEHOLD COMPARISIONS (post-blocking filters) -------------

[hh_comparisons]
comparison_type = "threshold"
feature_name = "byrdiff"
threshold_expr = "<= 10"

# --------- COMPARISON FEATURES --------------

[[comparison_features]]
alias = "region"
column_name = "region"
comparison_type = "fetch_a"
categorical = true

[[comparison_features]]
alias = "namefrst_std_jw"
column_name = "namefrst_std"
comparison_type = "jaro_winkler"

[[comparison_features]]
alias = "namelast_jw"
column_name = "namelast"
comparison_type = "jaro_winkler"

[[comparison_features]]
alias = "sex_equals"
column_name = "sex"
comparison_type = "equals"
categorical = true

[[comparison_features]]
alias = "relate_a"
column_name = "relate_div_100"
comparison_type = "fetch_a"

# --------- PIPELINE-GENERATED FEATURES ------------

[[pipeline_features]]
input_columns =  ["sex_equals", "region"]
output_column =  "sex_region_interaction"
transformer_type =  "interaction"

[[pipeline_features]]
input_column = "relate_a"
output_column = "relatetype"
transformer_type = "bucketizer"
categorical = true
splits = [1,3,5,9999]

# --------- TRAINING --------------

[training]

independent_vars = [ "namelast_jw", "region", "hits", "sex_region_interaction", "relatetype"]
scale_data = false

dataset = "/path/to/training_data.csv"
dependent_var = "match"
score_with_model = true
use_training_data_features = false
split_by_id_a = true
decision = "drop_duplicate_with_threshold_ratio"

n_training_iterations = 2
model_parameter_search = {strategy = "grid"}
model_parameters = [ 
    { type = "random_forest", maxDepth = [7], numTrees = [100], threshold = [0.05, 0.005], threshold_ratio = [1.2, 1.3] },
    { type = "logistic_regression", threshold = [0.50, 0.65, 0.80], threshold_ratio = [1.0, 1.1] }
]
    
chosen_model = { type = "logistic_regression", threshold = 0.5, threshold_ratio = 1.0 }

# --------- HOUSEHOLD TRAINING --------------

[hh_training]

prediction_col = "prediction"
hh_col = "serialp"

independent_vars = ["namelast_jw", "namefrst_std_jw", "relatetype", "sex_equals"]
scale_data = false

dataset = "/path/to/hh_training_data_1900_1910.csv"
dependent_var = "match"
score_with_model = true
use_training_data_features = false
split_by_id_a = true
decision = "drop_duplicate_with_threshold_ratio"

n_training_iterations = 10
model_parameter_search = {strategy = "explicit"}
model_parameters = [
    { type = "random_forest", maxDepth = 6, numTrees = 50, threshold = 0.5, threshold_ratio = 1.0 },
    { type = "probit", threshold = 0.5, threshold_ratio = 1.0 }
]
    
chosen_model = { type = "logistic_regression", threshold = 0.5, threshold_ratio = 1.0 }

```

## Top level configs

These configs should go at the top of your config file under no header:

*id_column*

Required.  Specify the id column that uniquely identifies a record in each dataset.
```
id_column = "id"
```

*drop_data_from_scored_matches*

Optional.  Whether or not the scored potential matches should be output with full features data, or just ids and match information.
```
drop_data_from_scored_matches = false
```

## Data sources

* Header names: `datasource_a`, `datasource_b`
* Description: Specifies your input data.
* Required: True
* Type: Object
* Attributes:
  * `alias` -- Type: `string`. The short name for the datasource. Must be alphanumeric with no spaces.
  * `file` -- Type: `string`. Required. The path to the input file. The file can be `csv` or `parquet`.
  * `convert_ints_to_longs` -- Type: `boolean`. Optional. If set to true, automatically
    convert each column with type `int` in the input file to type `long`. This can be
    especially useful when reading from CSV files, as Spark may assume that columns are
    type `int` when they should be `long`. Parquet files have their own schema included
    in the file, so this may be less useful for them. Note that Spark also sometimes
    uses the term `bigint` to mean the same thing as `long`.

```
[datasource_a]
alias = "us1900"
file = "/path/to/my_file.csv"
convert_ints_to_longs = true
```

## Filter

* Header name: `filter`
* Description: Specifies filters to apply to your input data.
* Required: False
* Type: List
* Attributes:
  * `expression` -- Type: `string`. SQL expression to apply to your input datasets. Can not be combined with `training_data_subset` in a single filter.
  * `training_data_subset` -- Type: `boolean`. If set to true, will subset your input data to only include records that are also in your training data. Can not be combined with `expression` in a single filter.
  * `datasource` -- Type: `string`. If you want to limit the filter to operate only on dataset a or b, you can specify that with this attribute.

```
[[filter]]
training_data_subset = true
datasource = "a"

[[filter]]
expression = "NAMELAST is not null and NAMELAST != ''"

[[filter]]
expression = "age >= 5"
datasource = "b"
```


## [Column Mappings](column_mappings)

* Header name: `column_mappings`
* Description: Base column mappings and transformations to extract from your
  input datasets. Each column mapping requires a `column_name` which tells it
  which input column to read from. Optionally you may provide an `alias` for
  the column and `transforms` to modify it as it is read in. There are some additional
  attributes listed below that are meant for advanced usage. These are described
  in more detail on the [column mappings](column_mappings) page.
* Required: True
* Type: List
* Attributes:
  * `column_name` -- Type: `string`. The name of the column in the input data.
  * `alias` -- Type: `string`. Optional. The new name of the column to use
    in hlink. By default, this is the same as `column_name`.
  * `transforms` -- Type: `List`. Optional. A list of transforms to apply, in
    order, to the input data. See the [column mapping transforms](column_mappings.html#transforms)
    section for more information.
  * `set_value_column_a` -- Type: `Any`. Optional. Set all records for dataset
    A to the given literal value.
  * `set_value_column_b` -- Type: `Any`. Optional. Set all records for dataset
    B to the given literal value.
  * `override_column_a` -- Type: `string`. Read from this column in dataset A
    instead of the column specified with `column_name`.
  * `override_column_b` -- Type: `string`. Read from this column in dataset B
    instead of the column specified with `column_name`.
  * `override_transforms` -- Type: `List`. Transforms to apply to the override
    column specified with `override_column_a` or `override_column_b`.

```
[[column_mappings]]
column_name = "age"

[[column_mappings]]
alias = "namefrst_clean"
column_name = "namefrst"
transforms = [
  { type = "lowercase_strip" },
  { type = "rationalize_name_words" },
  { type = "remove_qmark_hyphen"},
  { type = "replace_apostrophe"},
  { type = "remove_suffixes",  values = ["jr", "sr", "ii", "iii", "iv", "v", "vi", "vii", "viii"] },
  { type = "remove_alternate_names"},
  { type = "condense_strip_whitespace"}
]
```

## [Substitution Columns](substitutions)

* Header name: `substitution_columns`
* Description: Substitutions to apply to data after column mappings.
* Required: False
* Type: List
* Attributes:
  * `column_name` -- Type: `string`. Required. Column to apply substitutions to.
  * `substitutions` -- Type: `list`. A list of substitutions to apply. See the [substitutions](substitutions) section for more information.

```
[[substitution_columns]]
column_name = "namefrst_std"

[[substitution_columns.substitutions]]
join_column = "sex"
join_value = "1"
substitution_file = "/path/to/name_std/male.csv"

[[substitution_columns.substitutions]]
join_column = "sex"
join_value = "2"
substitution_file = "/path/to/name_std/female.csv"
```


## [Feature Selections](feature_selection_transforms)

* Header name: `feature_selections`
* Description: A list of feature selections to apply to the input data after substitutions and column mappings. See the [feature selection transforms](feature_selection_transforms) section for more information, including information on the specific transforms available.

* Required: False
* Type: List
* Attributes: 
  * `input_column` -- Type: `string`. Required.  The name of the input column.
  * `output_column` -- Type: `string`. Required.  The name of the output column.
  * `transform` -- Type: `string`.  The name of the transform to apply to the column.
  * Other attributes vary depending on transform type.

```
[[feature_selections]]
input_column = "namelast_clean"
output_column = "namelast_clean_bigrams"
transform = "bigrams"

[[feature_selections]]
input_column = "bpl_clean"
output_column = "region"
transform = "attach_variable"
region_dict = "/path/to/region.csv"
col_to_join_on = "bpl"
col_to_add = "region"
null_filler = 99
col_type = "float"
```

## Potential Matches Universe

* Header name: `potential_matches_universe`
* Description: Limits the universe of created potential matches generated using an expression fed to a SQL query.
* Required: False
* Type: List
* Attributes: 
  * `expression` -- Type: `string`.  Required.  The expression to use to filter prepped_df_(a/b) before generating potential matches.
  
```
[[potential_matches_universe]]
# limits potential matches created to only men
expression = "sex == 1"
```

## Blocking

* Header name: `blocking`
* Description: Describes what columns to block on and how to create the blocks for the potential matches.
* Required: True
* Type: List
* Attributes:
  * `column_name` -- Type: `string`. Required. The name of the column in the existing data to block on if not exploded; The name of the newly exploded column if `explode = true`.
  * `explode` -- Type: `boolean`. Optional. If true, will attempt to "explode" the column by creating duplicate rows for each value in the column. Only works on columns that are arrays of values or when `expand_length` is set.
  * `dataset` -- Type: `string`. Optional. Must be `a` or `b` and used in conjuction with `explode`. Will only explode the column from the `a` or `b` dataset when specified.
  * `derived_from` -- Type: `string`. Used in conjunction with `explode = true`.  Specifies an input column from the existing dataset to be exploded. 
  * `expand_length` -- Type: `integer`. When `explode` is used on a column that is an integer, this can be specified to create an array with a range of integer values from (`expand_length` minus `original_value`) to (`expand_length` plus `original_value`).  For example, if the input column value for birthyr is 1870, explode is true, and the expand_length is 3, the exploded column birthyr_3 value would be the array [1867, 1868, 1869, 1870, 1871, 1872, 1873].
  * `or_group` -- Type: `string`. Optional. The "OR group" to which this
    blocking table belongs. Blocking tables that belong to the same OR group
    are joined by OR in the blocking condition instead of AND. By default each
    blocking table belongs to a different OR group. For example, suppose that
    your dataset has 3 possible birthplaces BPL1, BPL2, and BPL3 for each
    record. If you don't provide OR groups when blocking on each BPL variable,
    then you will get a blocking condition like `(a.BPL1 = b.BPL1) AND (a.BPL2
    = b.BPL2) AND (a.BPL3 = b.BPL3)`. But if you set `or_group = "BPL"` for
    each of the 3 variables, then you will get a blocking condition like this
    instead: `(a.BPL1 = b.BPL1 OR a.BPL2 = b.BPL2 OR a.BPL3 = b.BPL3)`. Note
    the parentheses around the entire OR group condition. Other OR groups would
    be connected to the BPL OR group with an AND condition.


```
[[blocking]]
column_name = "bpl"

[[blocking]]
column_name = "birthyr_3"
dataset = "a"
derived_from = "birthyr"
expand_length = 3
explode = true
```

## [Comparisons](comparisons)

* Header name: `comparisons`
* Description: A set of comparisons which filter the potential matches.
  Only record pairs which satisfy the comparisons qualify as potential matches.
  See [comparisons](comparisons) for some more information.
* Required: True
* Type: Object

There are two different forms that the comparisons table may take. It may either
be a single comparison definition, or it may be a nested comparison definition
with multiple sub-comparisons.

### Single Comparison

  * Attributes:
    * `comparison_type` -- Type: `string`. Required. The type of the comparison.
    Currently the only supported comparison type is `"threshold"`, which compares
    a comparison feature to a given value.
    * `feature_name` -- Type: `string`. Required. The comparison feature to compare
    against.
    * `threshold` -- Type: `Any`. Optional. The value to compare against.
    * `threshold_expr` -- Type: `string`. Optional. A SQL condition which defines
    the comparison on the comparison feature named by `feature_name`.

  The comparison definition must contain either `threshold` or `threshold_expr`,
  but not both. Providing `threshold = X` is equivalent to the threshold
  expression `threshold_expr >= X`.

  ```
  # Only record pairs with namefrst_jw >= 0.79 can be
  # potential matches.
  [comparisons]
  comparison_type = "threshold"
  feature_name = "namefrst_jw"
  threshold = 0.79
  ```

  ```
  # Only record pairs with flag < 0.5 can be potential matches.
  [comparisons]
  comparison_type = "threshold"
  feature_name = "flag"
  threshold_expr = "< 0.5"
  ```

### Multiple Comparisons

* Attributes:
  * `operator` -- Type: `string`. Required. The logical operator which connects
  the two sub-comparisons. May be `"AND"` or `"OR"`.
  * `comp_a` -- Type: `object`. Required. The first sub-comparison.
  * `comp_b` -- Type: `object`. Required. The second sub-comparison.

Both `comp_a` and `comp_b` are recursive comparison sections and may contain
either a single comparison or another set of sub-comparisons. Please see the
[comparisons documentation](comparisons.html#defining-multiple-comparisons) for
more details and examples.

## [Household Comparisons](comparisons)

* Header name: `hh_comparisons`
* Description: A set of comparisons which filter the household potential
  matches. `hh_comparisons` has the same configuration structure as
  `comparisons` and works in a similar way, except that it applies during the
  `hh_matching` task instead of `matching`. You can read more about comparisons
  [here](comparisons).

```
# Only household record pairs with an age difference <= 10 can be
# household potential matches.
[hh_comparisons]
comparison_type = "threshold"
feature_name = "byrdiff"
threshold_expr = "<= 10"
```

## [Comparison Features](comparison_features)

* Header name: `comparison_features`
* Description: A list of comparison features to create when comparing records. Comparisons for individual and household linking rounds are both represented here -- no need to duplicate comparisons if used in both rounds, simply specify the `column_name` in the appropriate `training` or `hh_training` section of the config.  See the [comparison features documentation page](comparison_features) for more information.
* Required: True
* Type: List
* Attributes:
  * `alias` -- Type: `string`. Optional. The name of the comparison feature column to be generated.  If not specified, the output column will default to `column_name`.
  * `column_name` -- Type: `string`. The name of the columns to compare.
  * `comparison_type` -- Type: `string`. The name of the comparison type to use.
  * `categorical` -- Type: `boolean`.  Optional.  Whether the output data should be treated as categorical data (important information used during one-hot encoding and vectorizing in the machine learning pipeline stage).
  * Other attributes may be included as well depending on `comparison_type`.  See the [comparison features page](comparison_features) for details on each comparison type.

```
[[comparison_features]]
alias = "race"
column_name = "race"
comparison_type = "equals"
categorical = true

[[comparison_features]]
alias = "namefrst_jw"
column_name = "namefrst_unstd"
comparison_type = "jaro_winkler"

[[comparison_features]]
column_name = "durmarr"
comparison_type = "new_marr"
upper_threshold = 7
```

## [Pipeline-generated Features](pipeline_features)

* Header name: `pipeline_features`
* Description: Features to be added in the model pipeline created for scoring a dataset. These features cannot be used in the `comparisons` section of the config and are for creating more robust ML models.  They typically leverage code available in the Spark Pipeline API.
* Required: False
* Type: List
* Attributes:
  * `transformer_type` -- Type: `string`. Required. See [pipeline features](pipeline_features) for more information on the available transformer types.
  * `input_column` -- Type: `string`. Either use `input_column` or `input_columns`. Used if a single input_column is needed for the pipeline feature.
  * `input_columns` -- Type: List of strings. Either use `input_column` or `input_columns`.  Used if a list of input_columns is needed for the pipeline feature.
  * `output_column` -- Type: `string`. The name of the new pipeline feature column to be generated.
  * `categorical` -- Type: `boolean`. Optional.  Whether the output data should be treated as categorical data (important information used during one-hot encoding and vectorizing in the machine learning pipeline stage).
  * Other attributes may be included as well depending on the particular pipline feature `transformer_type`.

```
[[pipeline_features]]
input_columns =  ["sex_equals", "regionf"]
output_column =  "sex_regionf_interaction"
transformer_type =  "interaction"

[[pipeline_features]]
input_column = "immyear_diff"
output_column = "immyear_caution"
transformer_type = "bucketizer"
categorical = true
splits = [-1,0,6,11,9999]
```

## Training and [Model Exploration](model_exploration)

* Header name: `training`
* Description: Specifies the training data set as well as a myriad of attributes related to training a model including the dependent variable within that dataset, the independent variables created from the `comparison_features` section, and the different models you want to use for either model exploration or scoring.  
* Required: False
* Type: Object
* Attributes:
  * `dataset` -- Type: `string`. Location of the training dataset. Must be a csv file.
  * `dependent_var` -- Type: `string`. Name of dependent variable in training dataset.
  * `independent_vars` -- Type: `list`. List of independent variables to use in the model. These must be either part of `pipeline_features` or `comparison_features`.
  * `chosen_model` -- Type: `object`. The model to train with in the `training` task and score with in the `matching` task. See the [Models](models) section for more information on model specifications.
  * `threshold` -- Type: `float`. The threshold for which to accept model probability values as true predictions.  Can be used to specify a threshold to use for all models, or can be specified within each `chosen_model` and `model_parameters` specification.
  * `threshold_ratio` -- Type: `float`. Optional. For use when `decision` is `drop_duplicate_with_threshold_ratio` . Specifies the smallest possible ratio to accept between a best and second best link for a given record.  Can be used to specify a threshold ratio (beta threshold) to use for all models.  Alternatively, unique threshold ratios can be specified in each individual `chosen_model` and `model_parameters` specification.
  * `decision` -- Type: `string`. Optional. Specifies which decision function to use to create the final prediction. The first option is `drop_duplicate_a`, which drops any links for which a record in the `a` data set has a predicted match more than one time. The second option is `drop_duplicate_with_threshold_ratio` which only takes links for which the `a` record has the highest probability out of any other potential links, and the second best link for the `a` record is less than the `threshold_ratio`.
  * `score_with_model` -- Type: `boolean`. If set to false, will skip the `apply_model` step of the matching task. Use this if you want to use the `run_all_steps` command and are just trying to generate potential links, such as for the creation of training data.
  * `scale_data` -- Type: `boolean`.  Optional. Whether to scale the data as part of the machine learning pipeline.
  * `use_training_data_features` -- Type: `boolean`. Optional. If the identifiers in the training data set are not present in your raw input data, you will need to set this to `true`, or training features will not be able to be generated, giving null column errors.  For example, if the training data set you are using has individuals from 1900 and 1910, but you are about to train a model to score the 1930-1940 potential matches, you need this to be set to `true` or it will fail, since the individual IDs are not present in the 1930 and 1940 raw input data.  If you were about to train a model to score the 1900-1910 potential matches with this same training set, it would be best to set this to `false`, so you can be sure the training features are created from scratch to match your exact current configuration settings, although if you know the features haven't changed, you could set it to `true` to save a small amount of processing time.
  * `split_by_id_a` -- Type: `boolean`.  Optional.  Used in the `model_exploration` link task.  When set to true, ensures that all potential matches for a given individual with ID_a are grouped together in the same train-test-split group. For example, if individual histid_a "A304BT" has three potential matches in the training data, one each to histid_b "B200", "C201", and "D425", all of those potential matches would either end up in the "train" split or the "test" split when evaluating the model performance.
  * `feature_importances` -- Type: `boolean`. Optional.  Whether to record
    feature importances or coefficients for the training features when training
    the ML model. Set this to true to enable training step 3.
  * `model_parameters` -- Type: `list`. Specifies models to test out in the `model_exploration` task. See the [Model Exploration](model_exploration) page for a detailed description of how this works.
  * `model_parameter_search` -- Type: `object`. Specifies which strategy hlink should
  use to generate test models for [Model Exploration](model_exploration).
  * `n_training_iterations` -- Type: `integer`. Optional; default value is 10. The number of outer folds to use during the `model_exploration` task. See [here](model_exploration.html#the-details) for more details.


```
[training]
independent_vars = ["race", "srace", "race_interacted_srace", "hits", "hits2", "exact_mult", "ncount", "ncount2", "region", "namefrst_jw","namelast_jw","namefrst_std_jw","byrdiff", "f_interacted_jw_f", "jw_f", "f_caution", "f_pres", "fbplmatch", "m_interacted_jw_m", "jw_m", "m_caution", "m_pres", "mbplmatch", "sp_interacted_jw_sp", "jw_sp", "sp_caution", "sp_pres", "mi", "fsoundex", "lsoundex", "rel", "oth", "sgen", "nbors", "county_distance", "county_distance_squared", "street_jw", "imm_interacted_immyear_caution", "immyear_diff", "imm"]
scale_data = false
dataset = "/path/to/1900_1910_training_data_20191023.csv"
dependent_var = "match"
use_training_data_features = false
split_by_id_a = true

score_with_model = true
feature_importances = true

decision = "drop_duplicate_with_threshold_ratio"

n_training_iterations = 10
model_parameter_search = {strategy = "explicit"}
model_parameters = [
  { type = "random_forest", maxDepth = 6, numTrees = 50 },
  { type = "probit", threshold = 0.5}
]

chosen_model = { type = "logistic_regression", threshold = 0.5, threshold_ratio = 1.0 }
```

## Household Training and [Model Exploration](model_exploration)

* Header name: `hh_training`
* Description: Specifies the household training data set as well as a myriad of attributes related to training a model including the dependent var within that data set, the independent vars created from the `comparison_features` section, and the different models you want to use.  
* Required: False
* Type: Object
* Attributes:
  * All of the attributes and [models](models) available in [training](#training-and-models) may also be used here.
  * `prediction_col` -- Type: `string`.  Required. The name of the column that the final prediction value is recorded in the individual linking round scoring step.
  * `hh_col` -- Type: `string`. Required. The name of the column with the household identifier.

```
[hh_training]
prediction_col = "prediction"
hh_col = "serialp"

independent_vars = ["namelast_jw","namefrst_jw","namefrst_std_jw", "jw_max_a", "jw_max_b", "f1_match", "f2_match", "byrdifcat", "racematch", "imm", "bplmatch", "imm_interacted_bplmatch", "sexmatch", "mardurmatch", "relatetype", "relatematch", "relatetype_interacted_relatematch"]

scale_data = false
dataset = "/path/to/hh_training_data_1900_1910.csv"
dependent_var = "match"
use_training_data_features = false
split_by_id_a = true
score_with_model = true
feature_importances = true
decision = "drop_duplicate_with_threshold_ratio"

model_parameter_search = {strategy = "grid"}
n_training_iterations = 10
model_parameters = [
    { type = "logistic_regression", threshold = [0.5], threshold_ratio = [1.1]},
    { type = "random_forest", maxDepth = [5, 6, 7], numTrees = [50, 75, 100], threshold = [0.5], threshold_ratio = [1.0, 1.1, 1.2]}
]

chosen_model = { type = "logistic_regression", threshold = 0.5, threshold_ratio = 1.0 }
```

## Household Matching

* Header name: `hh_matching`
* Description: Settings for the `hh_matching` task.
* Required: False
* Type: Object
* Attributes:
  * `records_to_match` -- Type: `string`. Optional; default "unmatched_only". This option
  controls which records are eligible for linking in the `hh_matching` task. The
  default value of "unmatched_only" means that only records which are *not* linked
  by the `matching` task are eligible for linking in `hh_matching`. You can set this
  option to "all" to instead mark all records as eligible for matching in `hh_matching`,
  even if they are already matched by the `matching` task. Note that setting this option
  to "all" may lead to conflicts between the output of the two tasks, as there is
  no guarantee that a person matched in `matching` will receive the same link in `hh_matching`.
  *Added in version 4.1.0.*

```toml
[hh_matching]
records_to_match = "all"
```
