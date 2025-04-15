# Changelog

The format of this changelog is based on [Keep A Changelog][keep-a-changelog].
Hlink adheres to semantic versioning as much as possible.

## v4.1.0 (2025-04-15)

### Added

* Added a new configuration option `hh_matching.records_to_match` that controls
  which records are eligible for re-matching in the `hh_matching` task. You can
  find the documentation for this option in the new [Household Matching][household-matching-docs]
  section on the Configuration page. [PR #201][pr201]

* Added the `hh_training.feature_importances` configuration option for saving model
  feature importances or coefficients as step 3 of
  ["Household Training"][household-training-docs] when set to true. [PR #202][pr202]

### Fixed

* Fixed a bug in the calculation of predicted matches. Previously, if there was
a second-best probability, hlink applied the threshold ratio only if
the first and second-best probabilities were both at least at the alpha threshold. Now it always
applies the threshold ratio when the best probability is at least at
the alpha threshold and there is a second-best probability. [PR #200][pr200]

## v4.0.0 (2025-04-07)

### Added

* Added support for randomized parameter search to model exploration. [PR #168][pr168]
* Created an `hlink.linking.core.model_metrics` module with functions for computing
metrics on model confusion matrices. Added the F-measure model metric to model
exploration. [PR #180][pr180]
* Added this changelog! [PR #189][pr189]

### Changed

* Overhauled the model exploration task to use a nested cross-validation approach.
[PR #169][pr169]
* Changed `hlink.linking.core.classifier` functions to not interact with `threshold`
and `threshold_ratio`. Please ensure that the parameter dictionaries passed to these
functions only contain parameters for the chosen model. [PR #175][pr175]
* Simplified the parameters required for `hlink.linking.core.threshold.predict_using_thresholds`.
Instead of passing the entire `training` configuration section to this function,
you now need only pass `training.decision`. [PR #175][pr175]
* Added a new required `checkpoint_dir` argument to `SparkConnection`, which lets hlink set
different directories for the tmp and checkpoint directories. [PR #182][pr182]
* Swapped to using `tomli` as the default TOML parser. This should fix several issues
with how hlink parses TOML files. `load_conf_file()` provides the `use_legacy_toml_parser`
argument for backwards compatibility if necessary. [PR #185][pr185]

### Deprecated

* Deprecated the `training.param_grid` attribute in favor of the new, more flexible
`training.model_parameter_search` table. This is part of supporting the new randomized
parameter search. [PR #168][pr168]

### Removed

* Removed functionality for outputting "suspicious" training data from model exploration.
We determined that this is out of the scope of model exploration step 2. This change
greatly simplifies the model exploration code. [PR #178][pr178]
* Removed the deprecated `hlink.linking.transformers.interaction_transformer` module.
This module was deprecated in v3.5.0. Please use
[`pyspark.ml.feature.Interaction`][pyspark-interaction-docs] instead. [PR #184][pr184]
* Removed some alternate configuration syntax which has been deprecated since
v3.0.0. [PR #184][pr184]
* Removed `hlink.scripts.main.load_conf` in favor of a much simpler approach to
finding the configuration file and configuring spark. Please call
`hlink.configs.load_config.load_conf_file` directly instead. `load_conf_file` now
returns both the path to the configuration file and its contents as a mapping. [PR #182][pr182]

## v3.8.0 (2024-12-04)

### Added

* Added optional support for the XGBoost and LightGBM gradient boosting
machine learning libraries. You can find documentation on how to use these libraries
[here][gradient-descent-ml-docs]. [PR #165][pr165]
* Added a new `hlink.linking.transformers.RenameVectorAttributes` transformer which
can rename the attributes or "slots" of Spark vector columns. [PR #165][pr165]

### Fixed

* Corrected misleading documentation for comparisons, which are not the same thing
as comparison features. You can find the new documentation [here][comparison-docs]. [PR #159][pr159]
* Corrected the documentation for substitution files, which had the meaning of the
columns backwards. [PR #166][pr166]

## v3.7.0 (2024-10-10)

### Added

* Added an optional argument to `SparkConnection` to allow setting a custom Spark
app name. The default is still to set the app name to "linking". [PR #156][pr156]

### Changed

* Improved model exploration step 2's terminal output, logging, and documentation
to make the step easier to work with. [PR #155][pr155]

### Fixed

* Updated all modules to log to module-level loggers instead of the root logger. This gives
users of the library more control over filtering logs from hlink. [PR #152][pr152]


## v3.6.1 (2024-08-14)

### Fixed

* Fixed a crash in matching step 0 triggered when there were multiple exploded columns
in the blocking section. Multiple exploded columns are now supported. [PR #143][pr143]

## v3.6.0 (2024-06-18)

### Added

* Added OR conditions in blocking. This new feature supports connecting some or
all blocking conditions together with ORs instead of ANDs. Note that using many ORs in blocking
may have negative performance implications for large datasets since it increases
the size of the blocks and makes each block more difficult to compute. You can find
documentation on OR blocking conditions under the `or_group` bullet point [here][or-groups-docs].
[PR #138][pr138]

## v3.5.5 (2024-05-31)

### Added

* Added support for a variable number of columns in the array feature selection
transform, instead of forcing it to use exactly 2 columns. [PR #135][pr135]

## v3.5.4 (2024-02-20)

### Added

* Documented the `concat_two_cols` column mappings transform. You can see the
documentation [here][concat-two-cols-docs]. [PR #126][pr126]
* Documented column mapping overrides, which can let you read two columns with
different names in the input files into a single hlink column. The documentation for
this feature is [here][column-mapping-overrides-docs]. [PR #129][pr129]

### Fixed

* Fixed a bug where config validation checks did not respect column mapping overrides.
[PR #131][pr131]

## v3.5.3 (2023-11-02)

### Added

* Added config validation checks for duplicate comparison features, feature selections,
and column mappings. [PR #113][pr113]
* Added support for Python 3.12. [PR #119][pr119]
* Put the config file name in the script prompt. [PR #123][pr123]

### Fixed

* Reverted to keeping invalid categories in training data instead of erroring out.
This case actually does occasionally happen, and so we would rather not error out
on it. This reverts a change made in [PR #109][pr109], released in v3.5.2. [PR #121][pr121]

## v3.5.2 (2023-10-26)

### Changed

* Made some minor updates to the format of training step 3's output. There are now
3 columns: `feature_name`, `category`, and `coefficient_or_importance`. Feature
names are not suffixed with the category value anymore. [PR #112][pr112]
* BUG reverted in v3.5.3: Started erroring out on invalid categories in training
data instead of creating a new category for them. [PR #109][pr109]

### Fixed

* Fixed a bug with categorical features in training step 3. Each categorical feature
was getting a single coefficient when each *category* should get its own coefficient
instead. [PR #104][pr104], [PR #107][pr107]


## v3.5.1 (2023-10-23)

### Added

* Made a new training step 3 to replace model exploration step 3, which was buggy.
Training step 3 saves model feature importances or coefficients when `training.feature_importances`
is set to true. [PR #101][pr101]

### Removed

* Removed the buggy implementation of model exploration step 3. Training step 3 replaces
this. [PR #101][pr101]

## v3.5.0 (2023-10-16)

### Added

* Added support for Python 3.11. [PR #94][pr94]
* Created a new `multi_jaro_winkler_search` comparison feature. This is a complex
comparison feature which supports conditional Jaro-Winkler comparisons between
lists of columns with similar names. You can read more in the documentation [here][multi-jaro-winkler-search-docs].
[PR #99][pr99]

### Changed

* Upgraded from PySpark 3.3 to 3.5. [PR #94][pr94]

### Deprecated

* Deprecated the `hlink.linking.transformers.interaction_transformer` module.
Please use PySpark 3's [`pyspark.ml.feature.Interaction`][pyspark-interaction-docs]
class instead. Hlink's `interaction_transformer` module is scheduled for removal
in version 4. [PR #97][pr97]

### Fixed

* Fixed a bug where the hlink script's autocomplete feature sometimes did not work
correctly. [PR #96][pr96]

## v3.4.0 (2023-08-09)

### Added

* Created a new `convert_ints_to_longs` configuration setting for working with CSV
files. Documentation for this setting is available [here][ints-to-longs-docs]. [PR #87][pr87]
* Improved the link tasks documentation by adding more detail. This page is available
[here][link-tasks-docs]. [PR #86][pr86]

### Removed

* Dropped the `comment` column from the script's `desc` command. This column was
always full of nulls and cluttered up the screen. [PR #88][pr88]

## v3.3.1 (2023-06-02)

### Changed

* Updated documentation for column mapping transforms. [PR #77][pr77]
* Updated documentation for the `present_both_years` and `neither_are_null` comparison
types, clarifying how they are different. [PR #79][pr79]

### Fixed

* Fixed a bug where comparison features were marked as categorical whenever the
`categorical` key was present, even if it was set to false. [PR #82][pr82]

## v3.3.0 (2022-12-13)

### Added

* Added logging for user input to the script. This is extremely helpful for diagnosing
errors. [PR #64][pr64]
* Added and improved documentation for several comparison types. [PR #47][pr47]

### Changed

* Started writing to a unique log file for each script run. [PR #55][pr55]
* Updated and improved the tutorial in examples/tutorial. [PR #63][pr63]
* Changed to pyproject.toml instead of setup.py and setup.cfg. [PR #71][pr71]

### Fixed

* Fixed a bug which caused Jaro-Winkler scores to be 1.0 for two empty strings. The
scores are now 0.0 on two empty strings. [PR #59][pr59]

## v3.2.7 (2022-09-14)

### Added

* Added a configuration validation that checks that both data sources contain the id column. [PR #13][pr13]
* Added driver memory options to `SparkConnection`. [PR #40][pr40]

### Changed

* Upgraded from PySpark 3.2 to 3.3. [PR #11][pr11]
* Capped the number of partitions requested at 10,000. [PR #40][pr40]

### Fixed

* Fixed a bug where `feature_selections` was always required in the config file.
It now defaults to an empty list as intended. [PR #15][pr15]
* Fixed a bug where an error message in `conf_validations` was not formatted correctly. [PR #13][pr13]

## v3.2.6 (2022-07-18)

### Added

* Made hlink installable with `pip` via PyPI.org.

## v3.2.1 (2022-05-24)

### Added

* Improved logging during startup and for the `LinkTask.run_all_steps()` method.
[PR #7][pr7]

### Changed

* Added code to adjust the number of Spark partitions based on the size of the input
datasets for some link steps. This should help these steps scale better with large
datasets. [PR #10][pr10]

### Fixed

* Fixed a bug where model exploration's step 3 would run into a `TypeError` due to
trying to manually build up a file path. [PR #8][pr8]

## v3.2.0 (2022-05-16)

### Changed

* Upgraded from Python 3.6 to 3.10. [PR #5][pr5]
* Upgraded from PySpark 2 to PySpark 3. [PR #5][pr5]
* Upgraded from Java 8 to Java 11. [PR #5][pr5]
* Upgraded from Scala 2.11 to Scala 2.12. [PR #5][pr5]
* Upgraded from Scala Commons Text 1.4 to 1.9. This includes some bug fixes which
may slightly change Jaro-Winkler scores. [PR #5][pr5]

## v3.1.0 (2022-05-04)

### Added

* Started exporting true positive and true negative data along with false positive
and false negative data in model exploration. [PR #1][pr1]

### Fixed

* Fixed a bug where `exact_all_mult` was not handled correctly in config validation.
[PR #2][pr2]

## v3.0.0 (2022-04-27)

### Added

* This is the initial open-source version of hlink.


[pr1]: https://github.com/ipums/hlink/pull/1
[pr2]: https://github.com/ipums/hlink/pull/2
[pr5]: https://github.com/ipums/hlink/pull/5
[pr7]: https://github.com/ipums/hlink/pull/7
[pr8]: https://github.com/ipums/hlink/pull/8
[pr10]: https://github.com/ipums/hlink/pull/10
[pr11]: https://github.com/ipums/hlink/pull/11
[pr13]: https://github.com/ipums/hlink/pull/13
[pr15]: https://github.com/ipums/hlink/pull/15
[pr40]: https://github.com/ipums/hlink/pull/40
[pr47]: https://github.com/ipums/hlink/pull/47
[pr55]: https://github.com/ipums/hlink/pull/55
[pr59]: https://github.com/ipums/hlink/pull/59
[pr63]: https://github.com/ipums/hlink/pull/63
[pr64]: https://github.com/ipums/hlink/pull/64
[pr71]: https://github.com/ipums/hlink/pull/71
[pr77]: https://github.com/ipums/hlink/pull/77
[pr79]: https://github.com/ipums/hlink/pull/79
[pr82]: https://github.com/ipums/hlink/pull/82
[pr86]: https://github.com/ipums/hlink/pull/86
[pr87]: https://github.com/ipums/hlink/pull/87
[pr88]: https://github.com/ipums/hlink/pull/88
[pr94]: https://github.com/ipums/hlink/pull/94
[pr96]: https://github.com/ipums/hlink/pull/96
[pr97]: https://github.com/ipums/hlink/pull/97
[pr99]: https://github.com/ipums/hlink/pull/99
[pr101]: https://github.com/ipums/hlink/pull/101
[pr104]: https://github.com/ipums/hlink/pull/104
[pr107]: https://github.com/ipums/hlink/pull/107
[pr109]: https://github.com/ipums/hlink/pull/109
[pr112]: https://github.com/ipums/hlink/pull/112
[pr113]: https://github.com/ipums/hlink/pull/113
[pr119]: https://github.com/ipums/hlink/pull/119
[pr121]: https://github.com/ipums/hlink/pull/121
[pr123]: https://github.com/ipums/hlink/pull/123
[pr126]: https://github.com/ipums/hlink/pull/126
[pr129]: https://github.com/ipums/hlink/pull/129
[pr131]: https://github.com/ipums/hlink/pull/131
[pr135]: https://github.com/ipums/hlink/pull/135
[pr138]: https://github.com/ipums/hlink/pull/138
[pr143]: https://github.com/ipums/hlink/pull/143
[pr152]: https://github.com/ipums/hlink/pull/152
[pr155]: https://github.com/ipums/hlink/pull/155
[pr156]: https://github.com/ipums/hlink/pull/156
[pr159]: https://github.com/ipums/hlink/pull/159
[pr165]: https://github.com/ipums/hlink/pull/165
[pr166]: https://github.com/ipums/hlink/pull/166
[pr168]: https://github.com/ipums/hlink/pull/168
[pr169]: https://github.com/ipums/hlink/pull/169
[pr175]: https://github.com/ipums/hlink/pull/175
[pr178]: https://github.com/ipums/hlink/pull/178
[pr180]: https://github.com/ipums/hlink/pull/180
[pr182]: https://github.com/ipums/hlink/pull/182
[pr184]: https://github.com/ipums/hlink/pull/184
[pr185]: https://github.com/ipums/hlink/pull/185
[pr189]: https://github.com/ipums/hlink/pull/189
[pr200]: https://github.com/ipums/hlink/pull/200
[pr201]: https://github.com/ipums/hlink/pull/201
[pr202]: https://github.com/ipums/hlink/pull/202

[household-matching-docs]: config.html#household-matching
[household-training-docs]: config.html#household-training-and-model-exploration
[ints-to-longs-docs]: config.html#data-sources
[link-tasks-docs]: link_tasks
[pyspark-interaction-docs]: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Interaction.html
[multi-jaro-winkler-search-docs]: comparison_features.html#multi-jaro-winkler-search
[concat-two-cols-docs]: column_mappings.html#concat-two-cols
[column-mapping-overrides-docs]: column_mappings.html#advanced-usage
[or-groups-docs]: config.html#blocking
[gradient-descent-ml-docs]: models
[comparison-docs]: comparisons
[keep-a-changelog]: https://keepachangelog.com/en/1.0.0/
