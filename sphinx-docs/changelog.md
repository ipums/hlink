# Changelog

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

* Fixed a bug where "exact_all_mult" was not handled correctly in config validation.
[PR #2][pr2]


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
