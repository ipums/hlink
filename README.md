[![HLink Docker CI](https://github.com/ipums/hlink/actions/workflows/docker-build.yml/badge.svg)](https://github.com/ipums/hlink/actions/workflows/docker-build.yml)

# hlink: hierarchical record linkage at scale

hlink is a Python package that provides a flexible, configuration-driven solution to probabilistic record linking at scale. It provides a high-level API for python as well as a standalone command line interface for running linking jobs with little to no programming. hlink supports the linking process from beginning to end, including preprocessing, filtering, training, model exploration, blocking, feature generation and scoring.

It is used at [IPUMS](https://www.ipums.org/) to link U.S. historical census data, but can be applied to any record linkage job. 
A paper on the creation and applications of this program on historical census data can be found at <https://www.tandfonline.com/doi/full/10.1080/01615440.2021.1985027>.

### Suggested Citation
Wellington, J., R. Harper, and K.J. Thompson. 2022. "hlink." https://github.com/ipums/hlink: Institute for Social Research and Data Innovation, University of Minnesota.

## Installation

hlink requires

- Python 3.10, 3.11, or 3.12
- Java 8 or greater for integration with PySpark

You can install the newest version of the Python package directly from PyPI with pip:
```
pip install hlink
```

We do our best to make hlink compatible with Python 3.10-3.12. If you have a
problem using hlink on one of these versions of Python, please open an issue
through GitHub. Versions of Python older than 3.10 are not supported.

Note that PySpark 3.5 does not yet officially support Python 3.12. If you
encounter PySpark-related import errors while running hlink on Python 3.12, try

- Installing the setuptools package. The distutils package was deleted from the
  standard library in Python 3.12, but some versions of PySpark still import
  it. The setuptools package provides a hacky stand-in distutils library which
  should fix some import errors in PySpark. We install setuptools in our
  development and test dependencies so that our tests work on Python 3.12.

- Downgrading Python to 3.10 or 3.11. PySpark officially supports these
  versions of Python. So you should have better chances getting PySpark to work
  well on Python 3.10 or 3.11.

### Additional Machine Learning Algorithms

hlink has optional support for two additional machine learning algorithms,
[XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) and
[LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html). Both of these
algorithms are highly performant gradient boosting libraries, each with its own
characteristics. These algorithms are not implemented directly in Spark, so
they require some additional dependencies. To install the required Python
dependencies, run

```
pip install hlink[xgboost]
```

for XGBoost or

```
pip install hlink[lightgbm]
```

for LightGBM. If you would like to install both at once, you can run

```
pip install hlink[xgboost,lightgbm]
```

to get the Python dependencies for both. Both XGBoost and LightGBM also require
libomp, which will need to be installed separately if you don't already have it.

After installing the dependencies for one or both of these algorithms, you can
use them as model types in training and model exploration. You can read more
about these models in the hlink documentation [here](https://hlink.docs.ipums.org/models.html).

*Note: The XGBoost-PySpark integration provided by the xgboost Python package is
currently unstable. So the hlink xgboost support is experimental and may change
in the future.*

## Docs

The documentation site can be found at [hlink.docs.ipums.org](https://hlink.docs.ipums.org).
This includes information about installation and setting up your configuration files.

An example script and configuration file can be found in the `examples` directory.

## Quick Start

The main class in the library is LinkRun, which represents a complete linking job. It provides access to each of the link tasks and their steps. Here is an example script that uses LinkRun to do some linking.

```python
from hlink.linking.link_run import LinkRun
from hlink.spark.factory import SparkFactory
from hlink.configs.load_config import load_conf_file

# First we create a SparkSession with all default configuration settings.
factory = SparkFactory()
spark = factory.create()

# Now let's load in our config file. See the example config below.
# This config file is in toml format, but we also allow json format.
# Alternatively you can create a python dictionary directly with the same
# keys and values as is in the config.
config = load_conf_file("./my_conf.toml")

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

An example configuration file:

```toml
### hlink config file ###
# This is a sample config file for the hlink program in toml format.

# The name of the unique identifier in the datasets
id_column = "id" 

### INPUT ###

# The input datasets
[datasource_a]
alias = "a"
file = "data/A.csv"

[datasource_b]
alias = "b"
file = "data/B.csv"

### PREPROCESSING ###

# The columns to extract from the sources and the preprocessing to be done on them.
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


### BLOCKING ###

# Blocking parameters
# Here we are blocking on sex and +/- age. 
# This means that no comparisons will be done on records
# where the SEX fields don't match exactly and the AGE 
# fields are not within a distance of 2.
[[blocking]]
column_name = "SEX"

[[blocking]]
column_name = "AGE_2"
dataset = "a"
derived_from = "AGE"
expand_length = 2
explode = true

### COMPARISON FEATURES ###

# Here we detail the comparison features that are
# created between the two records. In this case
# we are comparing first and last names using 
# the jaro-winkler metric.

[[comparison_features]]
alias = "NAMEFRST_JW"
column_name = "NAMEFRST"
comparison_type = "jaro_winkler"

[[comparison_features]]
alias = "NAMELAST_JW"
column_name = "NAMELAST"
comparison_type = "jaro_winkler"

# Here we detail the thresholds at which we would
# like to keep potential matches. In this case
# we will keep only matches where the first name
# jaro winkler score is greater than 0.79 and
# the last name jaro winkler score is greater than 0.84.

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
