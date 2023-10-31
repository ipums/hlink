# Installation

## Requirements

Make sure that you have each of these installed on your system before running hlink.

- Python 3.10, 3.11, or 3.12
- Java 8 or greater, used for integration with [Apache Spark](https://spark.apache.org) via the pyspark
package

## Installing from PyPI 

hlink is available on the Python Package Index at [pypi.org](https://pypi.org) as the hlink package.
The latest version can be installed with pip by running `pip install hlink`.

## Installing from source

The easiest way to install hlink is through PyPI (see the instructions above). But hlink can also
be installed from source. To install hlink from source, first clone the GitHub repository. Then
in the root project directory, run `pip install .`.

To install hlink for development work, instead run `pip install -e .[dev]`. This will install
additional development dependencies and install hlink in editable mode so that any changes made
to the source code are automatically built by the Python packaging tools.
