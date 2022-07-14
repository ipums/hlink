[![HLink Docker CI](https://github.com/ipums/hlink/actions/workflows/docker-build.yml/badge.svg)](https://github.com/ipums/hlink/actions/workflows/docker-build.yml)

# hlink: Historical Record Linkage

A paper on the creation and applications of this program can be found at <https://www.tandfonline.com/doi/full/10.1080/01615440.2021.1985027>.

## Installation

Java 8 or [Java 11](https://openjdk.org/install/) is requried for the PySpark integration. 

You can install the python package from pip:
`pip install hlink`

## Docs

The documentation site can be found at [hlink.docs.ipums.org](https://hlink.docs.ipums.org).
This includes information about installation and setting up your configuration files.

An example script and configuration file can be found in the `examples` directory.

## Overview

Hlink is designed to link two datasets. The primary use case was for linking demographics in the Household -> Person hierarchical structure, however it can be used to link generic datasets as well by skipping household linking tasks. It allows for probabilistic and deterministic record linkage, and provides functionality for the following tasks:

1. Preprocessing: Preprocess each dataset to clean and transform it in preparation for linking.
2. Training: Train machine learning models on a set of features and compare results between models.
3. Matching: Match two datasets using a model created in training or with deterministic rules.
4. Household Training: Train machine learning models on a set of features for households and compare results between models.
5. Household Matching: Match households between two datasets.

In addition, it also provides functionality for the following research/development tasks:
1. Model Exploration and Household Model Exploration: Use a matrix of models and hyper-parameters to evaluate model performance and select a model to be used in the production run.  Also generates reports of suspected false positives and false negatives in the specified training data set if appropriate config flag is set.
2. Reporting: Generate reports on the linked data.
