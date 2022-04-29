# hlink: historical record linkage

A working paper on the creation and applications of this program can be found at <https://doi.org/10.18128/ipums2020-03>. A publication on the same topic is forthcoming.

## Docs

Documentation site can be found [here](https://pages.github.umn.edu/mpc/hlink).    
This includes information about installation and setting up your configuration files.

An example script and config file can be found in the `examples` directory.

## Overview

Hlink is designed to link two datasets. The primary use case was for linking demographics in the Household -> Person hierarchical structure, however it can be used to link generic datasets as well by skipping the Houehold linking task. It allows for probabilistic and deterministic record linkage, and provides functionality for the following tasks:

1. Preprocessing: preprocess each dataset to clean and or transform it in preperation for linking.
2. Training: train ML models on a set of features and compare results between models.
3. Matching: match two datasets using a model created in training or with deterministic rules.
4. Household linking: using the results from an individual linking process, compare household members of linked records to generate additional links.
5. Reporting: generate summarized information on linked datasets.
6. Model Exploration: compare various models and hyperparameter matrices to choose production model specs.

