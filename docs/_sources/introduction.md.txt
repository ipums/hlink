# Introduction

## Overview

hlink is a Python library and command-line tool that links records between two datasets. This linking
process identifies records in the datasets that share some characteristics and may correspond to the
same real-world person or object. hlink can use either deterministic rules or probabilistic machine
learning algorithms to match records between the two datasets. At [IPUMS](https://ipums.org), the primary
use case for hlink has been to link United States censuses, which have a hierarchical structure of
people nested within households. However, hlink can also be used to link generic datasets. Some of its
functionality is tailored to the household-person hierarchical structure and may be ignored for datasets
with a different structure.

hlink provides functionality for the following common linking tasks. It is highly configurable via an input
configuration file written in the [TOML](https://toml.io) configuration language. Each linking task is further
broken down into several smaller steps which can be run in sequence with hlink's
[`LinkRun` API](running_the_program.html#using-hlink-as-a-library) or from the command-line with the provided
[`hlink` script](running_the_program.html#interactive-mode).

1. [Preprocessing](link_tasks.html#preprocessing):
Preprocess each dataset to clean and transform it in preparation for linking.

2. [Training and Household Training](link_tasks.html#training-and-household-training):
Train machine learning models on a set of features and compare results between models.

3. [Matching](link_tasks.html#matching) and [Household Matching](link_tasks.html#household-matching):
Match two datasets using a model created in training or with deterministic rules specified in the
configuration file.

In addition, hlink provides functionality for several research and development tasks. These tasks are useful for
experimenting with configuration and machine learning algorithms to better understand and tune the record links
that hlink outputs.

1. [Model Exploration and Household Model Exploration](link_tasks.html#model-exploration-and-household-model-exploration):
Experiment with machine learning models and their hyper-parameters to evaluate model performance and
select a model to be used in the production run. Optionally generate reports of suspected false
positives and false negatives in the specified training data set.

2. [Reporting](link_tasks.html#reporting):
Generate reports on the linked data to better understand the relationship between linked records and the
datasets as a whole.
