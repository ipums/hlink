# Introduction

## Overview

Hlink is designed to link two datasets. The primary use case was for linking demographics in the Household -> Person hierarchical structure, however it can be used to link generic datasets as well by skipping the Household linking task. It allows for probabilistic and deterministic record linkage, and provides functionality for the following tasks:

1. [Preprocessing](link_tasks.html#preprocessing): Preprocess each dataset to clean and transform it in preparation for linking.
2. [Training](link_tasks.html#training-and-household-training): Train machine learning models on a set of features and compare results between models.
3. [Matching](link_tasks.html#matching): Match two datasets using a model created in training or with deterministic rules.
4. [Household Training](link_tasks.html#training-and-household-training): Train machine learning models on a set of features for households and compare results between models.
5. [Household Matching](link_tasks.html#household-matching): Match households between two datasets.

In addition, it also provides functionality for the following research/development tasks:
1. [Model Exploration and Household Model Exploration](link_tasks.html#model-exploration-and-household-model-exploration): Use a matrix of models and hyper-parameters to evaluate model performance and select a model to be used in the production run.  Also generates reports of suspected false positives and false negatives in the specified training data set if appropriate config flag is set.
2. [Reporting](link_tasks.html#reporting): Generate reports on the linked data.
