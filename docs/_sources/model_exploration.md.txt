# Model Exploration

## Overview

The model exploration task provides a way to try out different types of machine
learning models and sets of parameters to those models. It tests those models
on splits of the training data and outputs information on the performance of
the models. The purpose of model exploration is to help you choose a model that
performs well without having to test each model individually on the entire
input datasets. If you're interested in the exact workings of the model exploration
algorithm, see the [Details](#the-details) section below.

Model exploration uses several configuration attributes listed in the `training`
section because it is closely related to `training`.

## Searching for Model Parameters

Part of the process of model exploration is searching for model parameters which
give good results on the training data. Hlink supports three strategies for model
parameter searches, controlled by the `training.model_parameter_search` table.

### Explicit Search (`strategy = "explicit"`)

An explicit model parameter search lists out all of the parameter combinations
to be tested. Each element of the `training.model_parameters` list becomes one
set of parameters to evaluate. This is the simplest search strategy and is hlink's
default behavior.

This example `training` section uses an explicit search over two sets of model parameters.
Model exploration will train two random forest models. The first will have a
`maxDepth` of 3 and `numTrees` of 50, and the second will have a `maxDepth` of 3
and `numTrees` of 20.

```toml
[training.model_parameter_search]
strategy = "explicit"

[[training.model_parameters]]
type = "random_forest"
maxDepth = 3
numTrees = 50

[[training.model_parameters]]
type = "random_forest"
maxDepth = 3
numTrees = 20
```

### Grid Search (`strategy = "grid"`)

A grid search takes multiple values for each model parameter and generates one
model for each possible combination of the given parameters. This is often much more
compact than writing out all of the possible combinations in an explicit search.

For example, this `training` section generates 30 combinations of model
parameters for testing. The first has a `maxDepth` of 1 and `numTrees` of 20,
the second has a `maxDepth` of 1 and `numTrees` of 30, and so on.

```toml
[training.model_parameter_search]
strategy = "grid"

[[training.model_parameters]]
type = "random_forest"
maxDepth = [1, 2, 3, 5, 10]
numTrees = [20, 30, 40, 50, 60, 70]
```

Although grid search is more compact than explicitly listing out all of the model
parameters, it can be quite time-consuming to check every possible combination of
model parameters. Randomized search, described below, can be a more efficient way
to evaluate models with large numbers of parameters or large parameter ranges.


### Randomized Search (`strategy = "randomized"`)

*Added in version 4.0.0.*

A randomized parameter search generates model parameter settings by sampling each
parameter from a distribution or set. The number of samples is an additional parameter
to the strategy. This separates the size of the search space from the number of samples
taken, making a randomized search more flexible than a grid search. The downside of
this is that, unlike a grid search, a randomized search does not necessarily test
all of the possible values given for each parameter. It is necessarily non-exhaustive.

In a randomized search, each model parameter may take one of 3 forms:

* A list, which is a set of values to sample from with replacement. Each value has an equal chance
of being chosen for each sample.

```toml
[[training.model_parameters]]
type = "random_forest"
numTrees = [20, 30, 40]
```

* A single value, which "pins" the model parameter to always be that value. This
is syntactic sugar for sampling from a list with one element.

```toml
[[training.model_parameters]]
type = "random_forest"
# numTrees will always be 30.
# This is equivalent to numTrees = [30].
numTrees = 30
```

* A table defining a distribution from which to sample the parameter. The available
distributions are `"randint"`, to choose a random integer from a range, `"uniform"`,
to choose a random floating-point number from a range, and `"normal"`, to choose
a floating-point number from a normal distribution with a given mean and standard
deviation.

For example, this `training` section generates 20 model parameter combinations
for testing, using a randomized search. Each of the three given model parameters
uses a different type of distribution.

```toml
[training.model_parameter_search]
strategy = "randomized"
num_samples = 20

[[training.model_parameters]]
type = "random_forest"
numTrees = {distribution = "randint", low = 20, high = 70}
minInfoGain = {distribution = "uniform", low = 0.0, high = 0.3}
subsamplingRate = {distribution = "normal", mean = 1.0, standard_deviation = 0.2}
```

### The `training.param_grid` Attribute

As of version 4.0.0, the `training.param_grid` attribute is deprecated. Please use
`training.model_parameter_search` instead, as it is more flexible and supports additional
parameter search strategies. Prior to version 4.0.0, you will need to use `training.param_grid`.

`param_grid` has a direct mapping to `model_parameter_search`.

```toml
[training]
param_grid = true
```

is equivalent to

```toml
[training]
model_parameter_search = {strategy = "grid"}
```

and

```toml
[training]
param_grid = false
```

is equivalent to

```toml
[training]
model_parameter_search = {strategy = "explicit"}
```

### Types and Thresholds


There are 3 attributes which are hlink-specific and are not passed through as model parameters.
* `type` is the name of the model type.
* `threshold` and `threshold_ratio` control how hlink classifies potential matches
based on the probabilistic output of the models. They may each be either a float
or a list of floats, and hlink will always use a grid strategy to generate the
set of test combinations for these parameters.

For more details, please see the [Models](models) page and the [Details](#the-details)
section below.

## The Details

The current model exploration implementation uses a technique called nested cross-validation to evaluate each model which the search strategy generates. The algorithm follows this basic outline.

Let `N` be the value of `training.n_training_iterations`.
Let `J` be 3. (Currently `J` is hard-coded).

1. Split the prepared training data into `N` **outer folds**. This forms a partition of the training data into `N` distinct pieces, each of roughly equal size.
2. Choose the first **outer fold**.
3. Combine the `N - 1` other **outer folds** into the set of outer training data.
4. Split the outer training data into `J` **inner folds**. This forms a partition of the training data into `J` distinct pieces, each of roughly equal size.
5. Choose the first **inner fold**.
6. Combine the `J - 1` other **inner folds** into the test of inner training data.
7. Train, test, and score all of the models using the inner training data and the first **inner fold** as the test data.
8. Repeat steps 5 - 7 for each other **inner fold**.
9. After finishing all of the **inner folds**, choose the single model with the best aggregate score over those folds.
10. For each setting of `threshold` and `threshold_ratio`, train the best model on the outer training data and the chosen **outer fold**. Collect metrics on the performance of the model based on its confusion matrix.
11. Repeat steps 2-10 for each other **outer fold**.
12. Report on all of the metrics gathered for the best-scoring models.
