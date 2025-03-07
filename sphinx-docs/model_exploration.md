# Model Exploration

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

For example, this `training` section generates 90 combinations of model
parameters for testing. The first has a `threshold` of 0.8, `maxDepth` of 1, and
`numTrees` of 20; the second has a `threshold` of 0.8, `maxDepth` of 1, and `numTrees`
of 30; and so on.

```toml
[training.model_parameter_search]
strategy = "grid"

[[training.model_parameters]]
type = "random_forest"
threshold = [0.8, 0.9, 0.95]
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
[training.model_parameter_search]
strategy = "grid"
```

and

```toml
[training]
param_grid = false
```

is equivalent to

```toml
[training.model_parameter_search]
strategy = "explicit"
```
