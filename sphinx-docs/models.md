# Models

These are the machine learning models available for use in the model evaluation
and training tasks and in their household counterparts.

There are a few attributes available for all models.

* `type` -- Type: `string`. The name of the model type. The available model
  types are listed below.
* `threshold` -- Type: `float`.  The "alpha threshold". This is the probability
  score required for a potential match to be labeled a match. `0 ≤ threshold ≤
  1`.
* `threshold_ratio` -- Type: `float`. The threshold ratio or "beta threshold".
  This applies to records which have multiple potential matches when
  `training.decision` is set to `"drop_duplicate_with_threshold_ratio"`. For
  each record, only potential matches which have the highest probability, have
  a probability of at least `threshold`, *and* whose probabilities are at least
  `threshold_ratio` times larger than the second-highest probability are
  matches. This is sometimes called the "de-duplication distance ratio". `1 ≤
  threshold_ratio < ∞`.

In addition, any model parameters documented in a model type's Spark
documentation can be passed as parameters to the model through hlink's
`training.chosen_model` and `training.model_exploration` configuration
sections.

Here is an example `training.chosen_model` configuration. The `type`,
`threshold`, and `threshold_ratio` attributes are hlink specific. `maxDepth` is
a parameter to the random forest model which hlink passes through to the
underlying Spark classifier.

```toml
[training.chosen_model]
type = "random_forest"
threshold = 0.2
threshold_ratio = 1.2
maxDepth = 5
```

## random_forest

Uses [pyspark.ml.classification.RandomForestClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html).
* Parameters:
  * `maxDepth` -- Type: `int`. Maximum depth of the tree. Spark default value is 5.
  * `numTrees` -- Type: `int`. The number of trees to train.  Spark default value is 20, must be >= 1.
  * `featureSubsetStrategy` -- Type: `string`. Per the Spark docs: "The number of features to consider for splits at each tree node. Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n]."

```toml
[training.chosen_model]
type = "random_forest"
threshold = 0.15
threshold_ratio = 1.0
maxDepth = 5
numTrees = 75
featureSubsetStrategy = "sqrt"
```

## probit

Uses [pyspark.ml.regression.GeneralizedLinearRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.GeneralizedLinearRegression.html) with `family="binomial"` and `link="probit"`.  

```toml
[training.chosen_model]
type = "probit"
threshold = 0.85
threshold_ratio = 1.2
```

## logistic_regression

Uses [pyspark.ml.classification.LogisticRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html)

```toml
[training.chosen_model]
type = "logistic_regression"
threshold = 0.5
threshold_ratio = 1.0
```

## decision_tree

Uses [pyspark.ml.classification.DecisionTreeClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.DecisionTreeClassifier.html).

* Parameters:
  * `maxDepth` -- Type: `int`.  Maximum depth of the tree.
  * `minInstancesPerNode` -- Type `int`. Per the Spark docs: "Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1."
  * `maxBins` -- Type: `int`. Per the Spark docs: "Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature."

```toml
[training.chosen_model]
type = "decision_tree"
threshold = 0.5
threshold_ratio = 1.5
maxDepth = 6
minInstancesPerNode = 2
maxBins = 4
```

## gradient_boosted_trees

Uses [pyspark.ml.classification.GBTClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html).

* Parameters:
  * `maxDepth` -- Type: `int`.  Maximum depth of the tree.
  * `minInstancesPerNode` -- Type `int`. Per the Spark docs: "Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1."
  * `maxBins` -- Type: `int`. Per the Spark docs: "Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature."
  
```toml
[training.chosen_model]
type = "gradient_boosted_trees"
threshold = 0.7
threshold_ratio = 1.3
maxDepth = 4
minInstancesPerNode = 1
maxBins = 6
```

## xgboost

*Added in version 3.8.0.*

XGBoost is an alternate, high-performance implementation of gradient boosting.
It uses [xgboost.spark.SparkXGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.spark.SparkXGBClassifier).
Since the XGBoost-PySpark integration which the xgboost Python package provides
is currently unstable, support for the xgboost model type is disabled in hlink
by default. hlink will stop with an error if you try to use this model type
without enabling support for it. To enable support for xgboost, install hlink
with the `xgboost` extra.

```
pip install hlink[xgboost]
```

This installs the xgboost package and its Python dependencies. Depending on
your machine and operating system, you may also need to install the libomp
library, which is another dependency of xgboost. xgboost should raise a helpful
error if it detects that you need to install libomp.

You can view a list of xgboost's parameters
[here](https://xgboost.readthedocs.io/en/latest/parameter.html).

```toml
[training.chosen_model]
type = "xgboost"
threshold = 0.8
threshold_ratio = 1.5
max_depth = 5
eta = 0.5
gamma = 0.05
```

## lightgbm

*Added in version 3.8.0.*

LightGBM is another alternate, high-performance implementation of gradient
boosting. It uses
[synapse.ml.lightgbm.LightGBMClassifier](https://mmlspark.blob.core.windows.net/docs/1.0.8/pyspark/synapse.ml.lightgbm.html#module-synapse.ml.lightgbm.LightGBMClassifier).
`synapse.ml` is a library which provides various integrations with PySpark,
including integrations between the C++ LightGBM library and PySpark.

LightGBM requires some additional Scala libraries that hlink does not usually
install, so support for the lightgbm model is disabled in hlink by default.
hlink will stop with an error if you try to use this model type without
enabling support for it. To enable support for lightgbm, install hlink with the
`lightgbm` extra.

```
pip install hlink[lightgbm]
```

This installs the lightgbm package and its Python dependencies. Depending on
your machine and operating system, you may also need to install the libomp
library, which is another dependency of lightgbm. If you encounter errors when
training a lightgbm model, please try installing libomp if you do not have it
installed.

lightgbm has an enormous number of available parameters. Many of these are
available as normal in hlink, via the [LightGBMClassifier
class](https://mmlspark.blob.core.windows.net/docs/1.0.8/pyspark/synapse.ml.lightgbm.html#module-synapse.ml.lightgbm.LightGBMClassifier).
Others are available through the special `passThroughArgs` parameter, which
passes additional parameters through to the C++ library. You can see a full
list of the supported parameters
[here](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

```toml
[training.chosen_model]
type = "lightgbm"
# hlink's threshold and threshold_ratio
threshold = 0.8
threshold_ratio = 1.5
# LightGBMClassifier supports these parameters (and many more).
maxDepth = 5
learningRate = 0.5
# LightGBMClassifier does not directly support this parameter,
# so we have to send it to the C++ library with passThroughArgs.
passThroughArgs = "force_row_wise=true"
```
