# Models

These are models available to be used in the model evaluation, training, and household training link tasks.

* Attributes for all models:
  * `threshold` -- Type: `float`.  Alpha threshold (model hyperparameter).
  * `threshold_ratio` -- Type: `float`.  Beta threshold (de-duplication distance ratio).
  * Any parameters available in the model as defined in the Spark documentation can be passed as params using the label given in the Spark docs.  Commonly used parameters are listed below with descriptive explanations from the Spark docs.

## random_forest

Uses [pyspark.ml.classification.RandomForestClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html).  Returns probability as an array.
* Parameters:
  * `maxDepth` -- Type: `int`. Maximum depth of the tree. Spark default value is 5.
  * `numTrees` -- Type: `int`. The number of trees to train.  Spark default value is 20, must be >= 1.
  * `featureSubsetStrategy` -- Type: `string`. Per the Spark docs: "The number of features to consider for splits at each tree node. Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n]."

```
model_parameters = {
    type = "random_forest",
    maxDepth = 5,
    numTrees = 75,
    featureSubsetStrategy = "sqrt",
    threshold = 0.15,
    threshold_ratio = 1.0
}
```

## probit

Uses [pyspark.ml.regression.GeneralizedLinearRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.GeneralizedLinearRegression.html) with `family="binomial"` and `link="probit"`.  

```
model_parameters = {
    type = "probit",
    threshold = 0.85,
    threshold_ratio = 1.2
}
```

## logistic_regression

Uses [pyspark.ml.classification.LogisticRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html)

```
chosen_model = {
    type = "logistic_regression",
    threshold = 0.5,
    threshold_ratio = 1.0
}
```

## decision_tree

Uses [pyspark.ml.classification.DecisionTreeClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.DecisionTreeClassifier.html).

* Parameters:
  * `maxDepth` -- Type: `int`.  Maximum depth of the tree.
  * `minInstancesPerNode` -- Type `int`. Per the Spark docs: "Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1."
  * `maxBins` -- Type: `int`. Per the Spark docs: "Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature."

```
chosen_model = {
    type = "decision_tree",
    maxDepth = 6,
    minInstancesPerNode = 2,
    maxBins = 4
}
```

## gradient_boosted_trees

Uses [pyspark.ml.classification.GBTClassifier](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html).

* Parameters:
  * `maxDepth` -- Type: `int`.  Maximum depth of the tree.
  * `minInstancesPerNode` -- Type `int`. Per the Spark docs: "Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1."
  * `maxBins` -- Type: `int`. Per the Spark docs: "Max number of bins for discretizing continuous features. Must be >=2 and >= number of categories for any categorical feature."
  
```
chosen_model = {
    type = "gradient_boosted_trees",
    maxDepth = 4,
    minInstancesPerNode = 1,
    maxBins = 6,
    threshold = 0.7,
    threshold_ratio = 1.3
}
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

```
chosen_model = {
    type = "xgboost",
    max_depth = 5,
    eta = 0.5,
    gamma = 0.05,
    threshold = 0.8,
    threshold_ratio = 1.5
}
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

```
chosen_model = {
    type = "lightgbm",
    # LightGBMClassifier supports these parameters (and many more).
    maxDepth = 5,
    learningRate = 0.5,
    # LightGBMClassifier does not directly support this parameter,
    # so we have to send it to the C++ library with passThroughArgs.
    passThroughArgs = "force_row_wise=true",
    # hlink's threshold and threshold_ratio
    threshold = 0.8,
    threshold_ratio = 1.5,
}
```
