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
