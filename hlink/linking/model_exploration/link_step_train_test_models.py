# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import collections.abc
import itertools
import logging
import math
import random
import re
import statistics
import sys
from textwrap import dedent
from time import perf_counter
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from pyspark.ml import Model, Transformer
import pyspark.sql
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, count_if, mean
from functools import reduce
import hlink.linking.core.model_metrics as metrics_core
import hlink.linking.core.threshold as threshold_core
import hlink.linking.core.classifier as classifier_core

from hlink.linking.link_step import LinkStep

#   This is a refactor to make the train-test model process faster.
"""

Current Nested CV implementation:

1. Prepare train-test data
2.  Split prepared  data into 'n' outer folds (distinct pieces.)
3. For 'outer_index' in outer folds length:
    test_data := outer_folds[outer_fold_index]
    training_data := combine(outer_folds, excluding = outer_fold_index)

    model_results := []
    inner_folds := split training_data into 'j' inner folds
    for inner_fold_index in inner_folds length:
        inner_test_data := inner_folds[inner_fold_index]
        inner_training_data := combine(inner_folds, exclude = inner_fold_index)
        for param_set in all_hyper_params():
                model_results.append(train_test(params, inner_test_data, inner_training_data))
    score_models(model_results)
    best_model := select_best_model(model_results)

    for threshold_values in all_threshold_combinations:
        train_test_results :=  train_test(best_model, test_data, training_data)
        collect_train_test_results(train_test_results)
4.. Report train_test_results



Complexity: n*t + n*j*p

j == inner folds, n == outer folds, t == threshold combinations, p == hyper-parameter tests (grid, random)

Revised algorithm:

1. Prepare test-train data
2. split data into n pairs of training and test data. In our tests n == 10.
3. for every model type, for each combination of hyper-parameters
    for train, test in n splits:
        train the model with the training data
        test the trained model using the test data
        capture the probability of correct predictions on each split
    Score the model based on some function of the collected probabilities (like 'mean')
    Store the score with the model type and hyper-parameters that produced the score

4. Select the best performing model type + hyper-parameter set based on the associated score.
5. With the best scoring parameters and model:
    Obtain  a single training data and test data split
    for each threshold setting combination:
        Train the model type with the associated hyper-parameters
        Predict the matches on the test data using the trained model
        Evaluate the predictions and capture the threshold combination that made it.
6. Print the results of the threshold evaluations

p = hyper-parameter combinations
s = number of splits
t = threshold matrix size (x * y)

complexity = s * p + t ->    O(n^2)

We may end up needing to test the thresholds on multiple splits:

    s * p + s * t

It's hard to generalize the  number of passes on the data since 't' may be pretty large or not at all. 's' will probably be 10 or so and 'p' also can vary a lot from 2 or 3 to 100.


Original Algorithm:


1. Prepare test-train data
2. split data into n pairs of training and test data. In our tests n == 10.
3. for every model type, for each combination of hyper-parameters
    for train, test in n splits:
        train the model with the training data
        test the trained model using the test data
        capture the probability of correct predictions on each split

        4. With the best scoring parameters and model:
            for each threshold setting combination:
                Train the model type with the associated hyper-parameters
                Predict the matches on the test data using the trained model
                Evaluate the predictions and capture the threshold combination and hyper-parameters that made it.
6. Print the results of the threshold evaluations

complexity = p * s * t  ->  O(n^3)


"""


logger = logging.getLogger(__name__)


# Model evaluation score with the inputs that produced the score.
@dataclass(kw_only=True)
class ModelEval:
    model_type: str
    score: float
    hyperparams: dict[str, Any]
    threshold: float | list[float]
    threshold_ratio: float | list[float] | None

    def print(self):
        return f"{self.model_type} {self.score} params: {self.hyperparams}"

    def make_threshold_matrix(self) -> list[list[float]]:
        return _calc_threshold_matrix(self.threshold, self.threshold_ratio)


# Both training and test results can be captured in this type
@dataclass(kw_only=True)
class ThresholdTestResult:
    model_id: str
    alpha_threshold: float
    threshold_ratio: float
    true_pos: int
    true_neg: int
    false_pos: int
    false_neg: int
    precision: float
    recall: float
    mcc: float
    f_measure: float
    pr_auc: float


class LinkStepTrainTestModels(LinkStep):
    def __init__(self, task) -> None:
        super().__init__(
            task,
            "train test models",
            input_table_names=[
                f"{task.table_prefix}training_features",
                f"{task.table_prefix}training_vectorized",
            ],
            output_table_names=[
                f"{task.table_prefix}training_results",
                f"{task.table_prefix}repeat_fps",
                f"{task.table_prefix}repeat_fns",
                f"{task.table_prefix}repeat_tps",
                f"{task.table_prefix}repeat_tns",
            ],
        )

    # Takes a list of the PRAUC (Precision / Recall area under the curve) and the scoring strategy to use
    def _score_inner_kfold_cv_results(
        self, areas: list[float], score_strategy: str = "mean"
    ) -> float:
        if score_strategy == "mean":
            return statistics.mean(areas)
        else:
            raise RuntimeError(f"strategy {score_strategy} not implemented.")

    def _train_model(
        self, training_data, test_data, model_type, hyperparams, dep_var, id_a, id_b
    ) -> float:
        classifier, post_transformer = classifier_core.choose_classifier(
            model_type, hyperparams, dep_var
        )

        logger.debug("Training the model on the training data split")
        start_train_time = perf_counter()
        model = classifier.fit(training_data)
        end_train_time = perf_counter()
        logger.debug(
            f"Successfully trained the model in {end_train_time - start_train_time:.2f}s"
        )
        predictions_tmp = _get_probability_and_select_pred_columns(
            test_data, model, post_transformer, id_a, id_b, dep_var
        )

        test_pred = predictions_tmp.toPandas()
        precision, recall, thresholds_raw = precision_recall_curve(
            test_pred[dep_var],
            test_pred["probability"].round(2),
            pos_label=1,
        )
        pr_auc = auc(recall, precision)
        return pr_auc

    # Returns a PR AUC list computation for inner training data on the given model
    def _collect_inner_kfold_cv(
        self,
        inner_folds: list[pyspark.sql.DataFrame],
        model_type: str,
        hyperparams: dict[str, Any],
        dep_var: str,
        id_a: str,
        id_b: str,
    ) -> list[float]:
        start_time = perf_counter()
        # Collect auc values so we can pull out the highest
        validation_results = []
        for validation_index in range(len(inner_folds)):
            validation_data = inner_folds[validation_index]
            c_start_time = perf_counter()
            training_data = self._combine_folds(inner_folds, ignore=validation_index)
            c_end_time = perf_counter()
            logger.debug(
                f"Combined inner folds to make training data, except {validation_index}, took {c_end_time - c_start_time:.2f}"
            )

            cached_training_data = training_data.cache()
            cached_validation_data = validation_data.cache()

            # PRAUC = Precision Recall under the curve
            prauc = self._train_model(
                cached_training_data,
                cached_validation_data,
                model_type,
                hyperparams,
                dep_var,
                id_a,
                id_b,
            )
            training_data.unpersist()
            validation_data.unpersist()
            validation_results.append(prauc)
        end_time = perf_counter()
        logger.debug(
            f"Inner folds: Evaluated model + params on {len(inner_folds)} folds in {end_time - start_time:.2f}"
        )
        logger.debug(f"Validation results {validation_results}")
        return validation_results

    # Returns a list of  ModelEval instances.
    # This connects a score to each hyper-parameter combination. and the thresholds  listed with it in the config.
    def _evaluate_hyperparam_combinations(
        self,
        all_model_parameter_combos,
        inner_folds: list[pyspark.sql.DataFrame],
        dep_var: str,
        id_a: str,
        id_b: str,
        training_settings,
    ) -> list[ModelEval]:
        info = f"Begin evaluating all {len(all_model_parameter_combos)} selected hyperparameter combinations."
        print(info)
        logger.debug(info)
        results = []
        for index, params_combo in enumerate(all_model_parameter_combos, 1):
            eval_start_info = f"Starting run {index} of {len(all_model_parameter_combos)} with these parameters: {params_combo}"
            # print(eval_start_info)
            logger.info(eval_start_info)
            # Copy because the params combo will get stripped of extra key-values
            # so only the hyperparams remain.
            hyperparams = params_combo.copy()

            model_type = hyperparams.pop("type")

            # While we're not using thresholds in this function, we need to capture them here
            # since they can be different for different model types and
            # we need to use model_type, params, score and thresholds to
            # do the next step using thresholds.
            threshold, threshold_ratio = self._get_thresholds(
                hyperparams, training_settings
            )
            # thresholds and model_type  are mixed in with the model hyper-parameters
            # in the config; this removes them before passing to the model training.
            hyperparams.pop("threshold", None)
            hyperparams.pop("threshold_ratio", None)

            pr_auc_values = self._collect_inner_kfold_cv(
                inner_folds, model_type, hyperparams, dep_var, id_a, id_b
            )
            score = self._score_inner_kfold_cv_results(pr_auc_values, "mean")

            model_eval = ModelEval(
                model_type=model_type,
                score=score,
                hyperparams=hyperparams,
                threshold=threshold,
                threshold_ratio=threshold_ratio,
            )
            info = f"{index}: {model_eval.print()}"
            print(info)
            logger.debug(info)
            results.append(model_eval)
        return results

    # Grabs the threshold settings from a single model parameter combination row (after all combinations
    # are exploded.) Does not alter the params structure.)
    def _get_thresholds(self, model_parameters, training_settings) -> tuple[Any, Any]:
        alpha_threshold = model_parameters.get(
            "threshold", training_settings.get("threshold", 0.8)
        )
        if training_settings.get("decision") == "drop_duplicate_with_threshold_ratio":
            threshold_ratio = model_parameters.get(
                "threshold_ratio",
                threshold_core.get_threshold_ratio(training_settings, model_parameters),
            )
        else:
            threshold_ratio = None

        return alpha_threshold, threshold_ratio

    # Note: Returns only one model training session; if
    # your config specified more than one model type and thresholds, you'll get
    # the best result according to the scoring system, not the best for each
    # model type.
    def _choose_best_training_results(self, evals: list[ModelEval]) -> ModelEval:
        if len(evals) == 0:
            raise RuntimeError(
                "No model evaluations provided, cannot choose the best one."
            )
        print("\n\n**************************************************")
        print("    All Model - hyper-parameter combinations")
        print("**************************************************\n")
        best_eval = evals[0]
        for e in evals:
            print(e)
            if best_eval.score < e.score:
                best_eval = e
        print("--------------------------------------------------\n\n")
        return best_eval

    def _evaluate_threshold_combinations(
        self,
        best_model: ModelEval,
        split: dict[str : pyspark.sql.DataFrame],
        dep_var: str,
        id_a: str,
        id_b: str,
    ) -> tuple[dict[int, pd.DataFrame], Any]:
        training_config_name = str(self.task.training_conf)
        config = self.task.link_run.config

        id_column = config["id_column"]
        training_settings = config[training_config_name]

        thresholding_training_data = split.get("training")
        thresholding_test_data = split.get("test")
        if thresholding_training_data is None:
            raise RuntimeError("Must give some data with the 'training' key.")
        if thresholding_test_data is None:
            raise RuntimeError("Must give some data with the 'test' key.")

        print(f"\n======== Best Model and Parameters ========\n")
        print(f"\t{best_model}\n")
        print("=============================================\n\n")
        logger.debug(f"Best model results: {best_model}")

        threshold_matrix = best_model.make_threshold_matrix()
        logger.debug(f"The threshold matrix has {len(threshold_matrix)} entries")
        info = f"\nTesting the best model + parameters against all {len(threshold_matrix)} threshold combinations.\n"
        logger.debug(info)

        prediction_results: dict[int, ThresholdTestResult] = {}
        # training_results: dict[int, ThresholdTestResult] = {}

        cached_training_data = thresholding_training_data.cache()
        cached_test_data = thresholding_test_data.cache()

        thresholding_classifier, thresholding_post_transformer = (
            classifier_core.choose_classifier(
                best_model.model_type, best_model.hyperparams, dep_var
            )
        )
        start_time = perf_counter()
        thresholding_model = thresholding_classifier.fit(cached_training_data)
        end_time = perf_counter()
        logger.debug(
            f"Trained model on thresholding training data, took {end_time - start_time:.2f}s"
        )

        thresholding_predictions = _get_probability_and_select_pred_columns(
            cached_test_data,
            thresholding_model,
            thresholding_post_transformer,
            id_a,
            id_b,
            dep_var,
        )

        for threshold_index, (
            this_alpha_threshold,
            this_threshold_ratio,
        ) in enumerate(threshold_matrix, 0):

            diag = (
                f"Predicting with threshold matrix entry {threshold_index+1} of {len(threshold_matrix)}: "
                f"{this_alpha_threshold=} and {this_threshold_ratio=}"
            )
            logger.debug(diag)
            decision = training_settings.get("decision")
            start_predict_time = perf_counter()

            predictions = threshold_core.predict_using_thresholds(
                thresholding_predictions,
                this_alpha_threshold,
                this_threshold_ratio,
                id_column,
                decision,
            )

            end_predict_time = perf_counter()
            info = f"Predictions for test-train data on threshold took {end_predict_time - start_predict_time:.2f}s"
            logger.debug(info)

            prediction_results[threshold_index] = self._capture_prediction_results(
                predictions,
                dep_var,
                thresholding_model,
                this_alpha_threshold,
                this_threshold_ratio,
                best_model.score,
            )

        thresholding_test_data.unpersist()
        thresholding_training_data.unpersist()

        return prediction_results

    def _run(self) -> None:
        training_section_name = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config
        training_settings = config[training_section_name]

        self.task.spark.sql("set spark.sql.shuffle.partitions=1")

        dep_var = training_settings["dependent_var"]
        id_a = config["id_column"] + "_a"
        id_b = config["id_column"] + "_b"

        columns_to_keep = [id_a, id_b, "features_vector", dep_var]
        prepped_data = (
            self.task.spark.table(f"{table_prefix}training_vectorized")
            .select(columns_to_keep)
            .cache()
        )

        outer_fold_count = training_settings.get("n_training_iterations", 10)
        inner_fold_count = 3

        if outer_fold_count < 3:
            raise RuntimeError("You must use at least three outer folds.")

        # At the end we combine this information collected from every outer fold
        threshold_test_results: list[ThresholdTestResult] = []
        # threshold_training_results: list[ThresholdTestResult]
        best_models: list[ModelEval] = []

        seed = training_settings.get("seed", 2133)

        outer_folds = self._get_outer_folds(prepped_data, id_a, outer_fold_count, seed)

        for test_data_index, outer_test_data in enumerate(outer_folds):
            print(
                f"\nTesting fold {test_data_index} -------------------------------------------------\n"
            )
            # Explode params into all the combinations we want to test with the current model.
            # This may use a grid search or a random search or exactly the parameters in the config.
            model_parameters = _get_model_parameters(training_settings)

            outer_training_data = self._combine_folds(
                outer_folds, ignore=test_data_index
            )
            print(
                f"Combine non-test outer folds into {outer_training_data.count()} training data records."
            )

            inner_folds = self._split_into_folds(
                outer_training_data, inner_fold_count, seed
            )

            hyperparam_evaluation_results = self._evaluate_hyperparam_combinations(
                model_parameters,
                inner_folds,
                dep_var,
                id_a,
                id_b,
                training_settings,
            )

            print(
                f"Take the best hyper-parameter set from {len(hyperparam_evaluation_results)} results and test every threshold combination against it..."
            )

            # Note: We may change this to contain a list of best per model or something else
            # but for now it's a single ModelEval instance -- the one with the highest score.
            best_model = self._choose_best_training_results(
                hyperparam_evaluation_results
            )

            prediction_results = self._evaluate_threshold_combinations(
                best_model,
                {"test": outer_test_data, "training": outer_training_data},
                dep_var,
                id_a,
                id_b,
            )

            # Collect the outputs for each fold
            threshold_test_results.append(prediction_results)
            # threshold_training_results.append(training_results)
            best_models.append(best_model)

        combined_test = _combine_by_threshold_matrix_entry(threshold_test_results)
        # combined_train = (_combine_by_threshold_matrix_entry(training_results),)

        # there are 'm'  threshold_test_results items matching the number of
        # inner folds. Each entry has 'n' items matching the number of
        # threshold matrix entries.
        threshold_matrix_size = len(threshold_test_results[0])

        thresholded_metrics_df = pd.DataFrame()
        for i in range(threshold_matrix_size):
            print(f"Aggregate threshold matrix entry {i}")
            thresholded_metrics_df = _aggregate_per_threshold_results(
                thresholded_metrics_df, combined_test[i], best_models
            )

        print("***   Final thresholded metrics ***")

        # Convert the parameters column to dtype string so that Spark can handle it
        thresholded_metrics_df["parameters"] = thresholded_metrics_df[
            "parameters"
        ].apply(lambda t: str(t) if pd.notnull(t) else t)
        # thresholded_metrics_df has one row per threshold combination. and each outer fold
        with pd.option_context(
            "display.max_columns", None, "display.max_colwidth", None
        ):
            print(thresholded_metrics_df.sort_values(by="mcc_mean", ascending=False))
        print("\n")

        self._save_training_results(thresholded_metrics_df, self.task.spark)
        self.task.spark.sql("set spark.sql.shuffle.partitions=200")

    def _split_into_folds(
        self, data: pyspark.sql.DataFrame, fold_count: int, seed: int
    ) -> list[pyspark.sql.DataFrame]:
        weights = [1.0 / fold_count for i in range(fold_count)]
        return data.randomSplit(weights, seed=seed)

    def _combine_folds(
        self, folds: list[pyspark.sql.DataFrame], ignore=None
    ) -> pyspark.sql.DataFrame:

        folds_to_combine = []
        for fold_number, fold in enumerate(folds, 0):
            if fold_number != ignore:
                folds_to_combine.append(fold)

        combined = reduce(DataFrame.unionAll, folds_to_combine).cache()
        return combined

    def _get_outer_folds(
        self, prepped_data: pyspark.sql.DataFrame, id_a: str, k_folds: int, seed: int
    ) -> list[pyspark.sql.DataFrame]:

        print(
            f"Create {k_folds} outer folds from {prepped_data.count()} training records."
        )

        weights = [1.0 / k_folds for i in range(k_folds)]
        print(f"Split into folds using weights {weights}")
        fold_ids_list = (
            prepped_data.select(id_a).distinct().randomSplit(weights, seed=seed + 1)
        )
        outer_folds = [
            prepped_data.join(f_ids, on=id_a, how="inner") for f_ids in fold_ids_list
        ]
        print(f"There are {len(outer_folds)} outer folds")
        for i, f in enumerate(outer_folds, 0):
            print(f"Fold {i} has {f.count()} records.")

        return outer_folds

    def _get_splits(
        self,
        prepped_data: pyspark.sql.DataFrame,
        id_a: str,
        n_training_iterations: int,
        seed: int,
    ) -> list[list[pyspark.sql.DataFrame]]:
        """
        Get a list of random splits of the prepped_data into two DataFrames.
        There are n_training_iterations elements in the list. Each element is
        itself a list of two DataFrames which are the splits of prepped_data.
        The split DataFrames are roughly equal in size.
        """
        print(
            f"Splitting prepped data that starts with  {prepped_data.count()} total rows."
        )
        if self.task.link_run.config[f"{self.task.training_conf}"].get(
            "split_by_id_a", False
        ):
            print("Get distinct id_a for training")
            split_ids = [
                prepped_data.select(id_a)
                .distinct()
                .randomSplit([0.5, 0.5], seed=seed + i)
                for i in range(n_training_iterations)
            ]

            splits = []
            for ids_a, ids_b in split_ids:
                split_a = prepped_data.join(ids_a, on=id_a, how="inner")
                split_b = prepped_data.join(ids_b, on=id_a, how="inner")
                splits.append([split_a, split_b])
        else:
            print("Splitting randomly n times.")
            splits = [
                prepped_data.randomSplit([0.5, 0.5], seed=seed + i)
                for i in range(n_training_iterations)
            ]

        print(f"There are {len(splits)}")
        for index, s in enumerate(splits, 1):
            training_data = s[0]
            test_data = s[1]

            print(
                f"Split {index}: training rows {training_data.count()} test rows: {test_data.count()}"
            )
        return splits

    def _capture_prediction_results(
        self,
        predictions: pyspark.sql.DataFrame,
        dep_var: str,
        model: Model,
        alpha_threshold: float,
        threshold_ratio: float | None,
        pr_auc: float,
    ) -> ThresholdTestResult:
        table_prefix = self.task.table_prefix
        # write to sql tables for testing
        predictions.createOrReplaceTempView(f"{table_prefix}predictions")

        (
            true_pos,
            false_pos,
            false_neg,
            true_neg,
        ) = _get_confusion_matrix(predictions, dep_var)
        precision = metrics_core.precision(true_pos, false_pos)
        recall = metrics_core.recall(true_pos, false_neg)
        mcc = metrics_core.mcc(true_pos, true_neg, false_pos, false_neg)
        f_measure = metrics_core.f_measure(true_pos, false_pos, false_neg)

        result = ThresholdTestResult(
            model_id=model,
            alpha_threshold=alpha_threshold,
            threshold_ratio=threshold_ratio,
            true_pos=true_pos,
            true_neg=true_neg,
            false_pos=false_pos,
            false_neg=false_neg,
            precision=precision,
            recall=recall,
            mcc=mcc,
            f_measure=f_measure,
            pr_auc=pr_auc,
        )

        return result

    def _save_training_results(
        self, desc_df: pd.DataFrame, spark: pyspark.sql.SparkSession
    ) -> None:
        table_prefix = self.task.table_prefix

        if desc_df.empty:
            print("Training results dataframe is empty.")
        else:
            spark.createDataFrame(desc_df, samplingRatio=1).write.mode(
                "overwrite"
            ).saveAsTable(f"{table_prefix}training_results")
            # print(
            #    f"Training results saved to Spark table '{table_prefix}training_results'."
            # )


def _calc_threshold_matrix(
    alpha_threshold: float | list[float], threshold_ratio: float | list[float] | None
) -> list[list[float]]:
    if alpha_threshold and type(alpha_threshold) != list:
        alpha_threshold = [alpha_threshold]

    if threshold_ratio and type(threshold_ratio) != list:
        threshold_ratio = [threshold_ratio]

    if threshold_ratio:
        threshold_matrix = [[a, b] for a in alpha_threshold for b in threshold_ratio]
    else:
        threshold_matrix = [[a, np.nan] for a in alpha_threshold]

    return threshold_matrix


def _get_probability_and_select_pred_columns(
    pred_df: pyspark.sql.DataFrame,
    model: Model,
    post_transformer: Transformer,
    id_a: str,
    id_b: str,
    dep_var: str,
) -> pyspark.sql.DataFrame:
    all_prediction_cols = set(
        [
            f"{id_a}",
            f"{id_b}",
            dep_var,
            "probability",
            "probability_array",
            "prediction",
        ]
    )
    transformed_df = model.transform(pred_df)
    post_transform_df = post_transformer.transform(transformed_df)
    required_col_df = post_transform_df.select(
        list(all_prediction_cols & set(post_transform_df.columns))
    )
    return required_col_df


def _get_confusion_matrix(
    predictions: pyspark.sql.DataFrame,
    dep_var: str,
) -> tuple[int, int, int, int]:
    """
    Compute the confusion matrix for the given DataFrame of predictions. The
    confusion matrix is the count of true positives, false positives, false
    negatives, and true negatives for the predictions.

    Return a tuple (true_pos, false_pos, false_neg, true_neg).
    """
    prediction_col = col("prediction")
    label_col = col(dep_var)

    confusion_matrix = predictions.select(
        count_if((label_col == 1) & (prediction_col == 1)).alias("true_pos"),
        count_if((label_col == 0) & (prediction_col == 1)).alias("false_pos"),
        count_if((label_col == 1) & (prediction_col == 0)).alias("false_neg"),
        count_if((label_col == 0) & (prediction_col == 0)).alias("true_neg"),
    )
    [confusion_row] = confusion_matrix.collect()
    return (
        confusion_row.true_pos,
        confusion_row.false_pos,
        confusion_row.false_neg,
        confusion_row.true_neg,
    )


# The outer list  entries hold results from each outer fold, the inner list has a ThresholdTestResult per threshold
# matrix entry. We need to get data for each threshold entry together. Basically we need to invert the data.
def _combine_by_threshold_matrix_entry(
    threshold_results: list[dict[int, ThresholdTestResult]],
) -> list[ThresholdTestResult]:
    # This list will have a size of the number of threshold matrix entries
    results: list[list[ThresholdTestResult]] = []

    # Check number of folds
    if len(threshold_results) < 2:
        raise RuntimeError("Must have at least two outer folds.")

    # Check if there are more than 0 threshold matrix entries
    if len(threshold_results[0]) == 0:
        raise RuntimeError(
            "No entries in the first set of threshold results; can't determine threshold matrix size."
        )

    inferred_threshold_matrix_size = len(threshold_results[0])

    for t in range(inferred_threshold_matrix_size):
        # One list per threshold matrix entry
        results.append([])

    for fold_results in threshold_results:
        for t in range(inferred_threshold_matrix_size):
            threshold_results_for_this_fold = fold_results[t]
            results[t].append(threshold_results_for_this_fold)
    return results


def _compute_mean_and_stdev(values: list[float]) -> (float, float):
    """
    Given a list of floats, return a tuple (mean, stdev). If there aren't enough
    values to compute the mean and/or stdev, return np.nan for that entry.
    """
    try:
        mean = statistics.mean(values)
    except statistics.StatisticsError:
        mean = np.nan

    try:
        stdev = statistics.stdev(values)
    except statistics.StatisticsError:
        stdev = np.nan

    return (mean, stdev)


def _aggregate_per_threshold_results(
    thresholded_metrics_df: pd.DataFrame,
    prediction_results: list[ThresholdTestResult],
    # training_results: list[ThresholdTestResult],
    best_models: list[ModelEval],
) -> pd.DataFrame:

    # The threshold is the same for all entries in the lists
    alpha_threshold = prediction_results[0].alpha_threshold
    threshold_ratio = prediction_results[0].threshold_ratio

    # Pull out columns to be aggregated
    precision = [r.precision for r in prediction_results if not math.isnan(r.precision)]
    recall = [r.recall for r in prediction_results if not math.isnan(r.recall)]
    pr_auc = [r.pr_auc for r in prediction_results if not math.isnan(r.pr_auc)]
    mcc = [r.mcc for r in prediction_results if not math.isnan(r.mcc)]
    f_measure = [r.f_measure for r in prediction_results if not math.isnan(r.f_measure)]

    (precision_mean, precision_sd) = _compute_mean_and_stdev(precision)
    (recall_mean, recall_sd) = _compute_mean_and_stdev(recall)
    (pr_auc_mean, pr_auc_sd) = _compute_mean_and_stdev(pr_auc)
    (mcc_mean, mcc_sd) = _compute_mean_and_stdev(mcc)
    (f_measure_mean, f_measure_sd) = _compute_mean_and_stdev(f_measure)

    new_desc = pd.DataFrame(
        {
            "model": [best_models[0].model_type],
            "parameters": [best_models[0].hyperparams],
            "alpha_threshold": [alpha_threshold],
            "threshold_ratio": [threshold_ratio],
            "precision_mean": [precision_mean],
            "precision_sd": [precision_sd],
            "recall_mean": [recall_mean],
            "recall_sd": [recall_sd],
            "pr_auc_mean": [pr_auc_mean],
            "pr_auc_sd": [pr_auc_sd],
            "mcc_mean": [mcc_mean],
            "mcc_sd": [mcc_sd],
            "f_measure_mean": [f_measure_mean],
            "f_measure_sd": [f_measure_sd],
        },
    )

    thresholded_metrics_df = pd.concat(
        [thresholded_metrics_df, new_desc], ignore_index=True
    )

    return thresholded_metrics_df


def _custom_param_grid_builder(
    model_parameters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    print("Building param grid for models")
    given_parameters = model_parameters
    new_params = []
    for run in given_parameters:
        params = run.copy()
        model_type = params.pop("type")

        # dropping thresholds to prep for scikitlearn model exploration refactor
        threshold = params.pop("threshold", False)
        threshold_ratio = params.pop("threshold_ratio", False)

        keys = params.keys()
        values = params.values()

        params_exploded = []
        for prod in itertools.product(*values):
            params_exploded.append(dict(zip(keys, prod)))

        for subdict in params_exploded:
            subdict["type"] = model_type
            if threshold:
                subdict["threshold"] = threshold
            if threshold_ratio:
                subdict["threshold_ratio"] = threshold_ratio

        new_params.extend(params_exploded)
    return new_params


def _choose_randomized_parameters(
    rng: random.Random, model_parameters: dict[str, Any]
) -> dict[str, Any]:
    """
    Choose a randomized setting of parameters from the given specification.
    """
    parameter_choices = dict()

    for key, value in model_parameters.items():
        # If it's a Sequence (usually list) but not a string, choose one of the values at random.
        if isinstance(value, collections.abc.Sequence) and not isinstance(value, str):
            parameter_choices[key] = rng.choice(value)
        # If it's a Mapping (usually dict), it defines a distribution from which
        # the parameter should be sampled.
        elif isinstance(value, collections.abc.Mapping):
            distribution = value["distribution"]

            if distribution == "randint":
                low = value["low"]
                high = value["high"]
                parameter_choices[key] = rng.randint(low, high)
            elif distribution == "uniform":
                low = value["low"]
                high = value["high"]
                parameter_choices[key] = rng.uniform(low, high)
            elif distribution == "normal":
                mean = value["mean"]
                stdev = value["standard_deviation"]
                parameter_choices[key] = rng.normalvariate(mean, stdev)
            else:
                raise ValueError(
                    f"Unknown distribution '{distribution}'. Please choose one of 'randint', 'uniform', or 'normal'."
                )
        # All other types (including strings) are passed through unchanged.
        else:
            parameter_choices[key] = value

    return parameter_choices


def _get_model_parameters(training_settings: dict[str, Any]) -> list[dict[str, Any]]:
    if "param_grid" in training_settings:
        print(
            dedent(
                """\
                Deprecation Warning: training.param_grid is deprecated.

                Please use training.model_parameter_search instead by replacing

                `param_grid = True` with `model_parameter_search = {strategy = "grid"}` or
                `param_grid = False` with `model_parameter_search = {strategy = "explicit"}`

                [deprecated_in_version=4.0.0]"""
            ),
            file=sys.stderr,
        )

    model_parameters = training_settings["model_parameters"]
    model_parameter_search = training_settings.get("model_parameter_search")
    seed = training_settings.get("seed")
    use_param_grid = training_settings.get("param_grid", False)

    if model_parameters == []:
        raise ValueError(
            "model_parameters is empty, so there are no models to evaluate"
        )

    if model_parameter_search is not None:
        strategy = model_parameter_search["strategy"]
        if strategy == "explicit":
            return model_parameters
        elif strategy == "grid":
            return _custom_param_grid_builder(model_parameters)
        elif strategy == "randomized":
            num_samples = model_parameter_search["num_samples"]
            rng = random.Random(seed)

            return_parameters = []
            # These keys are special and should not be sampled or modified. All
            # other keys are hyper-parameters to the model and should be sampled.
            frozen_keys = {"type", "threshold", "threshold_ratio"}
            for _ in range(num_samples):
                parameter_spec = rng.choice(model_parameters)
                sample_parameters = {
                    key: value
                    for (key, value) in parameter_spec.items()
                    if key not in frozen_keys
                }
                frozen_parameters = {
                    key: value
                    for (key, value) in parameter_spec.items()
                    if key in frozen_keys
                }

                randomized = _choose_randomized_parameters(rng, sample_parameters)
                result = {**frozen_parameters, **randomized}
                return_parameters.append(result)

            return return_parameters
        else:
            raise ValueError(
                f"Unknown model_parameter_search strategy '{strategy}'. "
                "Please choose one of 'explicit', 'grid', or 'randomized'."
            )
    elif use_param_grid:
        return _custom_param_grid_builder(model_parameters)

    return model_parameters
