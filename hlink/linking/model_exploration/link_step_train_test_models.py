# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import statistics
import itertools
import logging
import math
import re
from time import perf_counter
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from pyspark.ml import Model, Transformer
import pyspark.sql
from pyspark.sql.functions import count, mean

import hlink.linking.core.threshold as threshold_core
import hlink.linking.core.classifier as classifier_core

from hlink.linking.link_step import LinkStep

#   This is a refactor to make the train-test model process faster.
"""

Current algorithm:

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
    threshold_ratio: float | list[float] | bool

    def make_threshold_matrix(self) -> list[list[float]]:
        return _calc_threshold_matrix(self.threshold, self.threshold_ratio)


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
    def _score_train_test_results(
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
        predict_train_tmp = _get_probability_and_select_pred_columns(
            training_data, model, post_transformer, id_a, id_b, dep_var
        )

        test_pred = predictions_tmp.toPandas()
        precision, recall, thresholds_raw = precision_recall_curve(
            test_pred[f"{dep_var}"],
            test_pred["probability"].round(2),
            pos_label=1,
        )
        pr_auc = auc(recall, precision)
        return pr_auc

    # Returns a PR AUC list computation for each split of training and test data run through the model using model params
    def _collect_train_test_splits(
        self, splits, model_type, hyperparams, dep_var, id_a, id_b
    ) -> list[float]:
        # Collect auc values so we can pull out the highest
        splits_results = []
        for split_index, (training_data, test_data) in enumerate(splits, 1):
            cached_training_data = training_data.cache()
            cached_test_data = test_data.cache()

            split_start_info = f"Training and testing the model on train-test split {split_index} of {len(splits)}"
            # print(split_start_info)
            logger.debug(split_start_info)
            prauc = self._train_model(
                cached_training_data,
                cached_test_data,
                model_type,
                hyperparams,
                dep_var,
                id_a,
                id_b,
            )
            training_data.unpersist()
            test_data.unpersist()
            splits_results.append(prauc)
        return splits_results

    # Returns a list of  ModelEval instances.
    # This connects a score to each hyper-parameter combination. and the thresholds  listed with it in the config.
    def _evaluate_hyperparam_combinations(
        self,
        all_model_parameter_combos,
        splits,
        dep_var,
        id_a,
        id_b,
        config,
        training_conf,
    ) -> list[ModelEval]:
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
                hyperparams, config, training_conf
            )
            # thresholds and model_type  are mixed in with the model hyper-parameters
            # in the config; this removes them before passing to the model training.
            hyperparams.pop("threshold", None)
            hyperparams.pop("threshold_ratio", None)

            pr_auc_values = self._collect_train_test_splits(
                splits, model_type, hyperparams, dep_var, id_a, id_b
            )
            score = self._score_train_test_results(pr_auc_values, "mean")

            model_eval = ModelEval(
                model_type=model_type,
                score=score,
                hyperparams=hyperparams,
                threshold=threshold,
                threshold_ratio=threshold_ratio,
            )
            results.append(model_eval)
        return results

    # Grabs the threshold settings from a single model parameter combination row (after all combinations
    # are exploded.) Does not alter the params structure.)
    def _get_thresholds(
        self, model_parameters, config, training_conf
    ) -> tuple[Any, Any]:
        alpha_threshold = model_parameters.get(
            "threshold", config[training_conf].get("threshold", 0.8)
        )
        if (
            config[training_conf].get("decision", False)
            == "drop_duplicate_with_threshold_ratio"
        ):
            threshold_ratio = model_parameters.get(
                "threshold_ratio",
                threshold_core.get_threshold_ratio(
                    config[training_conf], model_parameters
                ),
            )
        else:
            threshold_ratio = False

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
        hyperparam_evaluation_results: list[ModelEval],
        suspicious_data: Any,
        split: list[pyspark.sql.DataFrame],
        dep_var: str,
        id_a: str,
        id_b: str,
    ) -> tuple[pd.DataFrame, Any]:
        training_conf = str(self.task.training_conf)
        config = self.task.link_run.config

        thresholded_metrics_df = _create_thresholded_metrics_df()

        # Note: We may change this to contain a list of best per model or something else
        # but for now it's a single ModelEval instance -- the one with the highest score.
        best_results = self._choose_best_training_results(hyperparam_evaluation_results)

        print(f"\n======== Best Model and Parameters ========\n")
        print(f"\t{best_results}\n")
        print("=============================================\n\n")

        threshold_matrix = best_results.make_threshold_matrix()
        logger.debug(f"The threshold matrix has {len(threshold_matrix)} entries")
        print(
            f"\nTesting the best model + parameters against all {len(threshold_matrix)} threshold combinations.\n"
        )
        results_dfs: dict[int, pd.DataFrame] = {}
        for i in range(len(threshold_matrix)):
            results_dfs[i] = _create_results_df()

        thresholding_training_data = split[0]
        thresholding_test_data = split[1]

        cached_training_data = thresholding_training_data.cache()
        cached_test_data = thresholding_test_data.cache()

        thresholding_classifier, thresholding_post_transformer = (
            classifier_core.choose_classifier(
                best_results.model_type, best_results.hyperparams, dep_var
            )
        )
        thresholding_model = thresholding_classifier.fit(cached_training_data)

        thresholding_predictions = _get_probability_and_select_pred_columns(
            cached_test_data,
            thresholding_model,
            thresholding_post_transformer,
            id_a,
            id_b,
            dep_var,
        )
        thresholding_predict_train = _get_probability_and_select_pred_columns(
            cached_training_data,
            thresholding_model,
            thresholding_post_transformer,
            id_a,
            id_b,
            dep_var,
        )

        i = 0
        for threshold_index, (
            this_alpha_threshold,
            this_threshold_ratio,
        ) in enumerate(threshold_matrix, 1):

            diag = (
                f"Predicting with threshold matrix entry {threshold_index} of {len(threshold_matrix)}: "
                f"{this_alpha_threshold=} and {this_threshold_ratio=}"
            )
            logger.debug(diag)
            predictions = threshold_core.predict_using_thresholds(
                thresholding_predictions,
                this_alpha_threshold,
                this_threshold_ratio,
                config[training_conf],
                config["id_column"],
            )
            predict_train = threshold_core.predict_using_thresholds(
                thresholding_predict_train,
                this_alpha_threshold,
                this_threshold_ratio,
                config[training_conf],
                config["id_column"],
            )

            results_dfs[i] = self._capture_results(
                predictions,
                predict_train,
                dep_var,
                thresholding_model,
                results_dfs[i],
                suspicious_data,
                this_alpha_threshold,
                this_threshold_ratio,
                best_results.score,
            )

            # for i in range(len(threshold_matrix)):
            thresholded_metrics_df = _append_results(
                thresholded_metrics_df,
                results_dfs[i],
                best_results.model_type,
                best_results.hyperparams,
            )
            i += 1

        thresholding_test_data.unpersist()
        thresholding_training_data.unpersist()

        return thresholded_metrics_df, suspicious_data

    def _run(self) -> None:
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        self.task.spark.sql("set spark.sql.shuffle.partitions=1")

        dep_var = config[training_conf]["dependent_var"]
        id_a = config["id_column"] + "_a"
        id_b = config["id_column"] + "_b"

        columns_to_keep = [id_a, id_b, "features_vector", dep_var]
        prepped_data = (
            self.task.spark.table(f"{table_prefix}training_vectorized")
            .select(columns_to_keep)
            .cache()
        )

        # Stores suspicious data
        otd_data = self._create_otd_data(id_a, id_b)

        outer_fold_count= config[training_conf].get("n_training_iterations", 10)
        inner_fold_count = 3

        if outer_fold_count < 2:
            raise RuntimeError("You must use at least two training iterations.")

        seed = config[training_conf].get("seed", 2133)

        outer_folds = self._get_outer_folds(
            prepped_data, id_a, outer_fold_count, seed
        )

        for test_data_index, thresholding_test_data in enumerate(outer_folds):             
            # Explode params into all the combinations we want to test with the current model.
            model_parameters = self._get_model_parameters(config)
            combined_training_data = _combine(outer_folds, ignore=test_data_index)

            hyperparam_evaluation_results = self._evaluate_hyperparam_combinations(
                model_parameters,
                combined_training_data,
                dep_var,
                id_a,
                id_b,
                config,
                training_conf,
            )

        # TODO: We may want to recreate a new split or set of splits rather than reuse existing splits.
        thresholded_metrics_df, suspicious_data = self._evaluate_threshold_combinations(
            hyperparam_evaluation_results,
            otd_data,
            thresholding_split,
            dep_var,
            id_a,
            id_b,
        )

        # thresholded_metrics_df has one row per threshold combination.
        thresholded_metrics_df = _load_thresholded_metrics_df_params(
            thresholded_metrics_df
        )

        print("***   Final thresholded metrics ***")

        _print_thresholded_metrics_df(
            thresholded_metrics_df.sort_values(by="mcc_test_mean", ascending=False)
        )
        self._save_training_results(thresholded_metrics_df, self.task.spark)
        self._save_otd_data(suspicious_data, self.task.spark)
        self.task.spark.sql("set spark.sql.shuffle.partitions=200")

    def _get_outer_folds(
            self,
            prepped_data: pyspark.sql.DataFrame,
            id_a: str,
            k_folds: int,
            seed: int) -> list[list[pyspark.sql.DataFrame]]:
        
        weights = [1.0/k_folds for i in k_folds]
        split_ids = prepped_data.select(id_a).distinct().randomSplit(weights, seed=seed)
            
        splits = []
        for ids_a, ids_b in split_ids:
            split_a = prepped_data.join(ids_a, on=id_a, how="inner")
            split_b = prepped_data.join(ids_b, on=id_a, how="inner")
            splits.append([split_a, split_b])
        for index, s in enumerate(splits, 1):
            training_data = s[0]
            test_data = s[1]

            print(
                f"Split {index}: training rows {training_data.count()} test rows: {test_data.count()}"
            )
        return splits

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
        print(f"Splitting prepped data that starts with  {prepped_data.count()} total rows.")
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

    def _custom_param_grid_builder(self, conf: dict[str, Any]) -> list[dict[str, Any]]:
        print("Building param grid for models")
        given_parameters = conf[f"{self.task.training_conf}"]["model_parameters"]
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

    def _capture_results(
        self,
        predictions: pyspark.sql.DataFrame,
        predict_train: pyspark.sql.DataFrame,
        dep_var: str,
        model: Model,
        results_df: pd.DataFrame,
        otd_data: dict[str, Any] | None,
        alpha_threshold: float,
        threshold_ratio: float,
        pr_auc: float,
    ) -> pd.DataFrame:
        table_prefix = self.task.table_prefix

        # write to sql tables for testing
        predictions.createOrReplaceTempView(f"{table_prefix}predictions")
        predict_train.createOrReplaceTempView(f"{table_prefix}predict_train")
        # print("------------------------------------------------------------")
        # print(f"Capturing predictions:")
        # predictions.show()
        # print(f"Capturing predict_train:")
        # predict_train.show()
        # print("------------------------------------------------------------")

        (
            test_TP_count,
            test_FP_count,
            test_FN_count,
            test_TN_count,
        ) = _get_confusion_matrix(predictions, dep_var, otd_data)
        test_precision, test_recall, test_mcc = _get_aggregate_metrics(
            test_TP_count, test_FP_count, test_FN_count, test_TN_count
        )

        (
            train_TP_count,
            train_FP_count,
            train_FN_count,
            train_TN_count,
        ) = _get_confusion_matrix(predict_train, dep_var, otd_data)
        train_precision, train_recall, train_mcc = _get_aggregate_metrics(
            train_TP_count, train_FP_count, train_FN_count, train_TN_count
        )

        new_results = pd.DataFrame(
            {
                "precision_test": [test_precision],
                "recall_test": [test_recall],
                "precision_train": [train_precision],
                "recall_train": [train_recall],
                "pr_auc": [pr_auc],
                "test_mcc": [test_mcc],
                "train_mcc": [train_mcc],
                "model_id": [model],
                "alpha_threshold": [alpha_threshold],
                "threshold_ratio": [threshold_ratio],
            },
        )
        return pd.concat([results_df, new_results], ignore_index=True)

    def _get_model_parameters(self, conf: dict[str, Any]) -> list[dict[str, Any]]:
        training_conf = str(self.task.training_conf)

        model_parameters = conf[training_conf]["model_parameters"]
        if "param_grid" in conf[training_conf] and conf[training_conf]["param_grid"]:
            model_parameters = self._custom_param_grid_builder(conf)
        elif model_parameters == []:
            raise ValueError(
                "No model parameters found. In 'training' config, either supply 'model_parameters' or 'param_grid'."
            )
        return model_parameters

    def _save_training_results(
        self, desc_df: pd.DataFrame, spark: pyspark.sql.SparkSession
    ) -> None:
        table_prefix = self.task.table_prefix

        if desc_df.empty:
            print("Training results dataframe is empty.")
        else:
            desc_df.dropna(axis=1, how="all", inplace=True)
            spark.createDataFrame(desc_df, samplingRatio=1).write.mode(
                "overwrite"
            ).saveAsTable(f"{table_prefix}training_results")
            # print(
            #    f"Training results saved to Spark table '{table_prefix}training_results'."
            # )

    def _prepare_otd_table(
        self, spark: pyspark.sql.SparkSession, df: pd.DataFrame, id_a: str, id_b: str
    ) -> pyspark.sql.DataFrame:
        spark_df = spark.createDataFrame(df)
        counted = (
            spark_df.groupby(id_a, id_b)
            .agg(
                count("*").alias("count"),
                mean("probability").alias("mean_probability"),
            )
            .filter("count > 1")
            .orderBy(["count", id_a, id_b])
        )
        return counted

    def _save_otd_data(
        self, otd_data: dict[str, Any] | None, spark: pyspark.sql.SparkSession
    ) -> None:
        table_prefix = self.task.table_prefix

        if otd_data is None:
            print("OTD suspicious data is None, not saving.")
            return
        id_a = otd_data["id_a"]
        id_b = otd_data["id_b"]

        if not otd_data["FP_data"].empty:
            table_name = f"{table_prefix}repeat_fps"
            counted_FPs = self._prepare_otd_table(
                spark, otd_data["FP_data"], id_a, id_b
            )
            counted_FPs.write.mode("overwrite").saveAsTable(table_name)
            print(
                f"A table of false positives of length {counted_FPs.count()} was saved as '{table_name}' for analysis."
            )
        else:
            print("There were no false positives recorded.")

        if not otd_data["FN_data"].empty:
            table_name = f"{table_prefix}repeat_fns"
            counted_FNs = self._prepare_otd_table(
                spark, otd_data["FN_data"], id_a, id_b
            )
            counted_FNs.write.mode("overwrite").saveAsTable(table_name)
            print(
                f"A table of false negatives of length {counted_FNs.count()} was saved as '{table_name}' for analysis."
            )
        else:
            print("There were no false negatives recorded.")

        if not otd_data["TP_data"].empty:
            table_name = f"{table_prefix}repeat_tps"
            counted_TPs = self._prepare_otd_table(
                spark, otd_data["TP_data"], id_a, id_b
            )
            counted_TPs.write.mode("overwrite").saveAsTable(table_name)
            print(
                f"A table of true positives of length {counted_TPs.count()} was saved as '{table_name}' for analysis."
            )
        else:
            print("There were no true positives recorded.")

        if not otd_data["TN_data"].empty:
            table_name = f"{table_prefix}repeat_tns"
            counted_TNs = self._prepare_otd_table(
                spark, otd_data["TN_data"], id_a, id_b
            )
            counted_TNs.write.mode("overwrite").saveAsTable(table_name)
            print(
                f"A table of true negatives of length {counted_TNs.count()} was saved as '{table_name}' for analysis."
            )
        else:
            print("There were no true negatives recorded.")

    def _create_otd_data(self, id_a: str, id_b: str) -> dict[str, Any] | None:
        """Output Suspicious Data (OTD): used to check config to see if you should find sketchy training data that the models routinely mis-classify"""
        training_conf = str(self.task.training_conf)
        config = self.task.link_run.config

        if (
            "output_suspicious_TD" in config[training_conf]
            and config[training_conf]["output_suspicious_TD"]
        ):
            return {
                "FP_data": pd.DataFrame(),
                "FN_data": pd.DataFrame(),
                "TP_data": pd.DataFrame(),
                "TN_data": pd.DataFrame(),
                "id_a": id_a,
                "id_b": id_b,
            }
        else:
            return None


def _calc_mcc(TP: int, TN: int, FP: int, FN: int) -> float:
    """
    Given the counts of true positives (TP), true negatives (TN), false
    positives (FP), and false negatives (FN) for a model run, compute the
    Matthews Correlation Coefficient (MCC).
    """
    if (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))) != 0:
        mcc = ((TP * TN) - (FP * FN)) / (
            math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        )
    else:
        mcc = 0
    return mcc


def _calc_threshold_matrix(
    alpha_threshold: float | list[float], threshold_ratio: float | list[float]
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
    predictions: pyspark.sql.DataFrame, dep_var: str, otd_data: dict[str, Any] | None
) -> tuple[int, int, int, int]:

    TP = predictions.filter((predictions[dep_var] == 1) & (predictions.prediction == 1))
    TP_count = TP.count()

    FP = predictions.filter((predictions[dep_var] == 0) & (predictions.prediction == 1))
    FP_count = FP.count()

    # print(
    #   f"Confusion matrix -- true positives and false positivesTP {TP_count} FP {FP_count}"
    # )

    FN = predictions.filter((predictions[dep_var] == 1) & (predictions.prediction == 0))
    FN_count = FN.count()

    TN = predictions.filter((predictions[dep_var] == 0) & (predictions.prediction == 0))
    TN_count = TN.count()

    # print(
    #   f"Confusion matrix -- true negatives and false negatives: FN {FN_count}  TN {TN_count}"
    # )

    if otd_data:
        id_a = otd_data["id_a"]
        id_b = otd_data["id_b"]

        new_FP_data = FP.select(
            id_a, id_b, dep_var, "prediction", "probability"
        ).toPandas()
        otd_data["FP_data"] = pd.concat([otd_data["FP_data"], new_FP_data])

        new_FN_data = FN.select(
            id_a, id_b, dep_var, "prediction", "probability"
        ).toPandas()
        otd_data["FN_data"] = pd.concat([otd_data["FN_data"], new_FN_data])

        new_TP_data = TP.select(
            id_a, id_b, dep_var, "prediction", "probability"
        ).toPandas()
        otd_data["TP_data"] = pd.concat([otd_data["TP_data"], new_TP_data])

        new_TN_data = TN.select(
            id_a, id_b, dep_var, "prediction", "probability"
        ).toPandas()
        otd_data["TN_data"] = pd.concat([otd_data["TN_data"], new_TN_data])

    return TP_count, FP_count, FN_count, TN_count


def _get_aggregate_metrics(
    TP_count: int, FP_count: int, FN_count: int, TN_count: int
) -> tuple[float, float, float]:
    """
    Given the counts of true positives, false positives, false negatives, and
    true negatives for a model run, compute several metrics to evaluate the
    model's quality.

    Return a tuple of (precision, recall, Matthews Correlation Coefficient).
    """
    if (TP_count + FP_count) == 0:
        precision = np.nan
    else:
        precision = TP_count / (TP_count + FP_count)
    if (TP_count + FN_count) == 0:
        recall = np.nan
    else:
        recall = TP_count / (TP_count + FN_count)
    mcc = _calc_mcc(TP_count, TN_count, FP_count, FN_count)
    # print(f"XX Aggregates precision {precision} recall {recall}")
    return precision, recall, mcc


def _create_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "precision_test",
            "recall_test",
            "precision_train",
            "recall_train",
            "pr_auc",
            "test_mcc",
            "train_mcc",
            "model_id",
            "alpha_threshold",
            "threshold_ratio",
        ]
    )


def _append_results(
    thresholded_metrics_df: pd.DataFrame,
    results_df: pd.DataFrame,
    model_type: str,
    params: dict[str, Any],
) -> pd.DataFrame:
    # run.pop("type")
    #    print(f"appending results_df : {results_df}")

    new_desc = pd.DataFrame(
        {
            "model": [model_type],
            "parameters": [params],
            "alpha_threshold": [results_df["alpha_threshold"][0]],
            "threshold_ratio": [results_df["threshold_ratio"][0]],
            "precision_test_mean": [results_df["precision_test"].mean()],
            "precision_test_sd": [results_df["precision_test"].std()],
            "recall_test_mean": [results_df["recall_test"].mean()],
            "recall_test_sd": [results_df["recall_test"].std()],
            "pr_auc_mean": [results_df["pr_auc"].mean()],
            "pr_auc_sd": [results_df["pr_auc"].std()],
            "mcc_test_mean": [results_df["test_mcc"].mean()],
            "mcc_test_sd": [results_df["test_mcc"].std()],
            "precision_train_mean": [results_df["precision_train"].mean()],
            "precision_train_sd": [results_df["precision_train"].std()],
            "recall_train_mean": [results_df["recall_train"].mean()],
            "recall_train_sd": [results_df["recall_train"].std()],
            "mcc_train_mean": [results_df["train_mcc"].mean()],
            "mcc_train_sd": [results_df["train_mcc"].std()],
        },
    )

    thresholded_metrics_df = pd.concat(
        [thresholded_metrics_df, new_desc], ignore_index=True
    )

    return thresholded_metrics_df


def _print_thresholded_metrics_df(desc_df: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", None)
    print(
        desc_df.drop(
            [
                "recall_test_sd",
                "recall_train_sd",
                "precision_test_sd",
                "precision_train_sd",
            ],
            axis=1,
        ).iloc[-1]
    )
    print("\n")


def _load_thresholded_metrics_df_params(desc_df: pd.DataFrame) -> pd.DataFrame:
    params = [
        "maxDepth",
        "numTrees",
        "featureSubsetStrategy",
        "subsample",
        "minInstancesPerNode",
        "maxBins",
        "class_weight",
        "C",
        "kernel",
        "threshold",
        "maxIter",
    ]

    load_params = lambda j, param: j.get(param, np.nan)
    for param in params:
        desc_df[param] = desc_df["parameters"].apply(load_params, args=(param,))
    desc_df["class_weight"] = desc_df["class_weight"].apply(
        lambda x: str(x) if pd.notnull(x) else x
    )
    desc_df["parameters"] = desc_df["parameters"].apply(
        lambda t: str(t) if pd.notnull(t) else t
    )
    return desc_df


def _create_probability_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "parameters",
            "pr_auc_mean",
            "pr_auc_standard_deviation",
        ]
    )


def _create_thresholded_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "parameters",
            "alpha_threshold",
            "threshold_ratio",
            "precision_test_mean",
            "precision_test_sd",
            "recall_test_mean",
            "recall_test_sd",
            "mcc_test_mean",
            "mcc_test_sd",
            "precision_train_mean",
            "precision_train_sd",
            "recall_train_mean",
            "recall_train_sd",
            "pr_auc_mean",
            "pr_auc_sd",
            "mcc_train_mean",
            "mcc_train_sd",
        ]
    )
