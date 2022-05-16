# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import itertools
import math
import re
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from pyspark.sql.functions import count, mean

import hlink.linking.core.threshold as threshold_core
import hlink.linking.core.classifier as classifier_core

from hlink.linking.link_step import LinkStep


class LinkStepTrainTestModels(LinkStep):
    def __init__(self, task):
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

    def _run(self):
        training_conf = str(self.task.training_conf)
        table_prefix = self.task.table_prefix
        config = self.task.link_run.config

        self.task.spark.sql("set spark.sql.shuffle.partitions=1")

        dep_var = config[training_conf]["dependent_var"]
        id_a = config["id_column"] + "_a"
        id_b = config["id_column"] + "_b"
        desc_df = _create_desc_df()
        columns_to_keep = [id_a, id_b, "features_vector", dep_var]
        prepped_data = (
            self.task.spark.table(f"{table_prefix}training_vectorized")
            .select(columns_to_keep)
            .cache()
        )

        otd_data = self._create_otd_data(id_a, id_b)

        n_training_iterations = config[training_conf].get("n_training_iterations", 10)
        seed = config[training_conf].get("seed", 2133)

        splits = self._get_splits(prepped_data, id_a, n_training_iterations, seed)

        model_parameters = self._get_model_parameters(config)
        for run in model_parameters:
            params = run.copy()
            model_type = params.pop("type")

            alpha_threshold = params.pop(
                "threshold", config[training_conf].get("threshold", 0.8)
            )
            if (
                config[training_conf].get("decision", False)
                == "drop_duplicate_with_threshold_ratio"
            ):
                threshold_ratio = params.pop(
                    "threshold_ratio",
                    threshold_core.get_threshold_ratio(config[training_conf], params),
                )
            else:
                threshold_ratio = False

            threshold_matrix = _calc_threshold_matrix(alpha_threshold, threshold_ratio)
            results_dfs = {}
            for i in range(len(threshold_matrix)):
                results_dfs[i] = _create_results_df()

            first = True
            for training_data, test_data in splits:
                training_data.cache()
                test_data.cache()

                classifier, post_transformer = classifier_core.choose_classifier(
                    model_type, params, dep_var
                )

                model = classifier.fit(training_data)

                predictions_tmp = _get_probability_and_select_pred_columns(
                    test_data, model, post_transformer, id_a, id_b, dep_var
                ).cache()
                predict_train_tmp = _get_probability_and_select_pred_columns(
                    training_data, model, post_transformer, id_a, id_b, dep_var
                ).cache()

                test_pred = predictions_tmp.toPandas()
                precision, recall, thresholds_raw = precision_recall_curve(
                    test_pred[f"{dep_var}"],
                    test_pred["probability"].round(2),
                    pos_label=1,
                )

                thresholds_plus_1 = np.append(thresholds_raw, [np.nan])
                param_text = np.full(precision.shape, model_type + "_" + str(params))

                pr_auc = auc(recall, precision)
                print(f"Area under PR curve: {pr_auc}")

                if first:
                    prc = pd.DataFrame(
                        {
                            "params": param_text,
                            "precision": precision,
                            "recall": recall,
                            "threshold_gt_eq": thresholds_plus_1,
                        }
                    )
                    self.task.spark.createDataFrame(prc).write.mode(
                        "overwrite"
                    ).saveAsTable(
                        f"{self.task.table_prefix}precision_recall_curve_"
                        + re.sub("[^A-Za-z0-9]", "_", model_type + str(params))
                    )

                    first = False

                i = 0
                for at, tr in threshold_matrix:
                    predictions = threshold_core.predict_using_thresholds(
                        predictions_tmp,
                        at,
                        tr,
                        config[training_conf],
                        config["id_column"],
                    )
                    predict_train = threshold_core.predict_using_thresholds(
                        predict_train_tmp,
                        at,
                        tr,
                        config[training_conf],
                        config["id_column"],
                    )

                    results_dfs[i] = self._capture_results(
                        predictions,
                        predict_train,
                        dep_var,
                        model,
                        results_dfs[i],
                        otd_data,
                        at,
                        tr,
                        pr_auc,
                    )
                    i += 1

                training_data.unpersist()
                test_data.unpersist()

            for i in range(len(threshold_matrix)):
                desc_df = _append_results(desc_df, results_dfs[i], model_type, params)

        _print_desc_df(desc_df)
        desc_df = _load_desc_df_params(desc_df)
        self._save_training_results(desc_df, self.task.spark)
        self._save_otd_data(otd_data, self.task.spark)
        self.task.spark.sql("set spark.sql.shuffle.partitions=200")

    def _get_splits(self, prepped_data, id_a, n_training_iterations, seed):
        if self.task.link_run.config[f"{self.task.training_conf}"].get(
            "split_by_id_a", False
        ):
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
            splits = [
                prepped_data.randomSplit([0.5, 0.5], seed=seed + i)
                for i in range(n_training_iterations)
            ]

        return splits

    def _custom_param_grid_builder(self, conf):
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
        predictions,
        predict_train,
        dep_var,
        model,
        results_df,
        otd_data,
        at,
        tr,
        pr_auc,
    ):
        table_prefix = self.task.table_prefix

        print("Evaluating model performance...")
        # write to sql tables for testing
        predictions.createOrReplaceTempView(f"{table_prefix}predictions")
        predict_train.createOrReplaceTempView(f"{table_prefix}predict_train")

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
                "alpha_threshold": [at],
                "threshold_ratio": [tr],
            },
        )
        return pd.concat([results_df, new_results], ignore_index=True)

    def _get_model_parameters(self, conf):
        training_conf = str(self.task.training_conf)

        model_parameters = conf[training_conf]["model_parameters"]
        if "param_grid" in conf[training_conf] and conf[training_conf]["param_grid"]:
            model_parameters = self._custom_param_grid_builder(conf)
        elif model_parameters == []:
            raise ValueError(
                "No model parameters found. In 'training' config, either supply 'model_parameters' or 'param_grid'."
            )
        return model_parameters

    def _save_training_results(self, desc_df, spark):
        table_prefix = self.task.table_prefix

        if desc_df.empty:
            print("Training results dataframe is empty.")
        else:
            desc_df.dropna(axis=1, how="all", inplace=True)
            spark.createDataFrame(desc_df, samplingRatio=1).write.mode(
                "overwrite"
            ).saveAsTable(f"{table_prefix}training_results")
            print(
                f"Training results saved to Spark table '{table_prefix}training_results'."
            )

    def _prepare_otd_table(self, spark, df, id_a, id_b):
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

    def _save_otd_data(self, otd_data, spark):
        table_prefix = self.task.table_prefix

        if otd_data is None:
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

    def _create_otd_data(self, id_a, id_b):
        """Output Suspicous Data (OTD): used to check config to see if you should find sketchy training data that the models routinely mis-classify"""
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


def _calc_mcc(TP, TN, FP, FN):
    if (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))) != 0:
        mcc = ((TP * TN) - (FP * FN)) / (
            math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        )
    else:
        mcc = 0
    return mcc


def _calc_threshold_matrix(alpha_threshold, threshold_ratio):
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
    pred_df, model, post_transformer, id_a, id_b, dep_var
):
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


def _get_confusion_matrix(predictions, dep_var, otd_data):
    TP = predictions.filter((predictions[dep_var] == 1) & (predictions.prediction == 1))
    TP_count = TP.count()

    FP = predictions.filter((predictions[dep_var] == 0) & (predictions.prediction == 1))
    FP_count = FP.count()

    FN = predictions.filter((predictions[dep_var] == 1) & (predictions.prediction == 0))
    FN_count = FN.count()

    TN = predictions.filter((predictions[dep_var] == 0) & (predictions.prediction == 0))
    TN_count = TN.count()

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


def _get_aggregate_metrics(TP_count, FP_count, FN_count, TN_count):
    if (TP_count + FP_count) == 0:
        precision = np.nan
    else:
        precision = TP_count / (TP_count + FP_count)
    if (TP_count + FN_count) == 0:
        recall = np.nan
    else:
        recall = TP_count / (TP_count + FN_count)
    mcc = _calc_mcc(TP_count, TN_count, FP_count, FN_count)
    return precision, recall, mcc


def _create_results_df():
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


def _append_results(desc_df, results_df, model_type, params):
    # run.pop("type")
    print(results_df)

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

    desc_df = pd.concat([desc_df, new_desc], ignore_index=True)
    _print_desc_df(desc_df)
    return desc_df


def _print_desc_df(desc_df):
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


def _load_desc_df_params(desc_df):
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


def _create_desc_df():
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
