# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

from pyspark.ml.feature import (
    Imputer,
    StandardScaler,
    OneHotEncoder,
    VectorAssembler,
    Bucketizer,
)
import hlink.linking.transformers.interaction_transformer
import hlink.linking.transformers.float_cast_transformer


def generate_pipeline_stages(conf, ind_vars, tf, tconf):
    """Creates a Spark ML pipeline from the pipeline features.
    Parameters
    ----------
    conf: dictionary
        the base configuration dictionary
    ind_vars: list
        a list of columns that are going to be used as independent_vars
    tf: DataFrame
        a Spark DataFrame for the "training_features" table
    tconf: string
        the name of the training section of the configuration.
        Can either be "training" or "hh_training".

    Returns
    -------
    A Spark ML pipeline object.

    """
    pipeline_stages = []
    tf_cols = tf.columns
    all_cols = ind_vars + tf_cols

    # Get the input columns that will be used as inputs for pipeline features
    pipeline_feature_input_cols = get_pipeline_feature_input_cols(
        ind_vars, conf.get("pipeline_features")
    )
    pipeline_input_cols = ind_vars + pipeline_feature_input_cols
    cols_to_pass = list(
        (set(tf_cols) & set(ind_vars))
        | (set(tf_cols) & set(pipeline_feature_input_cols))
    )
    col_names_dict = dict(zip(all_cols, all_cols))

    (
        categorical_comparison_features,
        categorical_pipeline_features,
    ) = _calc_categorical_features(
        ind_vars,
        cols_to_pass,
        conf["comparison_features"],
        conf.get("pipeline_features"),
    )

    # Cast table columns to float
    dep_var = str(conf[tconf]["dependent_var"])
    # id_b = conf["id_column"] + "_b"
    # id_a = conf["id_column"] + "_a"
    # cols_to_float = list(set(tf_cols) - {id_a, id_b, dep_var})
    float_cast_transformer = (
        hlink.linking.transformers.float_cast_transformer.FloatCastTransformer(
            inputCols=cols_to_pass
        )
    )
    pipeline_stages.append(float_cast_transformer)

    # Impute null values for remaining non-null columns
    features_to_impute = []
    for x in cols_to_pass:
        if x in categorical_comparison_features or x == dep_var or "id" in x:
            continue
        else:
            features_to_impute.append(x)

    imputed_output_features = [x + "_imp" for x in features_to_impute]
    imputer = Imputer(
        inputCols=features_to_impute,
        strategy="mean",
        outputCols=imputed_output_features,
    )
    pipeline_stages.append(imputer)

    for x in features_to_impute:
        if x in col_names_dict.keys():
            col_names_dict[x] = x + "_imp"
    # feature_names = list((set(ind_vars) - set(features_to_impute)) | set(output_feature_names))

    if len(categorical_comparison_features) > 0:
        encoded_output_cols = [
            x + "_onehotencoded" for x in categorical_comparison_features
        ]
        encoder = OneHotEncoder(
            inputCols=categorical_comparison_features,
            outputCols=encoded_output_cols,
            handleInvalid="keep",
            dropLast=False,
        )
        # feature_names = list((set(feature_names) - set(categorical_comparison_features)) | set(encoded_output_cols))
        for x in categorical_comparison_features:
            if x in col_names_dict.keys():
                col_names_dict[x] = x + "_onehotencoded"
        pipeline_stages.append(encoder)

    if "pipeline_features" in conf:
        for x in conf["pipeline_features"]:
            if x["output_column"] in pipeline_input_cols:
                if x["transformer_type"] == "bucketizer":
                    splits = x["splits"]
                    if x["input_column"] in col_names_dict.keys():
                        input_col = col_names_dict[x["input_column"]]
                    else:
                        input_col = x["input_column"]
                    bucketizer = Bucketizer(
                        splits=splits, inputCol=input_col, outputCol=x["output_column"]
                    )
                    pipeline_stages.append(bucketizer)

                elif x["transformer_type"] == "interaction":
                    input_cols = []
                    for key in x["input_columns"]:
                        if key in col_names_dict.keys():
                            input_cols.append(col_names_dict[key])
                        else:
                            input_cols.append(key)
                    interaction_transformer = hlink.linking.transformers.interaction_transformer.InteractionTransformer(
                        inputCols=input_cols, outputCol=x["output_column"]
                    )
                    pipeline_stages.append(interaction_transformer)
            else:
                continue

    if len(categorical_pipeline_features) > 0:
        encoded_output_cols = [
            x + "_onehotencoded" for x in categorical_pipeline_features
        ]
        encoder = OneHotEncoder(
            inputCols=categorical_pipeline_features,
            outputCols=encoded_output_cols,
            handleInvalid="keep",
            dropLast=False,
        )
        # feature_names = list((set(feature_names) - set(categorical_pipeline_features)) | set(encoded_output_cols))
        for x in categorical_pipeline_features:
            if x in col_names_dict.keys():
                col_names_dict[x] = x + "_onehotencoded"
        pipeline_stages.append(encoder)

    vec_cols = []
    for col in ind_vars:
        if col in col_names_dict.keys():
            vec_cols.append(col_names_dict[col])
        else:
            vec_cols.append(col)

    scale_data = conf[tconf].get("scale_data", False)
    output_col = "features_vector_prelim" if scale_data else "features_vector"
    vecAssembler = VectorAssembler(inputCols=vec_cols, outputCol=output_col)
    pipeline_stages.append(vecAssembler)
    if scale_data:
        scaler = StandardScaler(
            inputCol="features_vector_prelim", outputCol="features_vector"
        )
        pipeline_stages.append(scaler)
    return pipeline_stages


def _calc_categorical_features(
    ind_vars, cols_to_pass, comparison_features, pipeline_features, for_hh=False
):
    categorical_comparison_features = []
    categorical_pipeline_features = []
    cols = set(cols_to_pass + ind_vars)

    # Check for categorical features in all comparison features
    for x in comparison_features:
        if x["alias"] in cols:
            if "categorical" in x.keys():
                categorical_comparison_features.append(x["alias"])
        else:
            continue

    # Check for categorical features in the pipeline-generated features (if exist)

    if pipeline_features is not None:
        for pipeline_feature in pipeline_features:
            if pipeline_feature["output_column"] in cols and pipeline_feature.get(
                "categorical", False
            ):
                categorical_pipeline_features.append(pipeline_feature["output_column"])

    return categorical_comparison_features, categorical_pipeline_features


def get_pipeline_feature_input_cols(ind_vars, pipeline_features):
    pipeline_feature_input_cols = []
    if pipeline_features is not None:
        for pipeline_feature in pipeline_features:
            if pipeline_feature["output_column"] in ind_vars:
                if pipeline_feature.get("input_column", False):
                    pipeline_feature_input_cols.append(pipeline_feature["input_column"])
                else:
                    pipeline_feature_input_cols += pipeline_feature["input_columns"]
            else:
                continue
    return pipeline_feature_input_cols
