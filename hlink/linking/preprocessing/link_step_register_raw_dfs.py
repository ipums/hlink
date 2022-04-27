# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import os.path

from hlink.errors import DataError
from hlink.linking.link_step import LinkStep


def handle_paths(datasource, a_or_b):
    if "parquet_file" in datasource:
        path = os.path.realpath(datasource["parquet_file"])
        file_type = "parquet"
        return path, file_type
    elif "file" in datasource:
        file_str = datasource["file"]
        filename, file_extension = os.path.splitext(file_str)
        if file_extension == ".csv" or file_extension == ".parquet":
            path = os.path.realpath(datasource["file"])
            file_type = file_extension.strip(".")
            return path, file_type
        else:
            raise ValueError(
                f"The file given for datasource {a_or_b} must be either a CSV or parquet file.  You provided a {file_extension} file."
            )
    else:
        raise ValueError(
            f"You must specify either a parquet or csv file to be used as datasource {a_or_b}. This should be a property of 'datasource_{a_or_b}' in the config file."
        )


class LinkStepRegisterRawDfs(LinkStep):
    def __init__(self, task):
        super().__init__(
            task,
            "register raw dataframes",
            input_table_names=[],
            output_table_names=["raw_df_a", "raw_df_b"],
        )

    def _run(self):
        config = self.task.link_run.config
        path_a, file_type_a = handle_paths(config["datasource_a"], "a")
        path_b, file_type_b = handle_paths(config["datasource_b"], "b")

        self._load_unpartitioned(file_type_a, "_a", path_a)
        self._load_unpartitioned(file_type_b, "_b", path_b)

        self.task.run_register_python(
            name="raw_df_a",
            func=lambda: self._filter_dataframe(config, "a"),
            persist=True,
        )
        self.task.run_register_python(
            name="raw_df_b",
            func=lambda: self._filter_dataframe(config, "b"),
            persist=True,
        )

        self._check_for_all_spaces_unrestricted_file("raw_df_a")
        self._check_for_all_spaces_unrestricted_file("raw_df_b")

    def _load_unpartitioned(self, file_type, a_or_b, path):
        if file_type == "parquet":
            self.task.run_register_python(
                "raw_df_unpartitioned" + a_or_b,
                lambda: self.task.spark.read.parquet(path),
            )
        elif file_type == "csv":
            self.task.run_register_python(
                "raw_df_unpartitioned" + a_or_b,
                lambda: self.task.spark.read.csv(path, header=True, inferSchema=True),
            )
        else:
            raise ValueError(
                f"{file_type} is not a valid file type for this operation."
            )

    def _filter_dataframe(self, config, a_or_b):
        spark = self.task.spark
        table_name = f"raw_df_unpartitioned_{a_or_b}"
        filtered_df = spark.table(table_name)
        if "filter" in config:
            for dataset_filter in config["filter"]:
                if "expression" in dataset_filter:
                    filter_expression = dataset_filter["expression"]
                    if (
                        "datasource" not in dataset_filter
                        or dataset_filter["datasource"] == a_or_b
                    ):
                        if (
                            "household" in dataset_filter
                            and dataset_filter["household"]
                        ):
                            serial_a = dataset_filter["serial_a"]
                            serial_b = dataset_filter["serial_b"]
                            ser = serial_a if a_or_b == "a" else serial_b
                            serials_df = (
                                filtered_df.filter(filter_expression)
                                .select(ser)
                                .distinct()
                            )
                            filtered_df = filtered_df.join(serials_df, on=[ser])
                        else:
                            filtered_df = filtered_df.filter(filter_expression)
                elif "training_data_subset" in dataset_filter:
                    if dataset_filter["training_data_subset"]:
                        if (
                            "datasource" not in dataset_filter
                            or dataset_filter["datasource"] == a_or_b
                        ):
                            if "training_data" not in str(
                                self.task.spark.catalog.listTables()
                            ):
                                self.task.run_register_python(
                                    "training_data",
                                    lambda: self.task.spark.read.csv(
                                        config["training"]["dataset"],
                                        header=True,
                                        inferSchema=True,
                                    ),
                                    persist=True,
                                )
                            filtered_df.createOrReplaceTempView("temp_filtered_df")
                            filtered_df = self.task.run_register_sql(
                                name=None,
                                template="training_data_subset",
                                t_ctx={
                                    "table_name": "temp_filtered_df",
                                    "a_or_b": a_or_b,
                                    "id": config["id_column"],
                                },
                            )
                            spark.catalog.dropTempView("temp_filtered_df")
                        else:
                            pass
                    else:
                        pass
                else:
                    raise ValueError(f"Invalid filter: {dataset_filter}")
        return filtered_df

    def _check_for_all_spaces_unrestricted_file(self, df_name):
        df = self.task.spark.table(df_name)
        col_types = dict(df.dtypes)
        string_cols = [name for name, type in col_types.items() if type == "string"]

        space_columns = []

        df_len = df.count()

        for column_name in string_cols:

            if ("name" in str.lower(column_name)) or (
                "street" in str.lower(column_name)
            ):
                if (
                    self.task.spark.sql(
                        f"SELECT count(*) from {df_name} where {column_name} rlike '^ +$'"
                    ).first()[0]
                    == df_len
                ):
                    space_columns.append(column_name)

        if space_columns:
            col_names = ", ".join(space_columns)
            raise DataError(
                f"The following columns in the {df_name} table contain data which consist of all spaces, as exported in unrestricted data files: {col_names}.\nPlease point to data files with restricted versions of the data in your configuration file."
            )
        else:
            pass
