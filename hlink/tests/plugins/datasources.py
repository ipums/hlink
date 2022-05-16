# This file is part of the ISRDI's hlink.
# For copyright and licensing information, see the NOTICE and LICENSE files
# in this project's top-level directory, and also on-line at:
#   https://github.com/ipums/hlink

import hlink.tests
import pytest
import os
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType
from pyspark.sql import Row


@pytest.fixture(scope="session")
def base_datasources(spark, tmpdir_factory):
    """Create a fixture for conf datasource input.  These test data are suitable for use in most of the preprocessing tests, and include really messy names for testing some name cleaning transforms, as well as bpl, age, serialp, and sex data."""
    datasources = tmpdir_factory.mktemp("datasources")
    # Create the first spark dataframe with test data and save it as parquet
    df_schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("serialp", StringType(), True),
            StructField("namelast", StringType(), True),
            StructField("namefrst", StringType(), True),
            StructField("namemiddle", StringType(), True),
            StructField("bpl", LongType(), True),
            StructField("sex", LongType(), True),
            StructField("age", LongType(), True),
        ]
    )
    data_a = [
        {
            "id": 10,
            "serialp": "A",
            "namelast": "",
            "namefrst": " John_M ",
            "bpl": 100,
            "sex": 1,
            "age": 23,
        },
        {
            "id": 20,
            "serialp": "B",
            "namelast": "Mc Last",
            "namefrst": "J  Marc'ell III",
            "bpl": 200,
            "sex": 2,
            "age": 30,
        },
        {
            "id": 30,
            "serialp": "B",
            "namelast": "L.T.",
            "namefrst": "Mr. Jon Jr.",
            "bpl": 300,
            "sex": 1,
        },
    ]
    pathname_a = os.path.join(datasources, "df1.parquet")
    spark.createDataFrame(data_a, schema=df_schema).write.parquet(pathname_a)

    # Create the second spark dataframe with test data and save it as parquet
    data_b = [
        {
            "id": 10,
            "serialp": "C",
            "namelast": "Name",
            "namefrst": "John?",
            "namemiddle": "M",
            "bpl": 400,
            "sex": 1,
        },
        {
            "id": 30,
            "serialp": "D",
            "namelast": None,
            "namemiddle": None,
            "bpl": 500,
            "sex": 0,
        },
        {
            "id": 50,
            "serialp": "E",
            "namefrst": "Je-an or Jeanie",
            "namemiddle": "Marc",
            "bpl": 700,
            "sex": 2,
        },
    ]
    pathname_b = os.path.join(datasources, "df2.parquet")
    spark.createDataFrame(data_b, schema=df_schema).write.parquet(pathname_b)
    return pathname_a, pathname_b


@pytest.fixture(scope="session")
def county_dist_datasources(spark, tmpdir_factory):
    """Create a fixture for conf datasource input.  These test data are suitable for use in testing county distance code calculation as well as the distance calculation itself."""
    datasources = tmpdir_factory.mktemp("datasources")
    # Create the first spark dataframe with test data and save it as parquet
    df_schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("namelast", StringType(), True),
            StructField("sex", LongType(), True),
            StructField("county_p", LongType(), True),
            StructField("statefip_p", LongType(), True),
        ]
    )
    data_a = [
        {"id": 10, "namelast": "Last", "sex": 0, "statefip_p": 3400, "county_p": 170},
        {"id": 20, "namelast": "Last", "sex": 0, "statefip_p": 5500, "county_p": 1210},
        {"id": 30, "namelast": "Lost", "sex": 0, "statefip_p": 1100, "county_p": 44999},
    ]
    pathname_a = os.path.join(datasources, "df1.parquet")
    spark.createDataFrame(data_a, schema=df_schema).write.parquet(pathname_a)

    # Create the second spark dataframe with test data and save it as parquet
    data_b = [
        {"id": 20, "namelast": "Last", "sex": 0, "statefip_p": 100, "county_p": 10},
        {"id": 40, "namelast": "Last", "sex": 0, "statefip_p": 1200, "county_p": 570},
    ]
    pathname_b = os.path.join(datasources, "df2.parquet")
    spark.createDataFrame(data_b, schema=df_schema).write.parquet(pathname_b)
    return pathname_a, pathname_b


@pytest.fixture(scope="function")
def datasource_preprocessing_simple_names(spark, conf, tmpdir_factory):
    """Synthetic data with name variants and sex data, designed for testing name substitution from gendered name files."""
    datasources = tmpdir_factory.mktemp("datasources")
    # Create the first spark dataframe with test data and save it as parquet
    df_schema = StructType(
        [
            StructField("id", StringType(), True),
            StructField("namefrst", StringType(), True),
            StructField("sex", LongType(), True),
        ]
    )
    data_a = [
        {"id": "10ah", "namefrst": "Cat", "sex": 2},
        {"id": "20bc", "namefrst": "Barney", "sex": 1},
        {"id": "34hi", "namefrst": "Cathy", "sex": 2},
        {"id": "54de", "namefrst": "Kat", "sex": 1},
    ]
    pathname_a = os.path.join(datasources, "df1.parquet")
    spark.createDataFrame(data_a, schema=df_schema).write.parquet(pathname_a)

    # Create the second spark dataframe with test data and save it as parquet
    data_b = [
        {"id": "c23", "namefrst": "Barry", "sex": 1},
        {"id": "d45", "namefrst": "Katie", "sex": 2},
        {"id": "e77", "namefrst": "Bernard", "sex": 1},
    ]
    pathname_b = os.path.join(datasources, "df2.parquet")
    spark.createDataFrame(data_b, schema=df_schema).write.parquet(pathname_b)
    return pathname_a, pathname_b


@pytest.fixture(scope="function")
def datasource_synthetic_households(spark, conf, tmpdir_factory):
    """This configuration includes data synthesized for testing the union feature on simple household/neighbors data."""
    datasources = tmpdir_factory.mktemp("datasources")
    # Create the first spark dataframe with test data and save it as parquet
    df_schema = StructType(
        [
            StructField("id", StringType(), True),
            StructField("namelast", StringType(), True),
            StructField("namefrst", StringType(), True),
            StructField("neighbors", ArrayType(StringType()), True),
            StructField("nonfamily_household", ArrayType(StringType()), True),
        ]
    )
    data_a = [
        {
            "id": "10ah",
            "namefrst": "jane",
            "neighbors": ["edie", "gerald"],
            "nonfamily_household": ["elmer"],
        },
        {
            "id": "20bc",
            "namefrst": "judy",
            "neighbors": ["edie", "elmer"],
            "nonfamily_household": [],
        },
        {
            "id": "34hi",
            "namefrst": "janice",
            "neighbors": ["edie"],
            "nonfamily_household": ["edie"],
        },
    ]
    pathname_a = os.path.join(datasources, "df1.parquet")
    spark.createDataFrame(data_a, schema=df_schema).write.parquet(pathname_a)

    # Create the second spark dataframe with test data and save it as parquet
    data_b = [
        {"id": "c23", "neighbors": [], "nonfamily_household": []},
        {
            "id": "d45",
            "namefrst": "gary",
            "neighbors": [],
            "nonfamily_household": ["colleen"],
        },
    ]
    pathname_b = os.path.join(datasources, "df2.parquet")
    spark.createDataFrame(data_b, schema=df_schema).write.parquet(pathname_b)
    return pathname_a, pathname_b


@pytest.fixture(scope="function")
def datasource_real_households(spark, conf, tmpdir_factory):
    """This configuration includes datasets for testing addition of household and neighbors features.  It's pulled from a sample of actual census data."""

    path_a = "input_data/training_data_households.parquet"
    path_b = "input_data/households_b.parquet"

    package_path = os.path.dirname(hlink.tests.__file__)

    full_path_a = os.path.join(package_path, path_a)
    full_path_b = os.path.join(package_path, path_b)
    return full_path_a, full_path_b


@pytest.fixture(scope="function")
def datasource_19thc_nativity_households_data(spark, conf):
    path_a = "input_data/19thc_nativity_test_hhs_a.csv"
    path_b = "input_data/19thc_nativity_test_hhs_b.csv"

    package_path = os.path.dirname(hlink.tests.__file__)

    full_path_a = os.path.join(package_path, path_a)
    full_path_b = os.path.join(package_path, path_b)

    return full_path_a, full_path_b


@pytest.fixture(scope="function")
def datasource_calc_mfbpl_pm_data(spark, conf):
    path_a = "input_data/calc_mfbpl_a.csv"
    path_b = "input_data/calc_mfbpl_b.csv"

    package_path = os.path.dirname(hlink.tests.__file__)

    full_path_a = os.path.join(package_path, path_a)
    full_path_b = os.path.join(package_path, path_b)

    return full_path_a, full_path_b


@pytest.fixture(scope="function")
def datasource_matching(spark, conf, matching):
    """Create the prepped_df_(a/b) dataframes for testing matching steps"""

    # Create the first spark dataframe with test data and save it as parquet
    df_schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("serialp", StringType(), True),
            StructField("namefrst", StringType(), True),
            StructField("namelast", StringType(), True),
            StructField("bpl", LongType(), True),
            StructField("sex", LongType(), True),
            StructField("street", StringType(), True),
            StructField("enum_dist", LongType(), True),
        ]
    )
    data_a = [
        {
            "id": 10,
            "serialp": "A",
            "namefrst": "Firste",
            "namelast": "Named",
            "bpl": 100,
            "sex": 1,
            "street": "First Avenue",
            "enum_dist": 0,
        },
        {
            "id": 20,
            "serialp": "B",
            "namefrst": "Firt",
            "namelast": "Last",
            "bpl": 200,
            "sex": 2,
            "street": "First Ave",
            "enum_dist": 0,
        },
        {
            "id": 30,
            "serialp": "B",
            "namefrst": "Frost",
            "namelast": "Lest",
            "bpl": 300,
            "sex": 2,
            "street": "Lyndale",
            "enum_dist": 2,
        },
    ]
    matching.run_register_python(
        "prepped_df_a",
        lambda: spark.createDataFrame(data_a, schema=df_schema),
        persist=True,
        overwrite_preexisting_tables=True,
    )

    # Create the second spark dataframe with test data and save it as parquet
    data_b = [
        {
            "id": 10,
            "serialp": "C",
            "namefrst": "First",
            "namelast": "Nameish",
            "bpl": 400,
            "sex": 1,
            "street": "First Ave",
            "enum_dist": 0,
        },
        {
            "id": 30,
            "serialp": "D",
            "namefrst": "Firt",
            "namelast": "Last",
            "bpl": 500,
            "sex": 2,
            "street": "1st Avenue",
            "enum_dist": 1,
        },
        {
            "id": 50,
            "serialp": "E",
            "namefrst": "Frst",
            "namelast": "List",
            "bpl": 700,
            "sex": 2,
            "street": "Franklin",
            "enum_dist": 2,
        },
    ]
    matching.run_register_python(
        "prepped_df_b",
        lambda: spark.createDataFrame(data_b, schema=df_schema),
        persist=True,
        overwrite_preexisting_tables=True,
    )


@pytest.fixture(scope="function")
def datasource_matching_comparisons(spark, conf, matching):
    """Create the prepped_df_(a/b) dataframes for testing matching comparison steps"""

    # Create the first spark dataframe with test data and save it as parquet
    df_schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("sex", LongType(), True),
            StructField("namelast", StringType(), True),
            StructField("mbpl", LongType(), True),
            StructField("mother_birthyr", LongType(), True),
            StructField("stepmom", LongType(), True),
            StructField("spouse_bpl", LongType(), True),
            StructField("spouse_birthyr", LongType(), True),
            StructField("durmarr", LongType(), True),
            StructField("mother_namefrst", StringType(), True),
            StructField("spouse_namefrst", StringType(), True),
            StructField("momloc", LongType(), True),
            StructField("sploc", LongType(), True),
        ]
    )
    data_a = [
        {
            "id": 10,
            "sex": 0,
            "namelast": "Last",
            "mbpl": 100,
            "mother_birthyr": 1925,
            "stepmom": 0,
            "spouse_bpl": 200,
            "spouse_birthyr": 1955,
            "durmarr": 2,
            "mother_namefrst": "eliza",
            "spouse_namefrst": "first",
            "momloc": 0,
            "sploc": 2,
        },
        {
            "id": 20,
            "sex": 0,
            "namelast": "Last",
            "mbpl": 100,
            "mother_birthyr": 1925,
            "stepmom": 0,
            "spouse_bpl": 200,
            "spouse_birthyr": 1955,
            "durmarr": 2,
            "mother_namefrst": "ellie",
            "spouse_namefrst": "frst",
            "momloc": 1,
            "sploc": 2,
        },
        {
            "id": 30,
            "sex": 0,
            "namelast": "Last",
            "mbpl": 100,
            "mother_birthyr": 1925,
            "stepmom": 0,
            "spouse_bpl": 200,
            "spouse_birthyr": 1955,
            "durmarr": 2,
            "mother_namefrst": "elizabeth",
            "spouse_namefrst": "firsty",
            "momloc": 3,
            "sploc": 0,
        },
    ]
    matching.run_register_python(
        "prepped_df_a",
        lambda: spark.createDataFrame(data_a, schema=df_schema),
        persist=True,
        overwrite_preexisting_tables=True,
    )

    # Create the second spark dataframe with test data and save it as parquet
    data_b = [
        {
            "id": 10,
            "sex": 0,
            "namelast": "Last",
            "mbpl": 100,
            "mother_birthyr": 1925,
            "stepmom": 0,
            "spouse_bpl": 200,
            "spouse_birthyr": 1955,
            "durmarr": 12,
            "mother_namefrst": "eliza",
            "spouse_namefrst": "fast",
            "momloc": 2,
            "sploc": 0,
        },
        {
            "id": 20,
            "sex": 0,
            "namelast": "Last",
            "mbpl": 200,
            "mother_birthyr": 1925,
            "stepmom": 0,
            "spouse_bpl": 300,
            "spouse_birthyr": 1955,
            "durmarr": 12,
            "mother_namefrst": "eliza",
            "momloc": 2,
            "sploc": 0,
        },
        {
            "id": 30,
            "sex": 0,
            "namelast": "Last",
            "mbpl": 100,
            "mother_birthyr": 1935,
            "stepmom": 0,
            "spouse_bpl": 200,
            "spouse_birthyr": 1961,
            "durmarr": 12,
            "spouse_namefrst": "fist",
            "momloc": 0,
            "sploc": 1,
        },
        {
            "id": 40,
            "sex": 0,
            "namelast": "Last",
            "mbpl": 100,
            "mother_birthyr": 1935,
            "stepmom": 3,
            "spouse_bpl": 200,
            "spouse_birthyr": 1955,
            "durmarr": 4,
            "momloc": 0,
            "sploc": 0,
        },
    ]
    matching.run_register_python(
        "prepped_df_b",
        lambda: spark.createDataFrame(data_b, schema=df_schema),
        persist=True,
        overwrite_preexisting_tables=True,
    )


@pytest.fixture(scope="function")
def datasource_training(spark, conf, matching):
    """Create the prepped_df_(a/b) dataframes and populate basic config values"""

    # Create the first spark dataframe with test data and save it as table
    df_schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("serialp", StringType(), True),
            StructField("namelast", StringType(), True),
            StructField("bpl", LongType(), True),
            StructField("sex", LongType(), True),
            StructField("region", LongType(), True),
        ]
    )
    data_a = [
        {
            "id": 10,
            "serialp": "A",
            "namelast": "Name",
            "bpl": 100,
            "sex": 1,
            "region": 1,
        },
        {
            "id": 20,
            "serialp": "B",
            "namelast": "Last",
            "bpl": 200,
            "sex": 2,
            "region": 2,
        },
        {
            "id": 30,
            "serialp": "B",
            "namelast": "Lest",
            "bpl": 300,
            "sex": 2,
            "region": 2,
        },
    ]

    matching.run_register_python(
        "prepped_df_a",
        lambda: spark.createDataFrame(data_a, schema=df_schema),
        persist=True,
        overwrite_preexisting_tables=True,
    )

    # Create the second spark dataframe with test data and save it as table
    data_b = [
        {
            "id": 10,
            "serialp": "C",
            "namelast": "Nameish",
            "bpl": 400,
            "sex": 1,
            "region": 1,
        },
        {
            "id": 30,
            "serialp": "D",
            "namelast": "Last",
            "bpl": 500,
            "sex": 2,
            "region": 2,
        },
        {
            "id": 50,
            "serialp": "E",
            "namelast": "List",
            "bpl": 700,
            "sex": 2,
            "region": 2,
        },
    ]

    matching.run_register_python(
        "prepped_df_b",
        lambda: spark.createDataFrame(data_b, schema=df_schema),
        persist=True,
        overwrite_preexisting_tables=True,
    )


@pytest.fixture(scope="function")
def datasource_training_input(spark, conf, tmpdir_factory):
    """This configuration includes datasets for testing specification of input data for a batch training step."""
    training_data = "input_data/training_data_long.csv"
    prepped_a_data = "input_data/training_data_long_a.csv"
    prepped_b_data = "input_data/training_data_long_b.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    td_path = os.path.join(package_path, training_data)
    pa_path = os.path.join(package_path, prepped_a_data)
    pb_path = os.path.join(package_path, prepped_b_data)

    return td_path, pa_path, pb_path


@pytest.fixture(scope="function")
def datasource_rel_jw_input(spark):
    """Create tables for testing rel_jaro_winkler"""

    schema = StructType(
        [
            StructField("id", StringType(), True),
            StructField(
                "namefrst_related_rows_birthyr",
                ArrayType(
                    StructType(
                        [
                            StructField("namefrst_std", StringType(), True),
                            StructField("birthyr", LongType(), True),
                            StructField("sex", LongType(), True),
                        ]
                    ),
                    True,
                ),
                True,
            ),
            StructField(
                "namefrst_related_rows",
                ArrayType(
                    StructType(
                        [
                            StructField("namefrst_std", StringType(), True),
                            StructField("replaced_birthyr", LongType(), True),
                            StructField("sex", LongType(), True),
                        ]
                    ),
                    True,
                ),
                True,
            ),
        ]
    )

    table_a = spark.createDataFrame(
        [
            (
                0,
                [
                    Row(namefrst_std="martha", birthyr=1855, sex=2),
                    Row(namefrst_std="minnie", birthyr=1857, sex=2),
                    Row(namefrst_std="martin", birthyr=1859, sex=1),
                ],
                [
                    Row(namefrst_std="martha", replaced_birthyr=1855, sex=2),
                    Row(namefrst_std="minnie", replaced_birthyr=1857, sex=2),
                    Row(namefrst_std="martin", replaced_birthyr=1859, sex=1),
                ],
            )
        ],
        schema,
    )

    table_b = spark.createDataFrame(
        [
            (
                0,
                [Row(namefrst_std="martha", birthyr=1855, sex=2)],
                [Row(namefrst_std="martha", replaced_birthyr=1855, sex=2)],
            ),
            (
                1,
                [Row(namefrst_std="tanya", birthyr=1855, sex=2)],
                [Row(namefrst_std="tanya", replaced_birthyr=1855, sex=2)],
            ),
        ],
        schema,
    )

    return table_a, table_b


@pytest.fixture(scope="function")
def datasource_extra_children_input(spark):
    """Create tables for testing rel_children"""
    schema = StructType(
        [
            StructField("histid", LongType(), True),
            StructField("relate", LongType(), True),
            StructField(
                "namefrst_related_rows",
                ArrayType(
                    StructType(
                        [
                            StructField("histid", StringType(), True),
                            StructField("namefrst", StringType(), True),
                            StructField("birthyr", LongType(), True),
                            StructField("sex", LongType(), True),
                            StructField("relate", LongType(), True),
                        ]
                    ),
                    True,
                ),
                True,
            ),
        ]
    )

    table_a = spark.createDataFrame(
        [
            (
                0,
                101,  # head of household
                [
                    Row(
                        histid=1, namefrst="martha", birthyr=1855, sex=2, relate=201
                    ),  # age 45
                    Row(
                        histid=2, namefrst="minnie", birthyr=1887, sex=2, relate=301
                    ),  # age 13
                    Row(
                        histid=3, namefrst="martin", birthyr=1897, sex=1, relate=301
                    ),  # age 3
                ],
            ),
            (
                4,
                301,  # child in the household
                [
                    Row(
                        histid=5, namefrst="george", birthyr=1887, sex=1, relate=301
                    ),  # age 18
                    Row(
                        histid=6, namefrst="marty", birthyr=1897, sex=1, relate=301
                    ),  # age 3
                    Row(
                        histid=7, namefrst="jean", birthyr=1835, sex=2, relate=601
                    ),  # age 65, mother in law
                ],
            ),
            (
                7,
                601,
                [
                    Row(
                        histid=5, namefrst="george", birthyr=1887, sex=1, relate=301
                    ),  # age 18
                    Row(
                        histid=6, namefrst="marty", birthyr=1897, sex=1, relate=301
                    ),  # age 3
                    Row(histid=4, namefrst="joe", birthyr=1896, sex=1, relate=301),
                ],
            ),
            (17, 101, []),
        ],
        schema,
    )

    table_b = spark.createDataFrame(
        [
            (
                8,
                101,
                [
                    Row(histid=9, namefrst="martha", birthyr=1855, sex=2, relate=201),
                    Row(histid=10, namefrst="martin", birthyr=1897, sex=1, relate=301),
                ],
            ),
            (
                11,
                301,
                [
                    Row(histid=12, namefrst="marc", birthyr=1888, sex=1, relate=301),
                    Row(histid=13, namefrst="tanya", birthyr=1899, sex=2, relate=301),
                    Row(histid=14, namefrst="erik", birthyr=1902, sex=1, relate=301),
                ],
            ),
            (
                15,
                101,
                [Row(histid=16, namefrst="marty", birthyr=1888, sex=1, relate=101)],
            ),
        ],
        schema,
    )

    return table_a, table_b


@pytest.fixture(scope="function")
def matching_test_input(spark, conf, tmpdir_factory):
    """This configuration includes datasets for testing matching steps."""
    prepped_a_data = "input_data/matching_test_a.csv"
    prepped_b_data = "input_data/matching_test_b.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    pa_path = os.path.join(package_path, prepped_a_data)
    pb_path = os.path.join(package_path, prepped_b_data)

    schema = StructType(
        [
            StructField("id", StringType(), True),
            StructField("namefrst", StringType(), True),
            StructField("namelast", StringType(), True),
            StructField("birthyr", LongType(), True),
            StructField("sex", LongType(), True),
        ]
    )

    pdfa = spark.read.csv(pa_path, schema)
    pdfb = spark.read.csv(pb_path, schema)

    return pdfa, pdfb


@pytest.fixture(scope="function")
def datasource_mi_comparison(spark, conf):
    """Create the prepped_df_(a/b) dataframes and populate basic config values"""

    # Create the first spark dataframe with test data and save it as table
    df_schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("namefrst_mid_init", StringType(), True),
        ]
    )
    data_a = [
        {"id": 10, "namefrst_mid_init": "a"},
        {"id": 20, "namefrst_mid_init": "b"},
        {"id": 30},
    ]
    table_a = spark.createDataFrame(data_a, schema=df_schema)

    data_b = [
        {"id": 40, "namefrst_mid_init": "a"},
        {"id": 50, "namefrst_mid_init": ""},
        {"id": 60},
    ]

    table_b = spark.createDataFrame(data_b, schema=df_schema)

    return table_a, table_b


@pytest.fixture(scope="session")
def datasource_unrestricted_blank_columns(spark, tmpdir_factory):
    """Create a fixture for conf datasource input.  These test data are suitable for use in the preprocessing tests which check for all-space columns in unrestricted data file."""

    datasources = tmpdir_factory.mktemp("datasources")
    # Create the first spark dataframe with test data and save it as parquet
    df_schema = StructType(
        [
            StructField("id", LongType(), True),
            StructField("serialp", StringType(), True),
            StructField("namelast", StringType(), True),
            StructField("namefrst", StringType(), True),
            StructField("namemiddle", StringType(), True),
            StructField("bpl", LongType(), True),
            StructField("sex", LongType(), True),
            StructField("age", LongType(), True),
            StructField("street", StringType(), True),
        ]
    )
    data_a = [
        {
            "id": 10,
            "serialp": "A",
            "namelast": "        ",
            "namefrst": " John_M ",
            "bpl": 100,
            "sex": 1,
            "age": 23,
            "street": "        ",
        },
        {
            "id": 20,
            "serialp": "B",
            "namelast": "    ",
            "namefrst": "J  Marc'ell III",
            "bpl": 200,
            "sex": 2,
            "age": 30,
            "street": "        ",
        },
        {
            "id": 30,
            "serialp": "B",
            "namelast": "    ",
            "namefrst": "Mr. Jon Jr.",
            "bpl": 300,
            "sex": 1,
            "street": "        ",
        },
    ]
    pathname_a = os.path.join(datasources, "df1.parquet")
    spark.createDataFrame(data_a, schema=df_schema).write.parquet(pathname_a)

    # Create the second spark dataframe with test data and save it as parquet
    data_b = [
        {
            "id": 10,
            "serialp": "C",
            "namelast": "Name",
            "namefrst": "John?",
            "namemiddle": "M",
            "bpl": 400,
            "sex": 1,
        },
        {
            "id": 30,
            "serialp": "D",
            "namelast": None,
            "namemiddle": None,
            "bpl": 500,
            "sex": 0,
        },
        {
            "id": 50,
            "serialp": "E",
            "namefrst": "Je-an or Jeanie",
            "namemiddle": "Marc",
            "bpl": 700,
            "sex": 2,
        },
    ]
    pathname_b = os.path.join(datasources, "df2.parquet")
    spark.createDataFrame(data_b, schema=df_schema).write.parquet(pathname_b)
    return pathname_a, pathname_b


@pytest.fixture(scope="function")
def datasource_sql_condition_input(spark, conf, tmpdir_factory):
    """This configuration includes datasets for testing specification of input data for a batch training step."""
    prepped_a_data = "input_data/sql_condition_marst_warn_a.csv"
    prepped_b_data = "input_data/sql_condition_marst_warn_b.csv"
    potential_matches = "input_data/potential_matches_sql_condition_marst_warn.csv"

    package_path = os.path.dirname(hlink.tests.__file__)
    pa_path = os.path.join(package_path, prepped_a_data)
    pb_path = os.path.join(package_path, prepped_b_data)
    pm_path = os.path.join(package_path, potential_matches)

    return pa_path, pb_path, pm_path
