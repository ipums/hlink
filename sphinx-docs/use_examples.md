# Advanced Workflow Examples 

## Export training data after generating features to reuse in different linking years

It is common to have a single training data set that spans two linked years, which is then used to train a model that is applied to a different set of linked years.  For example, we have a training data set that spans linked individuals from the 1900 census to the 1910 census.  We use this training data to predict links in the full count 1900-1910 linking run, but we also use this training data to link year pairs 1910-1920, 1920-1930, and 1930-1940.  

When this training data set is used for the years it was derived from, the only columns necessary are the HISTIDs identifying the individuals in the data and the dependent variable (usually a boolean `match` column) for the model training. Features for the machine learning model generation are created from the source data available in the full count run.  

However, when this training data set is used for other years, the program does not have access to the source full count files, and can't generate the ML features from the given data.  In this scenario, you would need to save a copy of the `training_features` and `hh_training_features` Spark tables to .csv so you can point to that in the other year pair runs, and indicate the `use_potential_matches_features = true` flag in both the `training` and `hh_training` sections of the configuration.

### Example training data export with generated ML features

1) Create a config file and put it in your hlink config directory.

2) Launch the hlink program in interactive mode:

    ```
    hlink --conf=full_count_1900_1910 --cores 50 --executor_memory 50G
    ```

3) Run the preprocessing and training link tasks:

    ```bash
    hlink $ run_all_steps preprocessing training
    ```

4) Ask the program what the arguments for the `csv` command are:

    ```bash
    hlink $ ? csv
    Writes a dataframe out to csv.
         Arg 1: dataframe
         Arg 2: path
         Arg 3 (optional): # of partitions
    ```
   
5) Export the results using the `csv` command: 

    ```bash
    hlink $ csv training_features /my/output/training_data_1900_1910_HLINK_FEATURES.csv
    ```

6) Continue with other linking work you might need to do with this year pair, otherwise shut down the hlink framework for this pair of linking years:

    ```bash
    hlink $ q
    ```

7) In the config file for the new year pairs (1910-1920, 1920-1930, etc.), point to this new file as your dataset, and set the `use_training_data_features`

    ```
    # config file for 1910-1920 linking run using the 1900-1910 training data with hlink-generated features
    [training]

    # more configs here...

    dataset = "/path/to/training_data_1900_1910_HLINK_FEATURES.csv"
    dependent_var = "match"

    # This needs to be changed to `true` to use the features we just generated
    use_training_data_features = true

    # configs continue here...
    ```

8) Launch the hlink program using your new config for the new year pair you want to link. Run your link tasks and export relevant data.

## An Example Model Exploration Workflow

`hlink` accepts a matrix of ML models and hyper-parameters to run train/test splits for you, and outputs data you can use to select and tune your models.  You can see example `training` and `hh_training` configuration sections that implement this in the [training](config.html#training-and-models) and [household training](config.html#household-training-and-models) sections of the configuration documentation.

1) Create a config file that has a `training` and/or `hh_training` section with model parameters to explore. For example:

    ```
    [training]

    independent_vars = ["race", "srace", "race_interacted_srace", "hits", "hits2", "exact_mult", "ncount", "ncount2", "region", "namefrst_jw","namelast_jw","namefrst_std_jw","byrdiff", "f_interacted_jw_f", "jw_f", "f_caution", "f_pres", "fbplmatch", "m_interacted_jw_m", "jw_m", "m_caution", "m_pres", "mbplmatch", "sp_interacted_jw_sp", "jw_sp", "sp_caution", "sp_pres", "mi", "fsoundex", "lsoundex", "rel", "oth", "sgen", "nbors", "county_distance", "county_distance_squared", "street_jw", "imm_interacted_immyear_caution", "immyear_diff", "imm"]

    scale_data = false
    dataset = "/path/to/training_data_1900_1910.csv"
    dependent_var = "match"

    # This would need to be changed to `true` in a run between other years if your
    # source data years weren't identical to the linked years of your training data.
    use_training_data_features = false

    split_by_id_a = true
    score_with_model = true
    feature_importances = false
    decision = "drop_duplicate_with_threshold_ratio"
    model_parameter_search = {strategy = "grid"}
    n_training_iterations = 10
    model_parameters = [
        { type = "logistic_regression", threshold = [0.5], threshold_ratio = [1.0, 1.1]},
        { type = "random_forest", maxDepth = [5, 6, 7], numTrees = [50, 75, 100], threshold = [0.5], threshold_ratio = [1.0, 1.1, 1.2]}
    ]

    # The chosen_model is the final selected model to use in the full count production
    # run. This is where you would manually update your config after running model
    # exploration and making decisions about your models and hyperparameters. This 
    # section isn't used by the model exploration task.
    chosen_model = { type = "logistic_regression", threshold = 0.5, threshold_ratio = 1.0 }
    ```

2) Launch the hlink program in interactive mode:

    ```bash
    hlink --conf=full_count_1900_1910 --cores 50 --executor_memory 50G
    ```

3) Run the preprocessing and model exploration link tasks:

    ```
    hlink $ run_all_steps preprocessing model_exploration
    ```

4) Export the results of the train/test split runs to csv for further analysis.  For `training` params, the results will be in the `training_results` table, and for `hh_training` in the `hh_training_results` table.

    ```
    hlink $ csv training_results /my/output/1900_1910_training_results.csv
    ```

5) Use your preferred methods to analyze the data you've just exported.  Update the `chosen_model` in your configuration, and/or create new versions of your training data following your findings and update the path to the new training data in your configs.
