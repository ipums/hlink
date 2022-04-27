from hlink.linking.link_run import LinkRun
from hlink.spark.factory import SparkFactory
from hlink.configs.load_config import load_conf_file
from hlink.scripts.lib.io import write_table_to_csv
from hlink.scripts.lib.conf_validations import analyze_conf


def main():
    # First let's create a LinkRun object. This will be the main way that we
    # interact with hlink. To create a LinkRun, we need to read in our config
    # file and set up spark.
    print("=== Loading config file")
    config = load_conf_file("tutorial_config.toml")
    print("=== Setting up spark")
    # Create a SparkSession. Connect to the local machine, simulating a cluster.
    spark = SparkFactory().set_local().set_num_cores(4).create()

    print("=== Creating the LinkRun")
    link_run = LinkRun(spark, config)

    # Now we've got the LinkRun created. Let's analyze our config file to look
    # for errors that could cause hlink to fail.
    print("=== Analyzing config file")
    analyze_conf(link_run)

    # Alright! Our config file looks good to go. Let's run the steps we need.
    # Since we're not using machine learning for our linking, this is fairly
    # simple. First we'll preprocess the data and load it into spark by running
    # all of the steps in the preprocessing link task.
    print("=== Running preprocessing")
    link_run.preprocessing.run_all_steps()

    # Now let's do the matching. We only need steps 0 and 1 because the last step
    # is only applicable when we're using machine learning.
    print("=== Running first two matching steps")
    link_run.matching.run_step(0)
    link_run.matching.run_step(1)

    # The matching task saves its output in the potential_matches spark table.
    # Let's output this table to a CSV so that we can read it in later and look
    # at our results!
    print("=== Saving potential matches to potential_matches.csv")
    write_table_to_csv(link_run.spark, "potential_matches", "potential_matches.csv")


if __name__ == "__main__":
    main()
