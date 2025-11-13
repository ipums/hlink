The Python script `setup-spark.py` automates the installation and configuration of an experimental Spark cluster. It shows how to download and install and configure Spark with HDFS as the storage layer and Spark in stand-alone mode. It can install a single node or two nodes for experimentation with a real cluster. While running it creates a log of every step for later review, so you can see how the setup really works.

I used Claude Code to write the script based on detailed sets of step by step directions for athe manual setup process.

The script will install either Spark 3.5.7 or Spark 4.0.1. You will need 'sudo' to set up 'ssh' if it's not already configured, which should be fine as this is intended mainly for use on a local developer machine. It should work on most Linux flavors and within a WSL2 VM.

See `--help` for more information. You probably should use `uv` to run the script but it's only using the Python standard library so it technically won't need  it's own virtual env.

There are directions on running a test "word count" program on the cluster including uploading data to HDFS. See WORDCOUNT_TUTORIAL.md. It includes some good basics on putting data on to HDFS and using HDFS.


