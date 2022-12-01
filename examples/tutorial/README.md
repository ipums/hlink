# hlink Tutorial

This is an example linking project that uses hlink to link people between two
very small example datasets, data/A.csv and data/B.csv. After installing hlink,
the tutorial can be run as `python tutorial.py` in this directory. This will
perform the matching and generate a list of potential matches between dataset A
and dataset B. These potential matches are printed to the screen and are also
saved to the file potential\_matches.csv.

## Dataset Overview

Each of the datasets has the following columns:
- ID (unique numeric identifier)
- NAMEFRST (first name)
- NAMELAST (last name)
- AGE
- SEX

These datasets are fictional example datasets. Each record in the datasets
represents a single person at a point in time. Dataset A models data sampled 10
years before dataset B's data was sampled. So we would expect someone aged 40 in
dataset A to be aged 50 in dataset B.

## The Config File and Linking Strategy

To link these two datasets, we need a configuration file that directs
hlink on what operations to perform and how to determine when a link is found. For
our tutorial example, we'll use deterministic linking, but hlink is also capable
of using machine learning models to classify possible links between the datasets.

In this section we'll walk through the process of creating the tutorial\_config.toml
file that can be found in this directory. Creating a config file can be complicated.
See the [hlink documentation](https://hlink.docs.ipums.org) for a detailed
explanation of the different config file sections and keys.

The first step in creating a config file is describing the data to hlink. The
`id_column` key tells hlink the name of the id column which uniquely identifies
each record in a dataset. In our case, this is "ID". The `datasource_a`
and `datasource_b` sections give hlink information about where to find the input
files. We give hlink the relative path to our data files in these sections. Each
column that we want to read from the dataset files into hlink must appear in a
`column_mappings` section. By default a `column_mappings` section reads in the
column unchanged, but it can also be used to perform some preprocessing and
cleaning on the column as it is read in. In our config file, we have hlink
lowercase names and strip leading and trailing whitespace to support comparability
between the datasets.

After describing the data to hlink, we need to think about our linking strategy.
How will we determine who links between the two datasets? Do we need to do any
data cleaning or reformatting to allow this? Are there any rules that exclude two
records from linking?

In our tutorial example, here is the general linking strategy that we'll use:
- First, we say that two records cannot link if they have a different SEX between
A and B.
- We say that two records may link only if the difference between AGE in A and
AGE in B is within 2 years of 10, so 8 to 12.
- Finally, we use the
[Jaro-Winkler string comparison algorithm](https://en.wikipedia.org/wiki/Jaro–Winkler_distance)
to compare each candidate link's NAMEFRST and NAMELAST between the two datasets.
If the names score sufficiently high, then we have a link!

The first two bullet points above correspond to the `blocking` section. In this
section, we separate records into different *blocks*. Then each record may link
only with other records in its block. In the case of SEX, this means that we split
the datasets roughly in two, creating two separate blocks where links may occur.

For AGE, we have a range of allowable differences. So we *explode* each record
in dataset A, creating five new records which go into 5 different blocks. The
five records have five different ages, AGE - 2, AGE - 1, AGE, AGE + 1, and
AGE + 2. Dataset B is blocked on age, and the records from dataset A go into
the block corresponding to their exact age. This allows us to do fuzzy matching
of ages; each record in dataset A is allowed to match with records in dataset B
with an AGE anywhere from 8 to 12 years greater than the AGE in dataset A.

The last bullet point corresponds to the `comparison_features` and `comparisons`
sections. In these sections, we tell hlink to compute the Jaro-Winkler score
between the NAMEFRST string in the dataset A record and the NAMEFRST string
in the dataset B record, then compare the score against a threshold of 0.79 to
determine if it's a link or not. We do the same thing for NAMELAST, with a
threshold of 0.84. If a single record pair reaches both thresholds, then we call
it a link! This pair of records will end up in `potential_matches.csv` when the
script completes.

Now that the config file is written, we can run hlink to generate some links. See
the next section for a description of the tutorial script that runs hlink.

## The Tutorial Script

The tutorial.py Python script contains code to load in the config file and run
hlink to generate potential matches between the two datasets. It creates a `LinkRun`,
which is the main way to control the hlink library. After analyzing the
config file for errors, it runs two link tasks: preprocessing and matching.

The preprocessing task reads in the data from the datasets and does the data
cleaning and column mapping that we've asked it to do for us in the config file.

The matching task does the real linking work, finding links between the two datasets.
It stores its results in a `potential_matches` Spark table. The script saves this
table to the `potential_matches.csv` file and prints it to the screen.

## Getting and Interpreting Results

After running the tutorial script, we have a `potential_matches.csv` file that
contains data on potential links that hlink identified between the two datasets.
Each record in this dataset identifies a potential link. The ID\_a and ID\_b
columns identify the records in dataset A and dataset B that have been linked.
There are also some more fields that are useful for reviewing the links and confirming
that they look reasonable. Some links may be more reasonable than others!

## Things to Try

- After running the tutorial script once, run it again. This time it should print
statements like `Preexisting table: raw_df_a`. If hlink finds that a Spark table
already exists when it goes to compute it, it will use the preexisting table
instead of recomputing it. To prevent this from happening, try passing the
`--clean` argument to tutorial.py. This will tell the script to drop all of the
preexisting tables before it runs the linking job.

- Try increasing or decreasing the Jaro-Winkler thresholds in the config file.
How does this affect the matches that are generated?
