# HLink Tutorial

This is an example linking project that uses hlink to link people between two
datasets, `data/A.csv` and `data/B.csv`. Note that these datasets are not
included, so the example script will throw an error if run out of the box.

## Dataset Overview

The tutorial script supposes that each of the datasets has the following columns:
id, NAMEFRST, NAMELAST, AGE, SEX. In addition, it supposes that dataset B was
created 10 years after dataset A. Each record in the datasets represents a single
person.

## The Config File and Linking Strategy

To link these two datasets, we need to create a configuration file that directs
hlink on what operations to perform and how to determine when a link is found. For
our tutorial example, we'll use deterministic linking, but hlink is also capable
of using machine learning models to classify possible links between the datasets.

Creating a config file is complicated. See the hlink documentation for a
more detailed explanation of the different config file sections and keys.

The first step in creating a config file is describing the data to hlink. The
`id_column` key tells hlink the name of the id column which uniquely identifies
each record in the databases. In our case, this is just "id". The `datasource_a`
and `datasource_b` sections give hlink information about where to find the input
files. We give hlink the relative path to our data files in these sections.

After describing the data to hlink, we need to think about our linking strategy.
How will we determine who links between the two datasets? Do we need to do any
data cleaning or reformatting to allow this? Are there any rules that exclude two
records from linking?

In our tutorial example, here is the general linking strategy that we'll use:
- First, we say that two records cannot link if they have a different SEX between
A and B.
- We say that two records may link only if the difference between AGE in A and
AGE in B is within 2 years of 10, so 8 to 12.
- Finally, we use the Jaro-Winkler string comparison algorithm to compare each
candidate link's NAMEFRST and NAMELAST between the two datasets. If the names score
sufficiently high, then we have a link!

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
between the NAMEFRST string in the dataset A record and the corresponding string
in the dataset B record, then compare the score against a threshold of 0.79 to
determine if it's a link or not. We do the same thing for NAMELAST, with a
threshold of 0.84. If a single record pair reaches both thresholds, then we call
it a link! This pair of records will end up in `potential_matches.csv` when the
script completes.

It's very likely that the names in dataset A and dataset B are not consistently
formatted. This is where the `column_mappings` section comes in. It tells hlink
to perform some data cleaning in the preprocessing step before matching occurs.
The column mappings in the config file strip whitespace from the names and lowercase
them to remove discrepancies in formatting between the two datasets.

Now that the config file is written, we can run hlink to generate some links. See
the next section for a description of the tutorial script that runs hlink.

## The Tutorial Script

The `tutorial.py` Python script contains code to load in the config file and run
hlink to generate potential links between the two datasets. It creates a `LinkRun`,
which is the main way to communicate with the hlink library. After analyzing the
config file for errors, it runs two link tasks: preprocessing and matching.

The preprocessing task reads the data from the datasets in and does the data
cleaning and column mapping that we've asked it to do for us in the config file.

The matching task does the real linking work, finding links between the two datasets.
It stores its results in a `potential_matches` spark table. The script saves this
table to the `potential_matches.csv` file.

## Getting and Interpreting Results

After running the tutorial script, we have a `potential_matches.csv` file that
contains data on potential links that hlink identified between the two datasets.
Each record in this dataset identifies a potential link. The id\_a and id\_b
columns identify the records in dataset A and dataset B that have been linked.
There are also some more fields that are useful for reviewing the links and confirming
that they look reasonable. Some links may be more reasonable than others!
Our linking strategy is deterministic and relatively simple, so it may catch
more or less links than another strategy.

