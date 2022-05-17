
## Program Structure

There are 4 modules of the program. See documentation in each specific class for more information.

1) `scripts` -- This contains the code for all of the CLI (command line interface). It contains the entrypoint into the program as well as all of the commands the user can run. 
2) `configs` -- This contains the code for reading and parsing the program configurations.
3) `spark` -- This contains the code for the spark initialization and connection.
4) `linking` -- This contains the code for all of the linking tasks. There is a separate README.md file in this module to further describe it.

In addition to these 4 modules, the `setup.py` file at the top level of the repo contains the configurations for packaging up the program with pip.

## Developing Code
To set up a copy of this project for development,

1. Clone the repository.
2. Run `pip install -e .[dev]` in the root project directory. This should install all dependencies.

## Running Tests

To run the project's test suite, run `pytest hlink/tests` in the root project directory. Running all of the tests
can take a while, depending on your computer's hardware and setup. To run a subset of tests that test some but not
all of the core features, try `pytest hlink/tests -m quickcheck`. These tests should run much more quickly.

## Building the Scala Jar

To build the Scala jar, do

```
cd scala_jar
sbt assembly
```

Then move the scala jar over to the hlink directory with `mv target/scala-2.11/*.jar ../hlink/spark/jars`.

## Creating Sphinx Docs

To write out the sphinx docs to the `docs` folder for the GitHub pages site, run

```
cd sphinx-docs
make github
```

## Creating a New Version and GitHub Release

hlink follows a semantic versioning scheme, so its version looks like MAJOR.MINOR.PATCH, where
each of MAJOR, MINOR, and PATCH is a nonnegative integer. Each component can be greater than
9, so a version like 3.11.0 is valid. When bumping the version of hlink, consider the types of
changes that have been made.

- If the changes are bug fixes and/or small internal changes, increment PATCH.
- If the changes are more significant API or behavior changes that could break user code,
increment MINOR and set PATCH back to 0.
- If the changes are major and very likely to break user code, increment MAJOR and set
both MINOR and PATCH back to 0.

For example, if the current version is 3.2.0, a bug fix would bump the version to 3.2.1.
A minor change would bump the version to 3.3.0, and a major change would bump it to 4.0.0.

Here are the steps to follow when creating the new version.

- Decide on the new version number A.B.C, following the scheme above.
- Set the new version number in `setup.py`, in the call to `setup()`.
- Regenerate the Sphinx docs so that they show the correct hlink version number.
- After committing your changes, create a git tag `vA.B.C` and push it to GitHub.
- Finally, create a GitHub release for the tag and add change notes describing the important
changes that are part of the release.
