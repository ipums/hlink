
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

To run the project's test suite, run `pytest hlink/tests` in the root project directory.

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
