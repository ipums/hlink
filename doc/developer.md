
## Program Structure

There are 4 modules of the program. See documentation in each specific class for more information.

1) `scripts` -- This contains the code for all of the CLI (command line interface). It contains the entrypoint into the program as well as all of the commands the user can run. 
2) `configs` -- This contains the code for reading and parsing the program configurations.
3) `spark` -- This contains the code for the spark initialization and connection.
4) `linking` -- This contains the code for all of the linking tasks. There is a separate README.md file in this module to further describe it.

In addition to these 4 modules, the `pyproject.toml` file at the top level of the repo contains the configurations for packaging up the program with pip.

## Developing Code
To set up a copy of this project for development,

1. Clone the repository.
2. Run `pip install --upgrade pip` to ensure that you have a recent version of pip.
3. Run `pip install -e .[dev]` in the root project directory. This should install all dependencies.

## Running Tests

To run the project's test suite, run `pytest` in the root project directory.
Running all of the tests can take a while, depending on your computer's
hardware and setup. If you are working on a particular bug or feature, there
are several good ways to filter the tests to run just tests that interest you.
Check out the pytest documentation
[here](https://docs.pytest.org/en/latest/how-to/usage.html#specifying-which-tests-to-run).

In particular, the `-k` argument is helpful for running only tests with names
that match the topics you are interested in, like this:

```
pytest -k "lightgbm or xgboost"
```

The GitHub Actions workflow runs all of the tests on each push or PR to the
main branch. It runs the tests on several versions of Python and in several
different Python environments.

## Building the Scala Jar

To build the Scala jar, do

```
cd scala_jar
sbt assembly
```

Then move the scala jar over to the hlink directory with `mv target/scala-2.11/*.jar ../hlink/spark/jars`.

## Working with the Sphinx Docs

We use Sphinx to generate the hlink documentation at [hlink.docs.ipums.org](hlink.docs.ipums.org).
These docs live in the `sphinx-docs` directory as Markdown files, and Sphinx converts them to html
files that live in the `docs` directory. To write out the Sphinx docs to the `docs` folder, run

```
cd sphinx-docs
make
```

To test out your changes without having to push to the official site, Python's `http.server` module
works nicely.

```
cd docs
python -m http.server <port>
```

starts up an HTTP server running on port `<port>` on your local machine. Visit `127.0.0.1:<port>`
in your browser to view the HTML.

## Creating a New Version and GitHub Release

hlink follows a semantic versioning scheme, so its version looks like MAJOR.MINOR.PATCH, where
each of MAJOR, MINOR, and PATCH is a nonnegative integer. Each component can be greater than
9, so a version like 3.11.0 is valid. When bumping the version of hlink, consider the types of
changes that have been made.

- If the changes are bug fixes and/or small internal changes, increment PATCH.
- If the changes are additions to the API that should not break user code, then
increment MINOR and set PATCH back to 0.
- If the changes are major and likely to break user code, increment MAJOR and set
both MINOR and PATCH back to 0.

For example, if the current version is 3.2.1, a bug fix would bump the version to 3.2.2.
A minor change would bump the version to 3.3.0, and a major change would bump it to 4.0.0.

Here are the steps to follow when creating the new version.

- Decide on the new version number A.B.C, following the scheme above.
- Set the new version number in `pyproject.toml`.
- Reinstall hlink with `pip install -e .[dev]` to update the version. Confirm that this worked by running `hlink --version`.
- Regenerate the Sphinx docs so that they show the correct hlink version number.
- After committing your changes, create a git tag `vA.B.C` and push it to GitHub.
- Finally, create a GitHub release for the tag and add change notes describing the important
changes that are part of the release.

## Deploying a new version to pypi

1) Make sure that the package is installed with dev dependencies: `pip install -e .[dev]`.
2) Run: `python -m build`. This creates a hlink-x.x.x.tar.gz file in the dist directory.
3) Run: `twine upload dist/hlink-x.x.x.tar.gz` where x.x.x is the version number of the software.
