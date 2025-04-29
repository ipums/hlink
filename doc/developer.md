## Developing Code
To set up a copy of this project for development,

1. Clone the repository.
2. Run `pip install --upgrade pip` to ensure that you have a recent version of pip.
3. Run `pip install -e .[dev]` in the root project directory. This should install all development dependencies. If you are working with the XGBoost and/or LightGBM machine learning models, then you'll also need to install those extras, like  `pip install -e .[dev,xgboost,lightgbm]`.

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

Hlink includes a Scala jar which defines Scala user-defined functions for use with Spark.
To build the Scala jar, do

```
cd scala_jar
sbt assembly
```

Then move the built Scala jar over to the hlink directory with `mv target/scala-2.11/*.jar ../hlink/spark/jars`.

## Working with the Sphinx Docs

We use Sphinx to generate the hlink documentation at [hlink.docs.ipums.org](hlink.docs.ipums.org).
These docs live in the `sphinx-docs` directory as Markdown files, and a GitHub Actions
workflow automatically builds them with Sphinx on pushes to main. To manually generate
a local copy of the HTML documentation, run

```
cd sphinx-docs
make html
```

This will output the documentation to the \_build/html/ subdirectory of
sphinx-docs/ by default. To test out your changes without having to push to the
official site, Python's `http.server` module works nicely.

```
python -m http.server -d _build/html <port>
```

starts up an HTTP server running on port `<port>` on the local machine.

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
- Update sphinx-docs/changelog.md with the new version number and make sure that it is up to date.
- After committing your changes, create a git tag `vA.B.C` and push it to GitHub.
- Finally, create a GitHub release for the tag. This is intended for record-keeping
for developers, so it's fine to automatically generate the release notes. The user-
facing changelog is sphinx-docs/changelog.md.

## Deploying a New Version to PyPI

1) Make sure that the package is installed with dev dependencies: `pip install -e .[dev]`.
2) Run `python -m build`. This creates distribution files in the dist directory.
3) Run `twine upload dist/hlink-A.B.C.tar.gz dist/hlink-A.B.C-py3-none-any.whl`, where A.B.C is the version number of the software.
