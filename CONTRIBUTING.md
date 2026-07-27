# Contributing to FLEX-DEPOT

Thank you for your interest in contributing. This document covers how to set up
the development environment, run linting and tests, and submit changes.

## Setting up

Python 3.10 or newer is required. Clone the repository and install the package
in editable mode with the development extras:

```bash
git clone https://github.com/TUMFTM/flex-depot.git
cd flex-depot
pip install -e ".[dev]"
```

The `[dev]` extras install [ruff](https://docs.astral.sh/ruff/) (linter) and
[pytest](https://docs.pytest.org/) (test runner). HiGHS is pulled in as a
regular dependency (`highspy`) and is used by the test suite.

## Running the tests

```bash
# Fast unit tests (no solver-heavy runs)
pytest -m "not slow" -v

# Full regression test — requires ~60 s, runs on main/v2 in CI
pytest tests/test_example_regression.py -m slow -v
```

## Linting

All code must pass `ruff` before it is merged:

```bash
ruff check .
```

Auto-fixable issues (import ordering, trailing whitespace) can be resolved with:

```bash
ruff check . --fix
```

The CI workflow (`.github/workflows/test.yml`) runs lint and unit tests on
every push and pull request on Python 3.10 and 3.12.

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes and add or update tests as appropriate.
3. Verify that `ruff check .` and `pytest -m "not slow" -v` both pass locally.
4. Open a pull request against `main` with a clear description of what was
   changed and why.

## Reporting bugs

Please open an issue at <https://github.com/TUMFTM/flex-depot/issues> and
include:

- A minimal, self-contained example that reproduces the problem.
- The Python version, OS, and the output of `pip show flex-depot`.
- The full error traceback.

## Scope and design principles

FLEX-DEPOT is a research software accompanying a peer-reviewed publication. The
core formulation (MILP/LP model, MPC workflow, gate-closure logic) is
intentionally kept stable. Contributions that fix bugs, improve documentation,
add tests, or extend I/O are welcome. Proposed changes to the optimisation
formulation should be discussed in an issue first.

## License

By contributing you agree that your changes will be released under the
[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) that covers
this project.