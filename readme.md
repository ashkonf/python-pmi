# Pointwise Mutual Information Calculator

A minimal implementation of the pointwise mutual information (PMI) metric for
label–word pairs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Linting and Type Checking](#linting-and-type-checking)
- [License](#license)

## Installation
This project uses [uv](https://github.com/astral-sh/uv) for dependency management.
Install the development dependencies with:

```bash
uv sync --all-extras --dev
```

## Usage
The `PMICalculator` trains on labelled documents and can then compute PMI
scores, retrieve the training vocabulary for a label, and look up word counts.

```python
from pmi import PMICalculator

corpus = [("label", ["some", "words"])]
calculator = PMICalculator()
calculator.train(corpus)

print(calculator.pmi("label", "some"))
```

## Testing
Run the test suite with coverage:

```bash
uv run pytest --cov --cov-report=term-missing
```

## Linting and Type Checking
Format, lint, and type-check the code:

```bash
uv run ruff format
uv run ruff check
uv run pyright
```

All of these checks along with the tests can be run together via pre-commit:

```bash
uv run pre-commit run --all-files
```

## License

This project is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
