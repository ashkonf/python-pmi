# Pointwise Mutual Information Calculator

The Pointwise Mutual Information (PMI) Calculator repository provides a Python implementation of the PMI measure, which is used to compute the similarity of words and their associated categories according to the following equation:

![PMI Equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/ff54cfce726857db855d4dd0a9dee2c6a5e7be99)

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

The three `print()` statements output the following:

1. The words associated with "ENTERTAINMENT".
2. The PMI (9.866219341079493) for the label "CRIME" and the word "killed".
3. The word count for "kids".

Remember that you must call the `train()` method with a suitable dataset before using the other `PMICalculator` methods.

And that's it! You can now integrate the PMI Calculator repo with your application of choice.

## Testing

Unit tests are provided in the `tests` directory and can be run with
[`pytest`](https://docs.pytest.org). From the project root, execute:

```
pytest -q
```

The tests verify word counts, key sets, and PMI calculations using a
sample corpus.

## License

This project is licensed under the

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
