# Pointwise Mutual Information Calculator

## Contents

- [Overview](#Overview)
- [Dependencies](#Dependencies)
- [Usage](#Usage)
- [Training](#Training)
- [Example](#Example)
- [Testing](#Testing)
- [License](#License)
- [Links](#Links)

## Overview

The Pointwise Mutual Information (PMI) Calculator repository provides a Python implementation of the PMI measure, which is used to compute the similarity of words and their associated categories according to the following equation:

![PMI Equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/ff54cfce726857db855d4dd0a9dee2c6a5e7be99)

That is, the PMI is the conditional probability of mutual information, where _x_ and _y_ are independent, discrete random variables. Please see the PMI [Wikipedia article](https://en.wikipedia.org/wiki/Pointwise_mutual_information) page.

## Dependencies

None.

## Usage

The `pmi` module exports the `PMICalculator` class.

```
pmi_calculator = PMICalculator()
```

Once initialized, a `PMICalculator` object may be used to call the `train`, `pmi`, `key_set`, and `count` instance methods:

```
pmi_calculator.train(corpus, smoothing_factor)
pmi_calculator.pmi(label, word)
pmi_calculator.key_set(label)
print(pmi_calculator.count(word))
```

These four methods provide the primary functionality of the PMI Calculator repo.

### train()

The `train()` method takes two arguments:

| Name | Type | Description | Optional? | Default<br/>Value |
| ---- | ---- | ----------- | --------- | ----------------- |
| `corpus` | `[(str, [str])]` | The data used to train the PMI model; that is, a list of tuples. For each tuple, `(label, document)`, `label` is a string and `document` is broken into a list of strings. See the [Training](#training) section for more information. | False |  |
| `smoothing_factor` | `float` | The [additive smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) factor to be applied to the PMI model. | True | `0.0` |

The `train()` method returns nothing. You must call this method to ensure the `PMICalculator` object is trained before calling any of the other `PMICalculator` instance methods.

### pmi()

The `pmi()` method takes two arguments:

| Name | Type | Description | Optional? | Default<br/>Value |
| ---- | ---- | ----------- | --------- | ----------------- |
| `label` | `str` | The label that the PMI model will compare. | False |  |
| `word` | `str` | The word that the PMI model will compare. | False |  |

The `pmi()` method returns the calculated PMI measure for a given label and word. For a more in-depth explanation of the measure, see the [Wikipedia article](https://en.wikipedia.org/wiki/Pointwise_mutual_information) on PMI.

### key_set()

The `key_set()` method takes a single argument:

| Name | Type | Description | Optional? | Default<br/>Value |
| ---- | ---- | ----------- | --------- | ----------------- |
| `label` | `str` | The label for which  associated words will be returned. | False |  |

The `key_set()` method returns a view object that displays all of the words associated with a given label as dictionary keys.

### count()

The `count()` method takes a single argument:

| Name | Type | Description | Optional? | Default<br/>Value |
| ---- | ---- | ----------- | --------- | ----------------- |
| `word` | `str` | The word for which the instance count will be retrieved. | False | |

The `count()` method returns a float denoting the number of times a given word appears in the training-data corpus.

## Training

The general format for PMI training data is a list of tuples. For each tuple, `(label, document)`, `label` is the category for the sample text, and `document` is the sample text itself; `label` should be a string, while `document` should be a list of strings (the strings being the words of the text).

The code below shows how to format data ([sample_data.csv](https://github.com/DocketScience/PythonPMI/blob/master/sample_data.csv) in this case) for the PMI model. This example uses the [pandas](https://pandas.pydata.org) library, which is optional.

```
import pandas as pd

training_data = pd.read_csv("sample_data.csv")
training_data["document"] = training_data["document"].astype(str)

for i, row in training_data.iterrows():
    training_data.at[i, "document"] = row["document"].split()

training_data_pairs = training_data.values.tolist()
```

The `training_data_pairs` list is then be passed to the PMI model as the training-data corpus, as shown in the example that follows.

## Example

Below is an implementation of the PMI module using [sample_data.csv](https://github.com/DocketScience/PythonPMI/blob/master/sample_data.csv) from the [Training](#training) section. Each tuple in `training_data_pairs` contains movie genres as labels and the associated movie descriptions as documents. The `PMICalculator` is first initialized and then trained.

```
from pmi import PMICalculator

# PMI Calculations
pmi_calculator = PMICalculator()
pmi_calculator.train(training_data_pairs)

print("Keyset of Label ENTERTAINMENT: ", pmi_calculator.key_set("ENTERTAINMENT"))
print("PMI for label CRIME and word Killed: ", pmi_calculator.pmi("CRIME", "killed"))
print("Word Count for word kids: ", pmi_calculator.count("kids"))
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

Python SCOTUS Dataset is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).


## Links

- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [pandas](https://pandas.pydata.org)
- [Wikpedia: additive smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)
- [Wikipedia: PMI](https://en.wikipedia.org/wiki/Pointwise_mutual_information)
