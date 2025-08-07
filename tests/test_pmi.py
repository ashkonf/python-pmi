import math
import sys
from pathlib import Path

# Ensure the project root is on the Python path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from pmi import PMICalculator


def build_calculator():
    calc = PMICalculator()
    corpus = [
        ("label1", ["word1", "word2"]),
        ("label2", ["word1"]),
    ]
    calc.train(corpus)
    return calc


def test_count_and_key_set():
    calc = build_calculator()
    assert calc.count("word1") == 2
    assert calc.count("word2") == 1
    assert set(calc.key_set("label1")) == {"word1", "word2"}
    assert set(calc.key_set("label2")) == {"word1"}


def test_pmi_values():
    calc = build_calculator()
    assert math.isclose(calc.pmi("label1", "word1"), 0.75)
    assert math.isclose(calc.pmi("label2", "word1"), 1.5)
    assert math.isclose(calc.pmi("label1", "word2"), 1.5)
