import math
import sys
from pathlib import Path
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    
@pytest.fixture
def trained_calc():
    corpus = [('A', ['x', 'y']), ('B', ['x'])]
    calc = PMICalculator()
    calc.train(corpus)
    return calc

  
def test_pmi_values(trained_calc):
    assert trained_calc.pmi('A', 'x') == pytest.approx(0.75)
    assert trained_calc.pmi('B', 'x') == pytest.approx(1.5)

    
def test_key_set(trained_calc):
    assert set(trained_calc.key_set('A')) == {'x', 'y'}

    
def test_count(trained_calc):
    assert trained_calc.count('x') == 2

    
def test_pmi_counts_and_keyset() -> None:
    """Test counting words and retrieving key sets."""
    corpus = [("label1", ["a", "b"]), ("label2", ["a"])]
    calc = PMICalculator()
    calc.train(corpus)

    assert calc.count("a") == 2
    assert "b" in calc.key_set("label1")
    assert calc.num_pairs == 3
    assert calc.pmi("label1", "a") > 0.0


def test_smoothing_factor() -> None:
    """Test applying a smoothing factor to unseen words."""
    corpus = [("x", ["y"])]
    calc = PMICalculator()
    calc.train(corpus, smoothing_factor=1.0)

    assert calc.count("missing") == 1.0
    assert "y" in calc.key_set("x")


def test_expected_pmi_value() -> None:
    """Test PMI equals the joint to marginal probability ratio."""
    corpus = [("label1", ["a"]), ("label2", ["b"])]
    calc = PMICalculator()
    calc.train(corpus)

    assert calc.pmi("label1", "a") == 2.0


def test_unseen_pair_pmi_is_zero() -> None:
    """Test PMI is zero for label-word pairs not in the corpus."""
    corpus = [("label1", ["a"]), ("label2", ["b"])]
    calc = PMICalculator()
    calc.train(corpus)

    assert calc.pmi("label1", "b") == 0.0
