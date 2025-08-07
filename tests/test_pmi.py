import os
import sys
import pytest

# Ensure pmi module is importable when tests are executed from any working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pmi import PMICalculator

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
