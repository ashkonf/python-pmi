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
