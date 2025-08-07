from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmi import PMICalculator


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
