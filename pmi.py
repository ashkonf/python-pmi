from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, KeysView


class PMICalculator:
    """Compute pointwise mutual information for label-word pairs."""

    def __init__(self) -> None:
        """Initialize the PMI calculator."""
        self.label_counts: defaultdict[str, float] = defaultdict(float)
        self.word_counts: defaultdict[str, float] = defaultdict(float)
        self.joint_counts: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.num_pairs: int = 0

    def train(
        self, corpus: Iterable[tuple[str, Iterable[str]]], smoothing_factor: float = 0.0
    ) -> None:
        """Populate counts from the corpus with optional smoothing."""
        self.label_counts = defaultdict(lambda: smoothing_factor)
        self.word_counts = defaultdict(lambda: smoothing_factor)
        self.joint_counts = defaultdict(lambda: defaultdict(lambda: smoothing_factor))
        self.num_pairs = 0

        for label, document in corpus:
            for word in document:
                weight: float = 1.0
                self.label_counts[label] += weight
                self.word_counts[word] += weight
                self.joint_counts[label][word] += weight
                self.num_pairs += 1

    def key_set(self, label: str) -> KeysView[str]:
        """Return the set of words associated with a label."""
        return self.joint_counts[label].keys()

    def pmi(self, label: str, word: str) -> float:
        """Calculate the PMI for a label and word."""
        joint_prob: float = float(self.joint_counts[label][word]) / float(
            self.num_pairs
        )
        label_prob: float = float(self.label_counts[label]) / float(self.num_pairs)
        word_prob: float = float(self.word_counts[word]) / float(self.num_pairs)
        return joint_prob / (label_prob * word_prob)

    def count(self, word: str) -> float:
        """Retrieve the observed count for a word."""
        return self.word_counts[word]
