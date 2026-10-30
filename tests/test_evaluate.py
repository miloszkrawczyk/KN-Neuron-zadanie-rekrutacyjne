"""Tests for the metrics, on examples small enough to work out by hand."""

import numpy as np

from src.evaluate import score, subject_scores


def test_subject_score_is_mean_of_its_windows():
    labels = np.array([1, 1, 1, 0, 0])
    probabilities = np.array([0.9, 0.6, 0.3, 0.2, 0.4])
    subjects = np.array(["a", "a", "a", "b", "b"])

    subject_labels, subject_probabilities = subject_scores(labels, probabilities, subjects)

    assert list(subject_labels) == [1, 0]
    assert np.allclose(subject_probabilities, [0.6, 0.3])


def test_known_confusion_matrix():
    # 4 ADHD, of which 3 found; 4 controls, of which 1 wrongly called ADHD.
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.1])

    result = score(labels, probabilities)

    assert result["confusion_matrix"] == [[3, 1], [1, 3]]
    assert result["sensitivity"] == 0.75
    assert result["specificity"] == 0.75
    assert result["balanced_accuracy"] == 0.75
