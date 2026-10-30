"""The one leak check cross-validation needs: no subject on both sides."""

import numpy as np

from src.config import CV_FOLDS, CV_REPEATS
from src.train_classic import cv_splits


def test_cv_never_puts_a_subject_on_both_sides():
    # 20 subjects, 5 windows each, half of them ADHD.
    subjects = np.repeat([f"s{number:02d}" for number in range(20)], 5)
    labels = np.repeat([1] * 10 + [0] * 10, 5)

    splits = cv_splits(labels, subjects)

    assert len(splits) == CV_FOLDS * CV_REPEATS
    for train, validation in splits:
        assert set(subjects[train]).isdisjoint(subjects[validation])
