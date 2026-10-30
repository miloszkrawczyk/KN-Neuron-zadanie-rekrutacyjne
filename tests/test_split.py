"""Tests for the subject-level split.

The leak these guard against is invisible at runtime: nothing crashes, the
score just comes out too high. The first test is the most important one in
the project for that reason.
"""

import pandas as pd
import pytest

from src.config import RAW_CSV, SUBJECT_COLUMN, TARGET_COLUMN
from src.data import load_eeg, subject_table
from src.split import split_subjects

needs_data = pytest.mark.skipif(
    not RAW_CSV.exists(),
    reason="needs the dataset, run: uv run python -m scripts.download_data",
)


def twenty_subjects():
    """A subject table like subject_table() returns: 10 ADHD, then 10 control."""
    return pd.DataFrame(
        {
            SUBJECT_COLUMN: [f"s{number:02d}" for number in range(20)],
            TARGET_COLUMN: [1] * 10 + [0] * 10,
        }
    )


def labels_of(ids, subjects):
    return subjects.set_index(SUBJECT_COLUMN).loc[ids, TARGET_COLUMN]


def test_no_subject_is_in_both_sets():
    train, test = split_subjects(twenty_subjects())

    assert set(train).isdisjoint(test)


def test_every_subject_is_used_exactly_once():
    subjects = twenty_subjects()

    train, test = split_subjects(subjects)

    assert sorted(train + test) == sorted(subjects[SUBJECT_COLUMN])


def test_both_sets_have_both_classes():
    subjects = twenty_subjects()

    train, test = split_subjects(subjects)

    assert set(labels_of(train, subjects)) == {0, 1}
    assert set(labels_of(test, subjects)) == {0, 1}


def test_adhd_share_is_kept():
    subjects = twenty_subjects()

    train, test = split_subjects(subjects)

    # 20% of 20 subjects is 4: with stratify, exactly 2 of each class.
    assert labels_of(test, subjects).sum() == 2
    assert labels_of(train, subjects).sum() == 8


def test_split_is_repeatable():
    subjects = twenty_subjects()

    first, _ = split_subjects(subjects)
    second, _ = split_subjects(subjects)
    other, _ = split_subjects(subjects, seed=7)

    assert first == second
    assert first != other


@needs_data
def test_real_split_is_96_train_and_25_test():
    subjects = subject_table(load_eeg())

    train, test = split_subjects(subjects)

    assert len(train) == 96
    assert len(test) == 25
    assert labels_of(train, subjects).sum() == 48
    assert labels_of(test, subjects).sum() == 13
    assert set(train).isdisjoint(test)
