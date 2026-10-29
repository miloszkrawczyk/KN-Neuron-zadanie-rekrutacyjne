"""Tests for loading the CSV.

Most of them build a tiny CSV in a temporary folder, so they run without the
dataset and can feed load_eeg the broken files the real one never is. The
last one checks the real file against the numbers the dataset's authors
publish, and skips when the data has not been downloaded.
"""

import pandas as pd
import pytest

from src.config import (
    CHANNELS,
    LABEL_COLUMN,
    RAW_CSV,
    SUBJECT_COLUMN,
    TARGET_COLUMN,
)
from src.data import load_eeg, subject_table

needs_data = pytest.mark.skipif(
    not RAW_CSV.exists(),
    reason="needs the dataset, run: uv run python -m scripts.download_data",
)


def tiny_recording():
    """Two subjects, three samples each, in the same layout as the real file."""
    rows = []
    for subject, label in [("s1", "ADHD"), ("s2", "Control")]:
        for sample in range(3):
            row = {channel: float(sample) for channel in CHANNELS}
            row[LABEL_COLUMN] = label
            row[SUBJECT_COLUMN] = subject
            rows.append(row)
    return pd.DataFrame(rows)


def write_csv(tmp_path, df):
    path = tmp_path / "eeg.csv"
    df.to_csv(path, index=False)
    return path


def test_labels_become_adhd_one_control_zero(tmp_path):
    df = load_eeg(write_csv(tmp_path, tiny_recording()))

    assert list(df.loc[df[SUBJECT_COLUMN] == "s1", TARGET_COLUMN]) == [1, 1, 1]
    assert list(df.loc[df[SUBJECT_COLUMN] == "s2", TARGET_COLUMN]) == [0, 0, 0]


def test_signal_is_loaded_as_float32(tmp_path):
    df = load_eeg(write_csv(tmp_path, tiny_recording()))

    assert all(df[channel].dtype == "float32" for channel in CHANNELS)


def test_missing_channel_is_rejected(tmp_path):
    broken = tiny_recording().drop(columns="Cz")

    with pytest.raises(ValueError, match="columns"):
        load_eeg(write_csv(tmp_path, broken))


def test_missing_value_is_rejected(tmp_path):
    broken = tiny_recording()
    broken.loc[1, "Cz"] = None

    with pytest.raises(ValueError, match="missing values"):
        load_eeg(write_csv(tmp_path, broken))


def test_unknown_label_is_rejected(tmp_path):
    broken = tiny_recording()
    broken.loc[0, LABEL_COLUMN] = "adhd"

    with pytest.raises(ValueError, match="unknown labels"):
        load_eeg(write_csv(tmp_path, broken))


def test_subject_with_two_labels_is_rejected(tmp_path):
    broken = tiny_recording()
    broken.loc[0, LABEL_COLUMN] = "Control"

    with pytest.raises(ValueError, match="more than one label"):
        load_eeg(write_csv(tmp_path, broken))


def test_subject_split_across_the_file_is_rejected(tmp_path):
    # Rows in the order s1, s1, s1, s2, s2, s2 become s1, s2, s1, s2, s1, s2.
    broken = tiny_recording().iloc[[0, 3, 1, 4, 2, 5]]

    with pytest.raises(ValueError, match="split across"):
        load_eeg(write_csv(tmp_path, broken))


def test_subject_table_has_one_row_per_subject(tmp_path):
    subjects = subject_table(load_eeg(write_csv(tmp_path, tiny_recording())))

    assert list(subjects[SUBJECT_COLUMN]) == ["s1", "s2"]
    assert list(subjects[TARGET_COLUMN]) == [1, 0]


@needs_data
def test_real_data_has_121_subjects_61_adhd_60_control():
    subjects = subject_table(load_eeg())

    assert len(subjects) == 121
    assert (subjects[TARGET_COLUMN] == 1).sum() == 61
    assert (subjects[TARGET_COLUMN] == 0).sum() == 60
