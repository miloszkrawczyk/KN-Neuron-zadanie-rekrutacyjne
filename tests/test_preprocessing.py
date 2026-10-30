"""Tests for filtering, windowing and artifact rejection, on made-up signals
whose answer is known.

A pure sine wave at one frequency is the easiest thing to reason about: the
filter should pass it at full size inside the band and wipe it out outside.
Only the middle of each signal is checked, away from the edges, where any
filter has less context to work with. Constant signals do the same job for
windows: a window that is not all one value has picked up another subject.
"""

import numpy as np
import pandas as pd

from src.config import (
    CHANNELS,
    CLASS_LABELS,
    LABEL_COLUMN,
    SAMPLING_RATE,
    SUBJECT_COLUMN,
    TARGET_COLUMN,
    TRIM_SECONDS,
    WINDOW_SECONDS,
)
from src.preprocessing import bandpass, cut_windows, drop_artifact_windows, filter_recordings

SECONDS = 30
AMPLITUDE = 100.0
WINDOW = WINDOW_SECONDS * SAMPLING_RATE


def sine(frequency, seconds=SECONDS):
    time = np.arange(seconds * SAMPLING_RATE) / SAMPLING_RATE
    return AMPLITUDE * np.sin(2 * np.pi * frequency * time)


def middle(signal):
    """Drop the first and last 5 seconds."""
    edge = 5 * SAMPLING_RATE
    return signal[edge:-edge]


def recording(subject, label, values):
    """One subject in the same layout load_eeg returns, every channel = values."""
    df = pd.DataFrame({channel: values.astype("float32") for channel in CHANNELS})
    df[LABEL_COLUMN] = label
    df[SUBJECT_COLUMN] = subject
    df[TARGET_COLUMN] = CLASS_LABELS[label]
    return df


def test_alpha_wave_passes_almost_unchanged():
    filtered = bandpass(sine(10))

    peak = np.abs(middle(filtered)).max()

    assert 0.95 * AMPLITUDE < peak < 1.05 * AMPLITUDE


def test_mains_hum_is_removed():
    filtered = bandpass(sine(50))

    assert np.abs(middle(filtered)).max() < 0.01 * AMPLITUDE


def test_constant_offset_is_removed():
    filtered = bandpass(np.full(SECONDS * SAMPLING_RATE, 135.0))

    assert np.abs(middle(filtered)).max() < 1.0


def test_filter_does_not_shift_the_wave():
    wave = sine(10)

    filtered = bandpass(wave)

    # A one-pass filter would delay the wave, and the difference at the same
    # moment in time would reach the full amplitude.
    assert np.abs(middle(filtered) - middle(wave)).max() < 0.02 * AMPLITUDE


def test_output_is_float32():
    assert bandpass(sine(10)).dtype == "float32"


def test_subjects_are_filtered_separately():
    # A loud wave for the first subject, silence for the second. Filtered as
    # one long signal, the end of the wave would ring into the silence.
    loud = recording("s1", "ADHD", 1000 * sine(10) / AMPLITUDE)
    silent = recording("s2", "Control", np.zeros(SECONDS * SAMPLING_RATE))
    df = pd.concat([loud, silent], ignore_index=True)

    filtered = filter_recordings(df)

    second = filtered.loc[filtered[SUBJECT_COLUMN] == "s2", list(CHANNELS)]
    assert np.abs(second.to_numpy()).max() == 0


def test_filtering_keeps_rows_and_labels():
    df = pd.concat(
        [recording("s1", "ADHD", sine(10)), recording("s2", "Control", sine(20))],
        ignore_index=True,
    )

    before = df.copy()

    filtered = filter_recordings(df)

    assert filtered.shape == df.shape
    for column in (LABEL_COLUMN, SUBJECT_COLUMN, TARGET_COLUMN):
        assert filtered[column].equals(df[column])
    # The input is left untouched, so the raw signal stays available.
    assert df.equals(before)


def test_window_count_matches_trimmed_length():
    windows, _, _ = cut_windows(recording("s1", "ADHD", sine(10)))

    # 30 s minus 2 s at each end leaves 26 s, and a 4 s window every 2 s
    # fits (26 - 4) / 2 + 1 = 12 times.
    assert len(windows) == 12


def test_windows_have_channels_first_shape():
    windows, labels, subjects = cut_windows(recording("s1", "ADHD", sine(10)))

    assert windows.shape == (12, len(CHANNELS), WINDOW)
    assert windows.dtype == "float32"
    assert len(labels) == len(subjects) == 12


def test_window_never_mixes_two_subjects():
    samples = SECONDS * SAMPLING_RATE
    df = pd.concat(
        [
            recording("s1", "ADHD", np.ones(samples)),
            recording("s2", "Control", np.full(samples, 2.0)),
        ],
        ignore_index=True,
    )

    windows, labels, subjects = cut_windows(df)

    for window, label, subject in zip(windows, labels, subjects, strict=True):
        expected_value, expected_label = (1.0, 1) if subject == "s1" else (2.0, 0)
        assert (window == expected_value).all()
        assert label == expected_label


def test_edges_are_trimmed():
    values = np.zeros(SECONDS * SAMPLING_RATE)
    edge = TRIM_SECONDS * SAMPLING_RATE
    values[:edge] = 999
    values[-edge:] = 999

    windows, _, _ = cut_windows(recording("s1", "ADHD", values))

    assert windows.max() < 999


def test_windows_overlap_by_half():
    windows, _, _ = cut_windows(recording("s1", "ADHD", sine(10)))

    half = WINDOW // 2
    assert np.array_equal(windows[0][:, half:], windows[1][:, :half])


def test_spiky_window_is_dropped():
    windows = np.ones((10, len(CHANNELS), WINDOW), dtype="float32")
    windows[3, 0, 100] = 50  # one blink-sized spike in one window
    labels = np.zeros(10, dtype=int)
    subjects = np.array(["s1"] * 10)

    kept, kept_labels, kept_subjects = drop_artifact_windows(windows, labels, subjects)

    assert len(kept) == len(kept_labels) == len(kept_subjects) == 9
    assert kept.max() == 1


def test_loud_subject_is_not_dropped():
    # The second subject is ten times louder throughout, which is a property
    # of the recording, not an artifact. A fixed cut-off would drop it all.
    windows = np.ones((10, len(CHANNELS), WINDOW), dtype="float32")
    windows[5:] *= 10
    labels = np.array([1] * 5 + [0] * 5)
    subjects = np.array(["s1"] * 5 + ["s2"] * 5)

    kept, _, _ = drop_artifact_windows(windows, labels, subjects)

    assert len(kept) == 10
