"""Tests for the band-pass filter, on made-up signals whose answer is known.

A pure sine wave at one frequency is the easiest thing to reason about: the
filter should pass it at full size inside the band and wipe it out outside.
Only the middle of each signal is checked, away from the edges, where any
filter has less context to work with.
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
)
from src.preprocessing import bandpass, filter_recordings

SECONDS = 30
AMPLITUDE = 100.0


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
