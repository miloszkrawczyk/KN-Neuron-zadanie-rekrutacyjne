"""From the raw signal to clean, fixed-length windows: filter, cut, drop.

The filter runs over each subject's recording separately and as a whole,
before it is cut into windows. Filtering across two subjects would let the
end of one recording bleed into the start of the next.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.config import (
    ARTIFACT_FACTOR,
    CHANNELS,
    FILTER_BAND,
    FILTER_ORDER,
    SAMPLING_RATE,
    SUBJECT_COLUMN,
    TARGET_COLUMN,
    TRIM_SECONDS,
    WINDOW_SECONDS,
    WINDOW_STEP_SECONDS,
)


def bandpass(signal, fs=SAMPLING_RATE, band=FILTER_BAND, order=FILTER_ORDER):
    """Filter a (samples x channels) array along time, without shifting it."""
    # "sos" stores the filter as a chain of small second-order steps. The
    # alternative, one long fraction, loses precision with a cut-off as low
    # as 0.5 Hz and can turn unstable.
    sos = butter(order, band, btype="band", fs=fs, output="sos")
    # Running the filter forwards and then backwards cancels the delay a
    # single pass would add, so peaks stay exactly where they were.
    filtered = sosfiltfilt(sos, signal, axis=0)
    # scipy works in float64; back to float32 to keep the memory halved.
    return filtered.astype("float32")


def filter_recordings(df):
    """Band-pass each subject's recording on its own and return a filtered copy.

    Row order and every non-signal column are left as they were.
    """
    filtered = df.copy()
    channels = list(CHANNELS)
    for rows in df.groupby(SUBJECT_COLUMN, sort=False).groups.values():
        filtered.loc[rows, channels] = bandpass(df.loc[rows, channels].to_numpy())
    return filtered


def cut_windows(df):
    """Cut each subject's recording into overlapping windows.

    Returns three arrays of the same length: the windows, shaped
    (windows, channels, samples), the 0/1 label of each window, and the ID of
    the subject it came from, which the split and subject-level scores need.
    """
    size = WINDOW_SECONDS * SAMPLING_RATE
    step = WINDOW_STEP_SECONDS * SAMPLING_RATE
    trim = TRIM_SECONDS * SAMPLING_RATE

    windows, labels, subjects = [], [], []
    # One subject at a time, so no window can straddle two recordings.
    for subject, recording in df.groupby(SUBJECT_COLUMN, sort=False):
        signal = recording[list(CHANNELS)].to_numpy()
        signal = signal[trim : len(signal) - trim]
        # The last start that still leaves a full window; a shorter leftover
        # at the end is dropped rather than padded.
        starts = range(0, len(signal) - size + 1, step)
        # Transposed to channels x samples, the layout PyTorch's Conv1d takes.
        windows += [signal[start : start + size].T for start in starts]
        labels += [recording[TARGET_COLUMN].iloc[0]] * len(starts)
        subjects += [subject] * len(starts)

    return np.stack(windows), np.array(labels), np.array(subjects)


def drop_artifact_windows(windows, labels, subjects, factor=ARTIFACT_FACTOR):
    """Drop windows whose peak is over factor x the median peak of that subject's windows.

    Compared within each subject, so a recording that is loud all the way
    through keeps its windows and only the sudden spikes go. It never looks
    at labels or at other subjects, so nothing leaks across a later split.
    """
    peaks = np.abs(windows).max(axis=(1, 2))
    keep = np.ones(len(windows), dtype=bool)
    for subject in np.unique(subjects):
        mine = subjects == subject
        # The median, because the spikes being hunted would pull a mean up
        # towards themselves.
        keep[mine] = peaks[mine] <= factor * np.median(peaks[mine])
    return windows[keep], labels[keep], subjects[keep]
