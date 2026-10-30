"""Cleaning the raw signal before anything is measured from it.

The filter runs over each subject's recording separately and as a whole,
before it is cut into windows. Filtering across two subjects would let the
end of one recording bleed into the start of the next.
"""

from scipy.signal import butter, sosfiltfilt

from src.config import CHANNELS, FILTER_BAND, FILTER_ORDER, SAMPLING_RATE, SUBJECT_COLUMN


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
