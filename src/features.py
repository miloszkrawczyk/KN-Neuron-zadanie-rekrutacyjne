"""Turning windows into the table of numbers the classical models train on.

Each window becomes 95 numbers: log10 power in the four bands on each of the
19 channels, plus the theta/beta ratio on each channel. Relative power is
left out on purpose: a model working on logs can recover it from differences
between columns, and fewer columns are safer with only 96 training subjects.
"""

import numpy as np
import pandas as pd
from scipy.signal import welch

from src.config import BANDS, CHANNELS, FEATURE_WELCH_SECONDS, SAMPLING_RATE


def band_features(windows, fs=SAMPLING_RATE):
    """One row per window: log10 power in each band on each channel, plus theta/beta.

    Takes windows shaped (windows, channels, samples) and returns a DataFrame
    with columns named like "Cz theta" and "Cz theta/beta".
    """
    # All windows at once, along the time axis: windows x channels x frequencies.
    freqs, psd = welch(windows, fs=fs, nperseg=FEATURE_WELCH_SECONDS * fs, axis=-1)
    step = freqs[1] - freqs[0]

    # Area under the spectrum in each band, the same rule as in the EDA:
    # lower edge in, upper edge out.
    power = {
        band: psd[..., (freqs >= low) & (freqs < high)].sum(axis=-1) * step
        for band, (low, high) in BANDS.items()
    }

    columns = {}
    for c, channel in enumerate(CHANNELS):
        for band in BANDS:
            # log10, because power spans orders of magnitude between subjects
            # and a linear model would otherwise be steered by the loudest.
            columns[f"{channel} {band}"] = np.log10(power[band][:, c])
        columns[f"{channel} theta/beta"] = power["theta"][:, c] / power["beta"][:, c]
    return pd.DataFrame(columns)
