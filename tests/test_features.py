"""Tests for the band power features, on a sine wave whose band is known."""

import numpy as np

from src.config import BANDS, CHANNELS, SAMPLING_RATE, WINDOW_SECONDS
from src.features import band_features


def alpha_windows(count=3):
    """Windows holding a 10 Hz sine on every channel, plus a little noise."""
    time = np.arange(WINDOW_SECONDS * SAMPLING_RATE) / SAMPLING_RATE
    wave = 100 * np.sin(2 * np.pi * 10 * time)
    noise = np.random.default_rng(0).normal(0, 1, (count, len(CHANNELS), len(time)))
    return (wave + noise).astype("float32")


def test_alpha_wave_lands_in_alpha():
    features = band_features(alpha_windows())

    for channel in CHANNELS:
        powers = {band: features[f"{channel} {band}"] for band in BANDS}
        assert (powers["alpha"] > powers["delta"]).all()
        assert (powers["alpha"] > powers["theta"]).all()
        assert (powers["alpha"] > powers["beta"]).all()


def test_one_row_per_window_and_95_columns():
    features = band_features(alpha_windows(count=5))

    assert features.shape == (5, len(CHANNELS) * (len(BANDS) + 1))
