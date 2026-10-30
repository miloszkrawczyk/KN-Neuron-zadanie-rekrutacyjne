"""The one leak the CNN adds: normalisation statistics taken from the wrong set."""

import numpy as np

from src.train_cnn import normalise


def test_normalisation_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    train = rng.normal(100, 20, (50, 19, 64)).astype("float32")
    other = rng.normal(0, 1, (10, 19, 64)).astype("float32")

    scaled_train, _ = normalise(train, other)
    # Changing the other set must not change how the training set is scaled.
    scaled_again, _ = normalise(train, other * 1000)

    assert np.allclose(scaled_train.mean(axis=(0, 2)), 0, atol=1e-4)
    assert np.allclose(scaled_train.std(axis=(0, 2)), 1, atol=1e-4)
    assert np.array_equal(scaled_train, scaled_again)
