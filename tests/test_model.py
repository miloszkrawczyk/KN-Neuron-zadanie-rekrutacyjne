"""Shape checks only. The weights are random until training runs, so there is
nothing about the answers worth asserting yet.
"""

import torch

from src.config import CHANNELS, SAMPLING_RATE
from src.models.cnn import EEGCNN


def test_one_logit_per_window():
    windows = torch.rand(8, len(CHANNELS), 4 * SAMPLING_RATE)

    output = EEGCNN()(windows)

    assert output.shape == (8,)


def test_shorter_windows_also_work():
    windows = torch.rand(8, len(CHANNELS), 2 * SAMPLING_RATE)

    output = EEGCNN()(windows)

    assert output.shape == (8,)
