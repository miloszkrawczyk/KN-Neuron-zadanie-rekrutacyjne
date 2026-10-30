"""A small 1D CNN that reads raw EEG windows instead of band features.

The classical models were handed band power, which is what the EEG
literature says matters. This network gets the filtered signal itself and
has to find its own patterns along time. It is kept small on purpose: about
6000 training windows from 96 subjects is little for a network to learn from
without memorising them.
"""

import torch.nn as nn

from src.config import CHANNELS


def conv_block(in_channels, out_channels):
    """Conv1d -> ReLU -> MaxPool: find short shapes, keep the strongest, shorten 4x."""
    return nn.Sequential(
        # A 7-sample kernel spans about 55 ms at 128 Hz; padding=3 keeps the
        # length unchanged so only the pooling shortens the signal.
        nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
        nn.ReLU(),
        nn.MaxPool1d(4),
    )


class EEGCNN(nn.Module):
    """Small 1D CNN: three conv blocks, average over time, one logit out.

    Takes windows shaped (batch, channels, samples) and returns one number
    per window. It is a logit, not a probability: BCEWithLogitsLoss applies
    the sigmoid itself, which is more stable than doing it here.
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        # 512 samples shrink to 128, 32 and 8. After three blocks each point
        # sees about a second of signal, enough for a slow delta wave.
        self.blocks = nn.Sequential(
            conv_block(len(CHANNELS), 32),
            conv_block(32, 64),
            conv_block(64, 64),
        )
        # Averaging over whatever length is left makes the network work for
        # any window length, which the 2 s variant later relies on.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)  # (batch, 64, 1) -> (batch, 64)
        x = self.dropout(x)
        return self.head(x).squeeze(-1)  # (batch, 1) -> (batch,)
