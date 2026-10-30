"""Where the network's maths gets done."""

import torch


def pick_device():
    """Use the Mac's graphics chip if it is there, then an NVIDIA card, then the processor."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
