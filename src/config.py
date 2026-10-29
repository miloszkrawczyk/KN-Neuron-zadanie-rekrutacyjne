"""Project-wide constants: where the data lives and which Kaggle dataset it
comes from. Facts about the recordings themselves get added here only after
they have been checked against the data.
"""

from pathlib import Path

# *** Paths ***
# Anchored to the project root so the scripts work no matter which
# directory I happen to run them from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

# The file exactly as Kaggle serves it, never edited. Anything derived from
# it goes elsewhere, so a fresh download always reproduces the starting point.
RAW_DIR: Path = DATA_DIR / "raw"
RAW_CSV: Path = RAW_DIR / "adhdata.csv"

# *** Kaggle ***
# A CSV conversion of the recordings published by Nasrabadi et al. on IEEE
# DataPort (doi:10.21227/rzfh-zn36).
KAGGLE_DATASET: str = "danizo/eeg-dataset-for-adhd"
