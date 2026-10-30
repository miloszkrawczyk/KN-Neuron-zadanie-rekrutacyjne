"""Project-wide constants: where the data lives, what the recordings look
like, and the seed that keeps runs comparable.

Every fact about the recordings below was checked against the CSV itself,
not copied from the dataset's description. Settings for later stages (filter,
windows, bands, split) are added in the step that first uses them.
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

# Numbers and plots that go into the repository, unlike data/. Every figure
# quoted in the README has to be traceable to a file in here.
RESULTS_DIR: Path = PROJECT_ROOT / "results"
EDA_DIR: Path = RESULTS_DIR / "eda"

# *** Kaggle ***
# A CSV conversion of the recordings published by Nasrabadi et al. on IEEE
# DataPort (doi:10.21227/rzfh-zn36).
KAGGLE_DATASET: str = "danizo/eeg-dataset-for-adhd"

# *** The recordings ***
# Samples per second: 128 rows of the CSV make one second of recording. The
# file has no time column, so this comes from the dataset's description. It
# checks out on the data: assuming 128, mains interference lands exactly on
# 50 Hz, the frequency of the power grid where the recordings were made.
SAMPLING_RATE: int = 128

# The 19 electrodes of the 10-20 system, in the order the CSV stores them.
# The file uses the newer names T7, T8, P7, P8, which older papers call T3,
# T4, T5, T6. The order is fixed because a channel's position becomes its row
# in every array built from the data, so reordering after training would
# silently feed the model the wrong electrodes.
CHANNELS: tuple = (
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
)

# The two non-signal columns. One recording per subject, and every row of a
# subject carries the same label.
SUBJECT_COLUMN: str = "ID"
LABEL_COLUMN: str = "Class"

# ADHD is the positive class, the one the model is asked to find. With ADHD
# as 1, recall means the share of children with ADHD that were caught, and
# PyTorch's pos_weight applies to them.
CLASS_LABELS: dict = {"Control": 0, "ADHD": 1}

# The 0/1 version of LABEL_COLUMN, added on loading. A separate column, so
# the text from the file stays as it was and the two can be compared.
TARGET_COLUMN: str = "label"


# *** Frequency bands ***
# The four classic EEG bands in Hz, as used across the ADHD literature. Each
# band includes its lower edge and excludes its upper one, so 8 Hz counts as
# alpha only and no frequency is counted twice.
BANDS: dict = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

# Length of the pieces Welch's method averages over. 4 s is 512 samples,
# which spaces the spectrum 0.25 Hz apart: 16 points across theta, the
# narrowest band. Longer pieces would sharpen that but leave fewer to average,
# and the spectrum would get noisier.
WELCH_SECONDS: int = 4

# Relative power is a band's share of the power in this range. The upper edge
# stops below the 50 Hz mains line, which is stronger in one group.
TOTAL_POWER_RANGE: tuple = (0.5, 40)


# *** Filtering ***
# Passes 0.5 to 40 Hz, which keeps all four bands above. The lower edge
# removes the constant offset of about +135 and slow electrode drift; the
# upper edge removes the 50 Hz mains line, stronger in the control group, and
# a smaller line at 56 Hz present in both.
FILTER_BAND: tuple = (0.5, 40)

# A 4th-order Butterworth: flat inside the band, so the waves it keeps come
# out at their original size. Run forwards and backwards by sosfiltfilt, it
# cuts 50 Hz to 0.6% of its amplitude (-44 dB).
FILTER_ORDER: int = 4


# *** Reproducibility ***
# One seed for numpy, scikit-learn and PyTorch. Without it every run would
# draw a different split and different starting weights, and two results
# could not be told apart from noise.
SEED: int = 42
