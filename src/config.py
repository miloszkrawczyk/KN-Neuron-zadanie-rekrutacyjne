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


# *** Windows ***
# Each recording is cut into windows of this length. The same 4 s as the
# Welch pieces, so a window's spectrum has the same 0.25 Hz resolution as
# the EDA. Shorter windows would leave too few points across theta.
WINDOW_SECONDS: int = 4

# A new window every 2 s, so neighbours overlap by half. That doubles the
# number of windows, and an event cut in two at one boundary still lands
# whole in the window that straddles it.
WINDOW_STEP_SECONDS: int = 2

# Skipped at both ends of every recording. After filtering, the first and
# last 2 s spread 1.3x as wide as the middle for the median subject and 2.4
# to 3.8x for one subject in ten: part filter edge, part a child settling in.
TRIM_SECONDS: int = 2

# A window is dropped as an artifact when its peak is over this many times
# the median peak of the same subject's windows. Relative, because the
# recordings differ in overall loudness: a fixed cut-off of 1000 would drop
# about half of all windows. At 3x it drops 2.5% of ADHD windows and 2.3% of
# control windows, so it cleans without tilting the classes.
ARTIFACT_FACTOR: float = 3.0


# *** Features ***
# Welch piece length inside each 4 s window: three half-overlapping 2 s
# pieces, averaged. Measured on log theta at Cz, that cuts the spread between
# windows of the same subject from 0.207 to 0.180 while the spread between
# subjects stays at 0.31. The cost is 0.5 Hz resolution, still 8 points
# across theta.
FEATURE_WELCH_SECONDS: int = 2


# *** Splitting ***
# Share of subjects held out for the final test: 25 of 121 (13 ADHD, 12
# control), leaving 96 to train and cross-validate on. One subject in a test
# set this small moves accuracy by 4 percentage points, so every choice is
# made on cross-validation and the test set is scored once, at the very end.
TEST_SIZE: float = 0.2


# *** Cross-validation ***
# Training subjects are dealt into this many groups; each takes one turn as
# the validation set. Five leaves about 19 subjects per validation group; ten
# would leave about 10, where one subject moves the score by 10 points.
CV_FOLDS: int = 5

# The whole cross-validation is repeated with a different shuffle each time
# (seeds SEED, SEED + 1, ...). Scores varied by +/- 0.10 between folds, so
# one deal of the groups is too much luck to compare settings on.
CV_REPEATS: int = 3


# *** Hyperparameters ***
# A deliberately small grid: with scores spread +/- 0.10 between folds, the
# best of many settings would partly be the luckiest one.
#
# Logistic regression: C is the inverse strength of the penalty on large
# weights. 1.0 is scikit-learn's default; smaller values make the model
# trust no single feature much, which tends to help with 95 features and
# only 96 subjects.
LOGISTIC_C_VALUES: tuple = (0.01, 0.1, 1.0)

# Random forest: the fewest windows a leaf may hold. 1 is the default and
# lets a tree memorise single windows; 20 is about a third of a typical
# subject's windows, 100 about one and a half subjects. Larger leaves also
# give less extreme probabilities, which may ease the forest's lean towards
# ADHD seen in cross-validation.
FOREST_MIN_SAMPLES_LEAF: tuple = (1, 20, 100)

# Picked from results/classic_settings.json by a rule set before looking:
# the best mean subject-level balanced accuracy, unless a more cautious
# setting trails it by less than one standard error (about 0.03). C=0.01 was
# best outright (0.718); leaf=100 trailed leaf=1 by 0.007 (0.733 vs 0.740).
CHOSEN_LOGISTIC_C: float = 0.01
CHOSEN_FOREST_LEAF: int = 100


# *** CNN training ***
# An epoch takes about 0.2 s on the Mac's graphics chip, and the weights kept
# are those of the epoch with the lowest validation loss, not the last one.
# Going past the best epoch costs only time.
CNN_EPOCHS: int = 30

# Windows per weight update. Small enough to fit easily in memory, large
# enough that one odd window cannot drag an update off course.
CNN_BATCH_SIZE: int = 64

# Adam's own default. The network starts from random weights, so the first
# steps have a long way to travel.
CNN_LEARNING_RATE: float = 1e-3

# Share of the training folds' subjects set aside, again by subject, to pick
# the best epoch. The fold being scored is never used for that.
CNN_VALIDATION_SHARE: float = 0.2


# *** Evaluation ***
# Probability from which a window, or a subject, counts as ADHD. Left at the
# middle: the models are trained with class weights, and here a missed ADHD
# and a false alarm cost about the same, unlike a missed melanoma. Tuning it
# on 96 subjects would mostly fit the threshold to chance.
DECISION_THRESHOLD: float = 0.5


# *** Reproducibility ***
# One seed for numpy, scikit-learn and PyTorch. Without it every run would
# draw a different split and different starting weights, and two results
# could not be told apart from noise.
SEED: int = 42
