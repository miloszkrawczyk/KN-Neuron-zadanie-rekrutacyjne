"""First look at the dataset before any signal processing: how many subjects
and samples there are, how long the recordings run in each group, and how
often the raw signal strays far from its usual level.

Writes the numbers to results/eda/structure.json and one plot of recording
lengths, so every figure later quoted in the README has a file behind it.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    CHANNELS,
    CLASS_LABELS,
    EDA_DIR,
    SAMPLING_RATE,
    SUBJECT_COLUMN,
    TARGET_COLUMN,
)
from src.data import load_eeg, subject_table

# Colours from the dataviz skill's validated palette, slots 1 and 2: told
# apart under colour blindness too, which a default red/green pair is not.
COLOURS = {"ADHD": "#2a78d6", "Control": "#eb6834"}

# Distances from a subject's own median, in the file's units. The typical
# spread is around 200, so these mark samples about 2.5x, 5x and 10x out.
OUTLIER_THRESHOLDS = (500, 1000, 2000)

df = load_eeg()
subjects = subject_table(df)
names = {number: name for name, number in CLASS_LABELS.items()}

summary = {
    "rows": len(df),
    "subjects": len(subjects),
    "subjects_per_class": {
        names[label]: int(count) for label, count in subjects[TARGET_COLUMN].value_counts().items()
    },
}

# *** Recording length ***
# Rows per subject divided by samples per second gives seconds.
seconds = df.groupby(SUBJECT_COLUMN, sort=False).size() / SAMPLING_RATE
subjects["seconds"] = subjects[SUBJECT_COLUMN].map(seconds)

summary["recording_seconds"] = {}
for label, group in subjects.groupby(TARGET_COLUMN):
    s = group["seconds"]
    summary["recording_seconds"][names[label]] = {
        "mean": round(s.mean(), 1),
        "median": round(s.median(), 1),
        "std": round(s.std(), 1),
        "min": round(s.min(), 1),
        "max": round(s.max(), 1),
        "total": round(s.sum(), 1),
    }

# Every window is cut from one subject's recording, so the share of samples
# is roughly the share of windows each class will contribute.
summary["adhd_share_of_samples"] = round((df[TARGET_COLUMN] == 1).mean(), 3)

# Kept on purpose: removing a row from a time series splices two moments
# that were never next to each other.
summary["duplicated_rows"] = int(df.duplicated().sum())

# *** Amplitude ***
signal = df[list(CHANNELS)]
medians = df.groupby(SUBJECT_COLUMN)[list(CHANNELS)].transform("median")
# The median rather than the mean, because the outliers this looks for
# would drag a mean towards themselves.
distance = (signal - medians).abs()

summary["amplitude"] = {
    "min": float(signal.min().min()),
    "max": float(signal.max().max()),
    "mean_subject_median": round(float(medians.mean().mean()), 1),
    "share_of_samples_beyond": {},
}
for label, name in names.items():
    in_class = distance[df[TARGET_COLUMN] == label]
    # (in_class > threshold) is a table of True/False; the mean of True/False
    # is the share of True, taken over rows and then over channels.
    summary["amplitude"]["share_of_samples_beyond"][name] = {
        str(threshold): round(float((in_class > threshold).mean().mean()), 5)
        for threshold in OUTLIER_THRESHOLDS
    }

EDA_DIR.mkdir(parents=True, exist_ok=True)
with open(EDA_DIR / "structure.json", "w") as file:
    json.dump(summary, file, indent=2)
print(json.dumps(summary, indent=2))

# *** Plot: recording length per group ***
# Two panels rather than two overlapping histograms, so neither group hides
# the other. Both axes are shared, so bar heights compare across panels.
bins = np.arange(60, subjects["seconds"].max() + 20, 20)
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=True)

for ax, name in zip(axes, ["ADHD", "Control"], strict=True):
    s = subjects.loc[subjects[TARGET_COLUMN] == CLASS_LABELS[name], "seconds"]
    ax.hist(s, bins=bins, color=COLOURS[name], edgecolor="white", linewidth=2)
    ax.axvline(s.median(), color="#0b0b0b", linewidth=1.5, linestyle="--")
    ax.set_title(f"{name}: {len(s)} subjects, median {s.median():.0f} s", loc="left", fontsize=11)
    ax.set_ylabel("subjects")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)

axes[-1].set_xlabel("recording length (seconds)")
fig.suptitle("Recordings of children with ADHD run longer and vary more", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(EDA_DIR / "recording_lengths.png", dpi=150)
print(f"saved plot and numbers to {EDA_DIR}")
