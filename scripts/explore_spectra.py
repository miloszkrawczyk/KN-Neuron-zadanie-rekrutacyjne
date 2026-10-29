"""Which frequencies differ between children with and without ADHD?

One power spectrum per subject and channel (Welch's method over the whole
recording), turned into power in the four classic bands, both absolute and
as a share of the total, plus the theta/beta ratio the ADHD literature
reports. Each of those features is scored by how well it alone separates the
two groups (AUC), one value per subject, since subjects are the real sample.

Writes results/eda/spectra.json and three plots.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

from src.config import (
    BANDS,
    CHANNELS,
    EDA_DIR,
    SAMPLING_RATE,
    SUBJECT_COLUMN,
    TARGET_COLUMN,
    TOTAL_POWER_RANGE,
    WELCH_SECONDS,
)
from src.data import load_eeg, subject_table

COLOURS = {"ADHD": "#2a78d6", "Control": "#eb6834"}

# Fronto-central sites, where the literature places the theta/beta effect.
FOCUS_CHANNELS = ("Fz", "Cz")

df = load_eeg()
subjects = subject_table(df)
labels = subjects[TARGET_COLUMN].to_numpy()

# *** One spectrum per subject ***
# welch() cuts the recording into 4 s pieces (half-overlapping by default),
# takes the spectrum of each and averages them. Its default detrend removes
# each piece's mean first, so the +135 offset in the raw signal does not
# show up as a huge spike at 0 Hz.
spectra = []
for _, recording in df.groupby(SUBJECT_COLUMN, sort=False):
    freqs, psd = welch(
        recording[list(CHANNELS)].to_numpy(),
        fs=SAMPLING_RATE,
        nperseg=WELCH_SECONDS * SAMPLING_RATE,
        axis=0,
    )
    spectra.append(psd)
spectra = np.stack(spectra)  # subjects x frequencies x channels
step = freqs[1] - freqs[0]


def band_power(low, high):
    """Area under each spectrum between low (included) and high (excluded)."""
    in_band = (freqs >= low) & (freqs < high)
    return spectra[:, in_band, :].sum(axis=1) * step  # subjects x channels


# *** Band features ***
total = band_power(*TOTAL_POWER_RANGE)
features = {}
for band, (low, high) in BANDS.items():
    power = band_power(low, high)
    # log10, because power differs between subjects by orders of magnitude
    # and a few loud recordings would otherwise dominate.
    features[f"{band} abs"] = np.log10(power)
    features[f"{band} rel"] = power / total
features["theta/beta"] = band_power(*BANDS["theta"]) / band_power(*BANDS["beta"])

# The ratio of the two relative powers is the same number, since both share
# the denominator. Checked rather than assumed.
same_ratio = np.allclose(features["theta/beta"], features["theta rel"] / features["beta rel"])
print(f"theta/beta identical for absolute and relative power: {same_ratio}")

# AUC: pick one child with ADHD and one without at random; how often is the
# feature higher for the child with ADHD? 0.5 is a coin toss, above 0.5 the
# feature runs higher in ADHD, below 0.5 lower.
auc = {
    name: [roc_auc_score(labels, values[:, c]) for c in range(len(CHANNELS))]
    for name, values in features.items()
}

summary = {
    "note": (
        "Descriptive only. With 61 and 60 subjects an uninformative feature still "
        "lands around 0.5 +/- 0.05, and among 171 features a few will stray "
        "further by chance."
    ),
    "theta_beta_same_for_absolute_and_relative": bool(same_ratio),
    "focus_channels": {},
}
for channel in FOCUS_CHANNELS:
    c = CHANNELS.index(channel)
    summary["focus_channels"][channel] = {
        name: {
            "auc": round(auc[name][c], 3),
            "median_adhd": round(float(np.median(values[labels == 1, c])), 3),
            "median_control": round(float(np.median(values[labels == 0, c])), 3),
        }
        for name, values in features.items()
    }

ranked = sorted(
    ((name, channel, auc[name][c]) for name in auc for c, channel in enumerate(CHANNELS)),
    key=lambda row: abs(row[2] - 0.5),
    reverse=True,
)
summary["furthest_from_chance"] = [
    {"feature": name, "channel": channel, "auc": round(value, 3)}
    for name, channel, value in ranked[:10]
]

EDA_DIR.mkdir(parents=True, exist_ok=True)
with open(EDA_DIR / "spectra.json", "w") as file:
    json.dump(summary, file, indent=2)
print(json.dumps(summary["focus_channels"]["Cz"], indent=2))
print(json.dumps(summary["furthest_from_chance"], indent=2))

# *** Plot 1: average spectrum at Cz ***
cz = CHANNELS.index("Cz")
shown = freqs > 0
fig, ax = plt.subplots(figsize=(8, 4.5))
# Alternating shades, since the bands touch and one shade would merge them.
for shade, (band, (low, high)) in zip(("#f0efec", "#e2e1dc") * 2, BANDS.items(), strict=True):
    ax.axvspan(low, high, color=shade, zorder=0)
    ax.text((low + high) / 2, 1.01, band, transform=ax.get_xaxis_transform(), ha="center")
for name, label in (("ADHD", 1), ("Control", 0)):
    # Averaging the logs gives the geometric mean, which a handful of very
    # loud subjects cannot pull up the way they would an ordinary mean.
    typical = 10 ** np.log10(spectra[labels == label][:, shown, cz]).mean(axis=0)
    ax.semilogy(freqs[shown], typical, color=COLOURS[name], linewidth=2, label=name)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("power (file units² / Hz, log scale)")
ax.set_xlim(0, SAMPLING_RATE / 2)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.suptitle(
    "Average spectrum at Cz: Control higher below 3 Hz, ADHD higher from alpha up",
    x=0.01,
    ha="left",
)
fig.tight_layout()
fig.savefig(EDA_DIR / "spectrum_cz.png", dpi=150)

# *** Plot 2: theta/beta ratio at Cz, one dot per subject ***
rng = np.random.default_rng(0)  # only spreads the dots sideways
fig, ax = plt.subplots(figsize=(5, 4.5))
for position, (name, label) in enumerate((("ADHD", 1), ("Control", 0))):
    values = features["theta/beta"][labels == label, cz]
    jitter = rng.uniform(-0.15, 0.15, len(values))
    ax.scatter(position + jitter, values, s=24, color=COLOURS[name], alpha=0.8)
    ax.hlines(np.median(values), position - 0.3, position + 0.3, color="#0b0b0b", linewidth=2)
ax.set_xticks([0, 1], ["ADHD", "Control"])
ax.set_ylabel("theta / beta power at Cz")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    f"Theta/beta at Cz is not higher in ADHD (AUC {auc['theta/beta'][cz]:.2f})",
    x=0.01,
    ha="left",
)
fig.tight_layout()
fig.savefig(EDA_DIR / "tbr_cz.png", dpi=150)

# *** Plot 3: AUC of every feature on every channel ***
# Diverging scale with a grey middle at 0.5. Blue where the feature runs
# higher in ADHD, red where it runs higher in Control, matching the group
# colours of the other plots.
grid = np.array([auc[name] for name in features]).T  # channels x features
reach = np.abs(grid - 0.5).max()
cmap = LinearSegmentedColormap.from_list("auc", ["#e34948", "#f0efec", "#2a78d6"])
fig, ax = plt.subplots(figsize=(8, 9))
image = ax.imshow(grid, cmap=cmap, vmin=0.5 - reach, vmax=0.5 + reach, aspect="auto")
for row in range(grid.shape[0]):
    for col in range(grid.shape[1]):
        ax.text(col, row, f"{grid[row, col]:.2f}", ha="center", va="center", fontsize=8)
ax.set_xticks(range(len(features)), list(features), rotation=45, ha="right")
ax.set_yticks(range(len(CHANNELS)), CHANNELS)
fig.colorbar(image, label="AUC (above 0.5: higher in ADHD)")
fig.suptitle("How well each feature alone separates the groups", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(EDA_DIR / "band_auc.png", dpi=150)
print(f"saved plots and numbers to {EDA_DIR}")
