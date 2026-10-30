"""Did the group differences from the EDA survive cleaning?

The strongest difference in the EDA was more delta in the control group at
the back of the head. Delta is also where blinks and movement land, so this
runs the same band features on two versions of the windows: raw, and after
the filter and artifact rejection. A difference that shrinks towards
AUC 0.5 once the signal is clean was probably not coming from the brain.

Only training subjects are used, so the test set stays unseen until the
final evaluation. Writes results/eda/clean_vs_raw_auc.json.
"""

import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import EDA_DIR, SUBJECT_COLUMN, TARGET_COLUMN
from src.data import load_eeg, subject_table
from src.features import band_features
from src.preprocessing import cut_windows, drop_artifact_windows, filter_recordings
from src.split import split_subjects

# Back of the head, where the EDA found the strongest delta difference.
POSTERIOR_CHANNELS = ("Pz", "P3", "O1", "P7")

df = load_eeg()
train_ids, _ = split_subjects(subject_table(df))
train = df[df[SUBJECT_COLUMN].isin(train_ids)]

versions = {
    "raw": cut_windows(train),
    "clean": drop_artifact_windows(*cut_windows(filter_recordings(train))),
}


def subject_auc(windows, labels, subjects):
    """AUC of every feature, with one value per subject: the mean over its windows."""
    features = band_features(windows)
    features[SUBJECT_COLUMN] = subjects
    features[TARGET_COLUMN] = labels
    per_subject = features.groupby(SUBJECT_COLUMN).mean()
    target = per_subject.pop(TARGET_COLUMN)
    return {name: roc_auc_score(target, per_subject[name]) for name in per_subject.columns}


auc = {name: subject_auc(*windows) for name, windows in versions.items()}
table = pd.DataFrame(auc).round(3)  # one row per feature, columns raw and clean
table["change"] = (table["clean"] - table["raw"]).round(3)

summary = {
    "subjects": len(train_ids),
    "note": (
        "Training subjects only. Subject-level AUC of each feature, the mean "
        "over a subject's windows. With 48 and 48 subjects an uninformative "
        "feature lands around 0.5 +/- 0.06."
    ),
    "posterior_delta": {
        channel: table.loc[f"{channel} delta", ["raw", "clean"]].to_dict()
        for channel in POSTERIOR_CHANNELS
    },
    "features_beyond_0.15_from_chance": {
        version: int(((table[version] - 0.5).abs() > 0.15).sum()) for version in ("raw", "clean")
    },
    "biggest_changes": (
        table.reindex(table["change"].abs().sort_values(ascending=False).index)
        .head(10)
        .reset_index(names="feature")
        .to_dict("records")
    ),
    "auc": table.to_dict("index"),
}

EDA_DIR.mkdir(parents=True, exist_ok=True)
with open(EDA_DIR / "clean_vs_raw_auc.json", "w") as file:
    json.dump(summary, file, indent=2)

print(f"windows: raw {len(versions['raw'][0])}, clean {len(versions['clean'][0])}")
print("posterior delta, raw -> clean:")
for channel, values in summary["posterior_delta"].items():
    print(f"  {channel}: {values['raw']:.3f} -> {values['clean']:.3f}")
print("features more than 0.15 from chance:", summary["features_beyond_0.15_from_chance"])
print("biggest changes:")
print(pd.DataFrame(summary["biggest_changes"]).to_string(index=False))
print(f"saved to {EDA_DIR / 'clean_vs_raw_auc.json'}")
