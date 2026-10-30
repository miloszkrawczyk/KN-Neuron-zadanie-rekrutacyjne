"""Which electrodes and bands do the classical models lean on?

Both chosen models are trained once on every window of the 96 training
subjects, then read out: the logistic regression's weights, which carry a
direction, and the random forest's importances, which say how much each
feature helped the trees split the data. The test subjects stay unseen.

Writes results/feature_importance.json and results/feature_importance.png.
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CHOSEN_FOREST_LEAF, CHOSEN_LOGISTIC_C, RESULTS_DIR
from src.train_classic import logistic_regression, random_forest, training_data

TOP = 15
COLOURS = {"ADHD": "#2a78d6", "Control": "#eb6834"}

features, labels, _ = training_data()

regression = logistic_regression(CHOSEN_LOGISTIC_C).fit(features, labels)
forest = random_forest(CHOSEN_FOREST_LEAF).fit(features, labels)

# The features were standardised inside the pipeline, so the weights are on
# one scale and can be compared. Positive pushes the prediction towards
# ADHD. Neighbouring channels carry much the same signal, though, so the
# model can split one effect into a plus on one and a minus on another:
# single weights are read with care, the band totals below more safely.
weights = pd.Series(regression[-1].coef_[0], index=features.columns)
importances = pd.Series(forest.feature_importances_, index=features.columns)

# Feature names are "<channel> <band>", so the band is the second word.
by_band = importances.groupby(importances.index.str.split(" ").str[1]).sum()

top_weights = weights.reindex(weights.abs().sort_values(ascending=False).index).head(TOP)
top_importances = importances.sort_values(ascending=False).head(TOP)

summary = {
    "models": {
        "logistic_regression": str(regression),
        "random_forest": str(forest),
    },
    "training_subjects_only": True,
    "forest_importance_by_band": by_band.sort_values(ascending=False).round(4).to_dict(),
    "top_regression_weights": top_weights.round(4).to_dict(),
    "top_forest_importances": top_importances.round(4).to_dict(),
    "all_regression_weights": weights.round(4).to_dict(),
    "all_forest_importances": importances.round(4).to_dict(),
}
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / "feature_importance.json", "w") as file:
    json.dump(summary, file, indent=2)

print("forest importance by band:")
print(by_band.sort_values(ascending=False).round(3).to_string())
print("\ntop regression weights (+ pushes towards ADHD):")
print(top_weights.round(3).to_string())
print("\ntop forest importances:")
print(top_importances.round(3).to_string())

# *** Plot: two rankings side by side ***
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.5))

# Largest at the top, so the lists read like a ranking.
shown = top_weights[::-1]
left.barh(
    shown.index,
    shown.to_numpy(),
    color=[COLOURS["ADHD"] if value > 0 else COLOURS["Control"] for value in shown],
)
left.axvline(0, color="#0b0b0b", linewidth=0.8)
left.set_title(
    "Logistic regression weights\nblue pushes towards ADHD, orange towards control", loc="left"
)
left.set_xlabel("weight on the standardised feature")

shown = top_importances[::-1]
right.barh(shown.index, shown.to_numpy(), color="#52514e")
right.set_title("Random forest importances\nshare of all splits' improvement", loc="left")
right.set_xlabel("importance")

for ax in (left, right):
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Top {TOP} features, models trained on the 96 training subjects", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
print(f"\nsaved to {RESULTS_DIR}")
