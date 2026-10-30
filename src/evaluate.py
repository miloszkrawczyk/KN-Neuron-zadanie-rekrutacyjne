"""Scoring a model the same way everywhere, on windows and on subjects.

A diagnosis is made for a child, not for a 4 second window, so the
subject-level score is the one that matters. It averages the model's
probability over all of a subject's windows and thresholds that once.
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from src.config import DECISION_THRESHOLD


def score(labels, probabilities, threshold=DECISION_THRESHOLD):
    """Balanced accuracy, sensitivity, specificity, F1, ROC-AUC and the confusion matrix.

    Returns plain numbers, ready to be written to JSON.
    """
    predicted = (probabilities >= threshold).astype(int)
    # labels=[0, 1] keeps the table 2x2 even when a model only ever predicts
    # one class, as the always-ADHD baseline does.
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn)  # share of ADHD that was found
    specificity = tn / (tn + fp)  # share of controls correctly left alone
    return {
        # The mean of the two, so a model cannot score well by favouring the
        # class with more windows: always-ADHD gets 0.5, not 0.56.
        "balanced_accuracy": round(float((sensitivity + specificity) / 2), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
        # Rows are the truth (control, ADHD), columns the prediction.
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def subject_scores(labels, probabilities, subjects):
    """Mean probability over each subject's windows, with the subject's label.

    Takes and returns labels first, like score() and scikit-learn, so the
    two chain as score(*subject_scores(labels, probabilities, subjects)).
    """
    per_subject = (
        pd.DataFrame({"label": labels, "probability": probabilities, "subject": subjects})
        .groupby("subject")
        .mean()
    )
    return per_subject["label"].astype(int).to_numpy(), per_subject["probability"].to_numpy()
