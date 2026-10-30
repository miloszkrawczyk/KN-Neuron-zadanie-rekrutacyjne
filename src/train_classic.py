"""Comparing classical models on band features, with cross-validation by subject.

Only the 96 training subjects are used. Each repeat deals them into
CV_FOLDS groups that never split a subject; every group takes one turn as
the validation set. The held-out test subjects are not loaded at all.

Run with: uv run python -m src.train_classic
"""

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.config import CV_FOLDS, CV_REPEATS, RESULTS_DIR, SEED
from src.data import load_eeg, subject_table
from src.evaluate import score, subject_scores
from src.features import band_features
from src.preprocessing import cut_windows, drop_artifact_windows, filter_recordings
from src.split import split_subjects


def training_data():
    """Band features of the clean windows of training subjects only."""
    df = load_eeg()
    train_ids, _ = split_subjects(subject_table(df))
    windows, labels, subjects = drop_artifact_windows(*cut_windows(filter_recordings(df)))
    keep = np.isin(subjects, train_ids)
    return band_features(windows[keep]), labels[keep], subjects[keep]


def cv_splits(labels, subjects):
    """Every (train, validation) split of every repeat of StratifiedGroupKFold."""
    splits = []
    for repeat in range(CV_REPEATS):
        cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED + repeat)
        # groups=subjects keeps all of a child's windows in one group.
        splits += list(cv.split(np.zeros(len(labels)), labels, groups=subjects))
    return splits


def build_models():
    """The three models compared here, untrained."""
    return {
        # Always answers ADHD, the class with more windows: the floor any
        # real model has to clear.
        "dummy": DummyClassifier(strategy="most_frequent"),
        # Inside a pipeline, the scaler learns its means from the training
        # fold only, so nothing about the validation fold leaks in.
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        # Trees split on thresholds, so they need no scaling.
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
    }


def cross_validate(model, features, labels, subjects):
    """Window- and subject-level scores on every validation fold."""
    folds = []
    for train, validation in cv_splits(labels, subjects):
        # clone gives a fresh, untrained copy, so no fold learns from another.
        fitted = clone(model).fit(features.iloc[train], labels[train])
        probabilities = fitted.predict_proba(features.iloc[validation])[:, 1]
        folds.append(
            {
                "windows": score(labels[validation], probabilities),
                "subjects": score(
                    *subject_scores(labels[validation], probabilities, subjects[validation])
                ),
            }
        )
    return folds


def summarise(folds):
    """Mean and standard deviation of every metric, per level, over all folds."""
    summary = {}
    for level in ("windows", "subjects"):
        table = pd.DataFrame([fold[level] for fold in folds]).drop(columns="confusion_matrix")
        summary[level] = {
            metric: {"mean": round(table[metric].mean(), 4), "std": round(table[metric].std(), 4)}
            for metric in table.columns
        }
    return summary


if __name__ == "__main__":
    features, labels, subjects = training_data()
    models = build_models()
    results = {
        name: cross_validate(model, features, labels, subjects) for name, model in models.items()
    }

    report = {
        "config": {
            "subjects": int(len(np.unique(subjects))),
            "windows": len(labels),
            "features": features.shape[1],
            "window_seconds": config.WINDOW_SECONDS,
            "window_step_seconds": config.WINDOW_STEP_SECONDS,
            "trim_seconds": config.TRIM_SECONDS,
            "filter_band": config.FILTER_BAND,
            "artifact_factor": config.ARTIFACT_FACTOR,
            "feature_welch_seconds": config.FEATURE_WELCH_SECONDS,
            "cv_folds": CV_FOLDS,
            "cv_repeats": CV_REPEATS,
            "decision_threshold": config.DECISION_THRESHOLD,
            "models": {name: str(model) for name, model in models.items()},
        },
        "summary": {name: summarise(folds) for name, folds in results.items()},
        "folds": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "classic_cv.json", "w") as file:
        json.dump(report, file, indent=2)

    for level in ("subjects", "windows"):
        print(f"\n{level}, mean +/- std over {CV_FOLDS * CV_REPEATS} folds")
        for name, summary in report["summary"].items():
            line = f"  {name:20}"
            for metric in ("balanced_accuracy", "roc_auc", "sensitivity", "specificity"):
                value = summary[level][metric]
                line += f"  {metric} {value['mean']:.3f}+/-{value['std']:.3f}"
            print(line)
    print(f"\nsaved to {RESULTS_DIR / 'classic_cv.json'}")
