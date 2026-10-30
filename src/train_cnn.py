"""Training the CNN and scoring it on the same 15 splits as the classical models.

Each split leaves one fold of subjects to score on, exactly as in
train_classic. The remaining subjects are split once more, by subject, into
those the network learns from and those that pick the best epoch. The fold
being scored never takes part in either, so its score stays honest.

Run with: uv run python -m src.train_cnn
"""

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.config import (
    CNN_BATCH_SIZE,
    CNN_EPOCHS,
    CNN_LEARNING_RATE,
    CNN_VALIDATION_SHARE,
    CNN_VARIANTS,
    CV_FOLDS,
    CV_REPEATS,
    RESULTS_DIR,
    SEED,
    SUBJECT_COLUMN,
    TARGET_COLUMN,
    WINDOW_SECONDS,
)
from src.data import load_eeg, subject_table
from src.device import pick_device
from src.evaluate import score, subject_scores
from src.models.cnn import EEGCNN
from src.preprocessing import cut_windows, drop_artifact_windows, filter_recordings
from src.split import split_subjects
from src.train_classic import cv_splits, summarise

DEVICE = pick_device()


def training_windows(window_seconds=WINDOW_SECONDS):
    """Clean windows of the training subjects only, as (windows, labels, subjects).

    Windows overlap by half whatever their length, so the step is half the window.
    """
    df = load_eeg()
    train_ids, _ = split_subjects(subject_table(df))
    cut = cut_windows(filter_recordings(df), window_seconds, step_seconds=window_seconds // 2)
    windows, labels, subjects = drop_artifact_windows(*cut)
    keep = np.isin(subjects, train_ids)
    return windows[keep], labels[keep], subjects[keep]


def normalise(train, *others):
    """Z-score each channel with the mean and spread of the training windows only.

    The same numbers are then applied to every other set. Measuring them on
    validation or test windows as well would let those sets leak into training.
    """
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    return [(windows - mean) / std for windows in (train, *others)]


def predict(model, windows):
    """Probability of ADHD for each window."""
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(windows), 512):
            batch = torch.from_numpy(windows[start : start + 512]).to(DEVICE)
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(probabilities)


def train_model(train_x, train_y, val_x, val_y, seed, dropout=0.0, learning_rate=CNN_LEARNING_RATE):
    """Train for CNN_EPOCHS and return the model at its lowest validation loss.

    Returns the model and a history with both losses per epoch and the best epoch.
    """
    torch.manual_seed(seed)
    model = EEGCNN(dropout).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # The network's version of class_weight: an ADHD window's error is scaled
    # by controls / ADHD, so the class with more windows cannot dominate.
    # float32 from the start: the Mac's graphics chip has no float64.
    pos_weight = torch.tensor(
        (train_y == 0).sum() / (train_y == 1).sum(), dtype=torch.float32, device=DEVICE
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).float()),
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_x = torch.from_numpy(val_x).to(DEVICE)
    val_y = torch.from_numpy(val_y).float().to(DEVICE)

    history = {"train_loss": [], "val_loss": []}
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, CNN_EPOCHS + 1):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimiser.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch_y)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()
        history["train_loss"].append(round(total / len(train_y), 4))
        history["val_loss"].append(round(val_loss, 4))

        # A copy, because the weights keep changing as training goes on.
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch

    model.load_state_dict(best_state)
    # Recorded where the weights are copied, so the reported epoch is always
    # the one whose weights were kept.
    history["best_epoch"] = best_epoch
    return model, history


def cross_validate_cnn(windows, labels, subjects, dropout=0.0, learning_rate=CNN_LEARNING_RATE):
    """Window- and subject-level scores of the CNN on every split of cv_splits."""
    folds = []
    for number, (train, scored) in enumerate(cv_splits(labels, subjects)):
        # Split the training folds once more, by subject, to pick the epoch.
        table = pd.DataFrame({SUBJECT_COLUMN: subjects[train], TARGET_COLUMN: labels[train]})
        learn_ids, val_ids = split_subjects(
            table.drop_duplicates(), test_size=CNN_VALIDATION_SHARE, seed=SEED + number
        )
        learn = train[np.isin(subjects[train], learn_ids)]
        val = train[np.isin(subjects[train], val_ids)]

        learn_x, val_x, scored_x = normalise(windows[learn], windows[val], windows[scored])
        model, history = train_model(
            learn_x,
            labels[learn],
            val_x,
            labels[val],
            seed=SEED + number,
            dropout=dropout,
            learning_rate=learning_rate,
        )
        probabilities = predict(model, scored_x)
        folds.append(
            {
                "windows": score(labels[scored], probabilities),
                "subjects": score(*subject_scores(labels[scored], probabilities, subjects[scored])),
                "history": history,
            }
        )
    return folds


def plot_curves(folds, path):
    """Training and validation loss of every run, to show where overfitting starts."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, CNN_EPOCHS + 1)
    for name, colour in (("train_loss", "#2a78d6"), ("val_loss", "#eb6834")):
        curves = np.array([fold["history"][name] for fold in folds])
        for curve in curves:
            ax.plot(epochs, curve, color=colour, linewidth=0.8, alpha=0.25)
        ax.plot(epochs, curves.mean(axis=0), color=colour, linewidth=2.5, label=f"{name} (mean)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    best = np.mean([fold["history"]["best_epoch"] for fold in folds])
    fig.suptitle(
        f"CNN loss over {len(folds)} runs; best validation epoch on average {best:.1f}",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_variants(reports, path):
    """Mean validation loss of every variant, one line each."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, CNN_EPOCHS + 1)
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
    for (name, report), colour in zip(reports.items(), colours, strict=True):
        curves = np.array([fold["history"]["val_loss"] for fold in report["folds"]])
        ax.plot(epochs, curves.mean(axis=0), color=colour, linewidth=2, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean validation loss over 15 runs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.suptitle("CNN variants: validation loss per epoch", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def variant_report(variant, folds, windows):
    """Configuration, summary and every fold of one variant, ready for JSON."""
    return {
        "config": {
            **variant,
            "windows": windows,
            "device": str(DEVICE),
            "model": str(EEGCNN(variant["dropout"])),
            "epochs": CNN_EPOCHS,
            "batch_size": CNN_BATCH_SIZE,
            "validation_share": CNN_VALIDATION_SHARE,
            "cv_folds": CV_FOLDS,
            "cv_repeats": CV_REPEATS,
            "decision_threshold": config.DECISION_THRESHOLD,
        },
        "summary": summarise(folds),
        "best_epochs": [fold["history"]["best_epoch"] for fold in folds],
        "folds": folds,
    }


if __name__ == "__main__":
    reports = {}
    for name, variant in CNN_VARIANTS.items():
        windows, labels, subjects = training_windows(variant["window_seconds"])
        print(f"{name}: {len(labels)} windows, {len(np.unique(subjects))} subjects, {DEVICE}")
        folds = cross_validate_cnn(
            windows, labels, subjects, variant["dropout"], variant["learning_rate"]
        )
        reports[name] = variant_report(variant, folds, len(labels))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # "4s" is the network trained in the previous step; its full report keeps
    # its own file, so that step's results are rebuilt by this same command.
    with open(RESULTS_DIR / "cnn_cv.json", "w") as file:
        json.dump(reports["4s"], file, indent=2)
    plot_curves(reports["4s"]["folds"], RESULTS_DIR / "cnn_training_curves.png")

    with open(RESULTS_DIR / "cnn_variants.json", "w") as file:
        json.dump(reports, file, indent=2)
    plot_variants(reports, RESULTS_DIR / "cnn_variants_curves.png")

    print("\nsubjects, mean +/- std over 15 folds")
    for name, report in reports.items():
        line = f"  {name:12}"
        for metric in ("balanced_accuracy", "roc_auc", "sensitivity", "specificity"):
            value = report["summary"]["subjects"][metric]
            line += f"  {metric} {value['mean']:.3f}+/-{value['std']:.3f}"
        best = np.mean(report["best_epochs"])
        print(f"{line}  best epoch {best:.1f}")
    print(f"saved to {RESULTS_DIR}")
