"""Evaluation metrics and shared result plots. Runs are logged with experiment_log.py."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

import plot_style
from config import FIGURES_DIR, LABEL_ENGLISH, LABEL_NAMES, NUM_CLASSES, PREDICTIONS_DIR, RESULTS_DIR

plt = plot_style.plt


def compute_metrics(y_true, probabilities):
    """Macro-F1 is the main metric because the classes are imbalanced 13 to 1."""
    y_true = np.asarray(y_true)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    y_pred = probabilities.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0
    )
    result = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "log_loss": log_loss(y_true, probabilities, labels=range(NUM_CLASSES)),
        "roc_auc_macro": roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"),
    }
    for name, value in zip(LABEL_NAMES, f1):
        result[f"f1_{name}"] = value
    return {key: round(float(value), 4) for key, value in result.items()}


def bootstrap_macro_f1(y_true, y_pred, rounds=1000, seed=0):
    """95% confidence interval for macro-F1 by resampling the test set."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(rounds):
        index = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[index], y_pred[index], average="macro"))
    return round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)


def save_predictions(model_key, split, frame, probabilities):
    """Keep test probabilities so the comparison and error analysis scripts can reuse them."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(probabilities, columns=[f"prob_{n}" for n in LABEL_NAMES])
    output.insert(0, "label", frame["label"].to_numpy())
    output.insert(1, "predicted", probabilities.argmax(axis=1))
    output.to_csv(PREDICTIONS_DIR / f"{model_key}_{split}.csv", index=False)


def save_metrics(model_key, metrics):
    with open(RESULTS_DIR / f"{model_key}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(y_true, y_pred, title, path, ax=None):
    matrix = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES), normalize="true")
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(matrix, cmap=plot_style.BLUES, vmin=0, vmax=1)
    ax.grid(False)
    ticks = [f"{n}\n({LABEL_ENGLISH[n][:5]}.)" for n in LABEL_NAMES]
    ax.set_xticks(range(NUM_CLASSES), ticks, fontsize=7)
    ax.set_yticks(range(NUM_CLASSES), LABEL_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            value = matrix[i, j]
            if value >= 0.005:
                color = "white" if value > 0.55 else plot_style.TEXT
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)
    if own_figure:
        plot_style.save(fig, path)


def plot_learning_curves(history, title, path):
    """history is a list of dicts with epoch, train_loss, val_loss and val_macro_f1."""
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(frame["epoch"], frame["train_loss"], marker="o", markersize=4,
                 color=plot_style.SERIES_COLORS[0], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], marker="o", markersize=4,
                 color=plot_style.SERIES_COLORS[1], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["val_macro_f1"], marker="o", markersize=4,
                 color=plot_style.SERIES_COLORS[1])
    best = frame["val_macro_f1"].idxmax()
    axes[1].annotate(f"best {frame.loc[best, 'val_macro_f1']:.3f}",
                     (frame.loc[best, "epoch"], frame.loc[best, "val_macro_f1"]),
                     textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation macro-F1")
    axes[1].set_title("Validation macro-F1")
    fig.suptitle(title, fontweight="bold", fontsize=11)
    fig.tight_layout()
    plot_style.save(fig, path)


def figure_path(name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name
