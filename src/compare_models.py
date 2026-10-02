"""Compare the five approaches on the test set: tables, ROC curves, calibration and significance tests.

Each model family is represented by its best run on validation macro-F1 in results/experiment_log.csv.
Test scores never influence which run is chosen.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import auc, roc_curve

import plot_style
from config import LABEL_ENGLISH, LABEL_NAMES, NUM_CLASSES, PREDICTIONS_DIR, RESULTS_DIR
from metrics import bootstrap_macro_f1, compute_metrics, figure_path, plot_confusion_matrix

plt = plot_style.plt
LOG_PATH = RESULTS_DIR / "experiment_log.csv"

# prediction file prefix -> name used in tables and figures (names match plot_style.MODEL_COLORS)
MODELS = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "bilstm": "BiLSTM",
    "textcnn": "TextCNN",
    "afriberta": "AfriBERTa",
}


def read_log():
    log = pd.read_csv(LOG_PATH)
    log["timestamp"] = pd.to_datetime(log["timestamp"])
    return log


def selected_runs():
    """Best run per model family on validation macro-F1, among runs that were scored on test."""
    log = read_log()
    tested = log[log["test_macro_f1"].notna()]
    chosen = {}
    for key in MODELS:
        rows = tested[tested["model"] == key]
        if rows.empty:
            continue
        best = rows.sort_values(["val_macro_f1", "timestamp"], ascending=[False, True]).iloc[0]
        chosen[key] = best["run_name"]
    return chosen


def prediction_path(key, run_name, split):
    """Neural runs write <run_name>_<split>.csv; baselines and AfriBERTa write <model>_<split>.csv."""
    own = PREDICTIONS_DIR / f"{run_name}_{split}.csv"
    return own if own.exists() else PREDICTIONS_DIR / f"{key}_{split}.csv"


def load_predictions(split="test"):
    """Return {model name: DataFrame with label, predicted and prob_<class> columns} for the selected runs."""
    predictions = {}
    for key, run_name in selected_runs().items():
        path = prediction_path(key, run_name, split)
        if not path.exists():
            print(f"no {split} predictions for {MODELS[key]} ({path.name}), leaving it out")
            continue
        frame = pd.read_csv(path)
        # train_neural.py writes "prediction"; the baselines and AfriBERTa write "predicted".
        frame = frame.rename(columns={"prediction": "predicted"}).drop(columns=["row"], errors="ignore")
        predictions[MODELS[key]] = frame
    return predictions


def probabilities_of(frame):
    return frame[[f"prob_{n}" for n in LABEL_NAMES]].to_numpy()


def run_details(key, run_name):
    """Parameters, training time and seed spread for a selected run, where the run recorded them."""
    row = read_log().set_index("run_name").loc[run_name]
    details = {"run_name": run_name, "val_macro_f1": float(row["val_macro_f1"]),
               "parameters": row.get("trainable_parameters"), "train_seconds": row.get("train_seconds"),
               "seed_mean_std": "single run"}
    metrics_path = RESULTS_DIR / f"{key}_metrics.json"
    if metrics_path.exists():
        summary = json.load(open(metrics_path)).get("summary")
        if summary and "seeds" in summary:
            details["seed_mean_std"] = f"{summary['test_macro_f1_mean']:.3f} ± {summary['test_macro_f1_std']:.3f}"
    if key in ("naive_bayes", "logistic_regression"):
        details["seed_mean_std"] = "deterministic"
    return details


def comparison_table(predictions):
    rows = []
    chosen = selected_runs()
    for key, name in MODELS.items():
        if name not in predictions:
            continue
        frame = predictions[name]
        metrics = compute_metrics(frame["label"], probabilities_of(frame))
        details = run_details(key, chosen[key])
        low, high = bootstrap_macro_f1(frame["label"], frame["predicted"])
        rows.append({
            "model": name,
            "selected_run": details["run_name"],
            "val_macro_f1": details["val_macro_f1"],
            "test_macro_f1": metrics["macro_f1"],
            "macro_f1_95ci": f"{low:.3f}-{high:.3f}",
            "seed_mean_std": details["seed_mean_std"],
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "log_loss": metrics["log_loss"],
            "roc_auc_macro": metrics["roc_auc_macro"],
            "parameters": details["parameters"],
            "train_seconds": details["train_seconds"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(table.to_string(index=False))
    return table


def per_class_table(predictions):
    rows = {}
    for name, frame in predictions.items():
        metrics = compute_metrics(frame["label"], probabilities_of(frame))
        rows[name] = {label: metrics[f"f1_{label}"] for label in LABEL_NAMES}
    table = pd.DataFrame(rows).T
    table.to_csv(RESULTS_DIR / "per_class_f1.csv")

    fig, ax = plt.subplots(figsize=(10, 3.6))
    width = 0.8 / len(table)
    positions = np.arange(NUM_CLASSES)
    for i, (name, values) in enumerate(table.iterrows()):
        ax.bar(positions + i * width - 0.4 + width / 2, values, width * 0.9,
               color=plot_style.MODEL_COLORS[name], label=name)
    ax.set_xticks(positions, [f"{n}\n({LABEL_ENGLISH[n]})" for n in LABEL_NAMES])
    ax.set_ylabel("test F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 on the test set")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5)
    ax.grid(axis="x", visible=False)
    plot_style.save(fig, figure_path("per_class_f1.png"))
    return table


def plot_roc_curves(predictions):
    """Macro-average one-vs-rest ROC curve per model."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    grid = np.linspace(0, 1, 500)
    for name, frame in predictions.items():
        probabilities = probabilities_of(frame)
        true_positive_rates = []
        for label_id in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(frame["label"] == label_id, probabilities[:, label_id])
            true_positive_rates.append(np.interp(grid, fpr, tpr))
        mean_tpr = np.mean(true_positive_rates, axis=0)
        ax.plot(grid, mean_tpr, color=plot_style.MODEL_COLORS[name], label=f"{name} (AUC {auc(grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], color=plot_style.GRID, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_xlabel("false positive rate (log scale)")
    ax.set_ylabel("true positive rate")
    ax.set_title("Macro-average ROC (one-vs-rest)")
    ax.legend(fontsize=8, loc="lower right")
    plot_style.save(fig, figure_path("roc_curves.png"))


def plot_calibration(predictions, bins=10):
    """Reliability of the top-class confidence. Log loss punishes over-confident mistakes."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    rows = []
    for name, frame in predictions.items():
        probabilities = probabilities_of(frame)
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == frame["label"].to_numpy()
        edges = np.linspace(0, 1, bins + 1)
        which = np.clip(np.digitize(confidence, edges) - 1, 0, bins - 1)
        centers, accuracies, error = [], [], 0.0
        for b in range(bins):
            chosen = which == b
            if chosen.sum() >= 20:
                centers.append(confidence[chosen].mean())
                accuracies.append(correct[chosen].mean())
            if chosen.any():
                error += chosen.mean() * abs(confidence[chosen].mean() - correct[chosen].mean())
        rows.append({"model": name, "expected_calibration_error": round(error, 4)})
        ax.plot(centers, accuracies, marker="o", markersize=4, color=plot_style.MODEL_COLORS[name],
                label=f"{name} (ECE {error:.3f})")
    ax.plot([0, 1], [0, 1], color=plot_style.TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Calibration on the test set")
    ax.legend(fontsize=8)
    plot_style.save(fig, figure_path("calibration.png"))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "calibration.csv", index=False)


def mcnemar_tests(predictions, table):
    """Exact McNemar test for every pair of models: do they make significantly different errors?"""
    order = table.sort_values("test_macro_f1", ascending=False)["model"].tolist()
    rows = []
    for i, first in enumerate(order):
        for second in order[i + 1:]:
            first_correct = predictions[first]["predicted"] == predictions[first]["label"]
            second_correct = predictions[second]["predicted"] == predictions[second]["label"]
            only_first = int((first_correct & ~second_correct).sum())
            only_second = int((~first_correct & second_correct).sum())
            total = only_first + only_second
            p_value = binomtest(only_first, total, 0.5).pvalue if total else 1.0
            rows.append({"better_model": first, "compared_with": second, "only_better_correct": only_first,
                         "only_other_correct": only_second, "p_value": float(f"{p_value:.3g}")})
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "significance.csv", index=False)
    print(result.to_string(index=False))


def plot_experiment_progression():
    """Validation macro-F1 of every logged run, in the order it was run, grouped by model."""
    log = read_log()
    log = log[~log["run_name"].str.contains("-")]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    start, ticks, labels = 0, [], []
    for key, name in MODELS.items():
        rows = log[log["model"] == key].sort_values("timestamp")
        if rows.empty:
            continue
        x = np.arange(start, start + len(rows))
        color = plot_style.MODEL_COLORS[name]
        ax.plot(x, rows["val_macro_f1"], marker="o", markersize=6, linewidth=1.2, color=color, label=name)
        ax.step(x, rows["val_macro_f1"].cummax(), where="post", color=color, linewidth=1, linestyle="--", alpha=0.7)
        ticks.extend(x)
        labels.extend(rows["run_name"].str.replace("bilstm_", "").str.replace("textcnn_", ""))
        start += len(rows) + 1
    ax.set_xticks(ticks, labels, rotation=90, fontsize=8)
    ax.set_ylabel("validation macro-F1")
    ax.set_title("Validation macro-F1 of every logged run (dashed line: best so far)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    plot_style.save(fig, figure_path("experiment_progression.png"))


def plot_all_confusions(predictions):
    columns = 3
    rows = (len(predictions) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4.1 * rows), squeeze=False)
    for ax, (name, frame) in zip(axes.ravel(), predictions.items()):
        plot_confusion_matrix(frame["label"], frame["predicted"], name, None, ax=ax)
    for ax in axes.ravel()[len(predictions):]:
        ax.axis("off")
    fig.tight_layout()
    plot_style.save(fig, figure_path("confusion_all_models.png"))


def load_history(key, run_name):
    """Per-epoch history: train_neural.py writes CSV, train_transformer.py writes JSON per experiment."""
    csv_path = RESULTS_DIR / f"{run_name}_history.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    json_path = RESULTS_DIR / "history" / f"{run_name}.json"
    if json_path.exists():
        return pd.DataFrame(json.load(open(json_path)))
    return None


def plot_all_learning_curves():
    """Learning curves of the selected run of each neural model, side by side."""
    chosen = selected_runs()
    keys = [k for k in ("bilstm", "textcnn", "afriberta") if k in chosen]
    histories = [(k, load_history(k, chosen[k])) for k in keys]
    histories = [(k, h) for k, h in histories if h is not None]
    if not histories:
        return
    fig, axes = plt.subplots(2, len(histories), figsize=(4.2 * len(histories), 6), squeeze=False)
    for column, (key, history) in enumerate(histories):
        name = MODELS[key]
        top, bottom = axes[0, column], axes[1, column]
        top.plot(history["epoch"], history["train_loss"], marker="o", markersize=4,
                 color=plot_style.SERIES_COLORS[0], label="train")
        top.plot(history["epoch"], history["val_loss"], marker="o", markersize=4,
                 color=plot_style.SERIES_COLORS[1], label="validation")
        top.set_title(f"{name} ({chosen[key]})")
        top.set_ylabel("cross-entropy loss")
        top.legend(fontsize=8)
        bottom.plot(history["epoch"], history["val_macro_f1"], marker="o", markersize=4,
                    color=plot_style.MODEL_COLORS[name])
        bottom.set_xlabel("epoch")
        bottom.set_ylabel("validation macro-F1")
        for ax in (top, bottom):
            ax.set_xticks(history["epoch"])
    fig.tight_layout()
    plot_style.save(fig, figure_path("learning_curves_all.png"))


def main():
    chosen = selected_runs()
    with open(RESULTS_DIR / "selected_runs.json", "w") as f:
        json.dump({MODELS[k]: v for k, v in chosen.items()}, f, indent=2)
    print("selected runs:", chosen)
    predictions = load_predictions()
    table = comparison_table(predictions)
    per_class_table(predictions)
    plot_roc_curves(predictions)
    plot_calibration(predictions)
    mcnemar_tests(predictions, table)
    plot_experiment_progression()
    plot_all_confusions(predictions)
    plot_all_learning_curves()


if __name__ == "__main__":
    main()
