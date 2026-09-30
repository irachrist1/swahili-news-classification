"""Train a word-level neural classifier and save its metrics, history and test predictions.

Examples (run from the project root):
    python src/train_neural.py --model bilstm
    python src/train_neural.py --model bilstm --embeddings random --run-name bilstm_random
    python src/train_neural.py --model bilstm --limit 500 --epochs 1      # quick smoke test

Outputs, named after --run-name:
    results/<run>_metrics.json        settings, validation and test scores
    results/<run>_history.csv         loss and scores per epoch
    results/predictions/<run>_test.csv  true label, prediction and class probabilities
    figures/<run>_training_curves.png
    results/<run>.pt                  best weights (not committed)
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

import plot_style
from config import (
    FIGURES_DIR,
    LABEL_NAMES,
    NUM_CLASSES,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    SEED,
    get_device,
    make_output_dirs,
    set_seed,
)
from data import load_splits
from neural_models import build_model
from preprocess import clean_splits
from sequence_data import build_vocab, load_fasttext, make_loader

plt = plot_style.plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="bilstm")
    parser.add_argument("--run-name", default=None, help="defaults to the model name")
    parser.add_argument("--preprocess", choices=["full", "raw"], default="full")
    parser.add_argument("--stopwords", action="store_true", help="remove Swahili stopwords")
    parser.add_argument("--embeddings", choices=["fasttext", "random"], default="fasttext")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=400)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--pooling", choices=["attention", "max", "mean"], default="attention")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--class-weights", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3, help="epochs without validation gain")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=None, help="use only N articles per split")
    args = parser.parse_args()
    args.run_name = args.run_name or args.model
    return args


def model_options(args):
    """Constructor arguments for each model in neural_models.MODELS."""
    shared = {"dropout": args.dropout, "freeze_embeddings": args.freeze_embeddings}
    if args.model == "bilstm":
        return {**shared, "hidden_size": args.hidden_size, "num_layers": args.num_layers, "pooling": args.pooling}
    return shared


def score(y_true, y_pred):
    # TODO: switch to metrics.py from baseline-models once it is merged, so every model is scored the same way.
    labels = list(range(NUM_CLASSES))
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0), 4),
        "f1_per_class": {name: round(v, 4) for name, v in zip(LABEL_NAMES, per_class)},
    }


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over loader. Trains when an optimizer is given. Returns loss, labels, probabilities."""
    model.train(optimizer is not None)
    total_loss, labels, probs = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for ids, lengths, y in loader:
            ids, y = ids.to(device), y.to(device)
            logits = model(ids, lengths)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            labels.append(y.cpu().numpy())
            probs.append(torch.softmax(logits.float(), dim=1).detach().cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(labels), np.concatenate(probs)


def plot_history(history, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(history["epoch"], history["train_loss"], label="train", color=plot_style.SERIES_COLORS[0])
    axes[0].plot(history["epoch"], history["val_loss"], label="validation", color=plot_style.SERIES_COLORS[1])
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_macro_f1"], color=plot_style.SERIES_COLORS[1])
    best = history["val_macro_f1"].idxmax()
    axes[1].scatter(history.loc[best, "epoch"], history.loc[best, "val_macro_f1"], color=plot_style.TEXT, zorder=3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation macro-F1")
    axes[1].set_title("Validation macro-F1 (dot = kept)")
    fig.suptitle(title, fontweight="bold")
    plot_style.save(fig, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    make_output_dirs()
    device = get_device()
    print(f"run={args.run_name} device={device}")

    train, validation, test = load_splits()
    if args.limit:
        train, validation, test = (f.head(args.limit).copy() for f in (train, validation, test))
    train, validation, test, _ = clean_splits(
        train, validation, test, mode=args.preprocess, remove_stopwords=args.stopwords
    )

    vocab = build_vocab(train["clean_text"], min_count=args.min_count)
    embeddings, coverage = None, None
    if args.embeddings == "fasttext":
        embeddings, coverage = load_fasttext(vocab)
        print(f"vocab={len(vocab):,} fastText coverage={coverage:.1%}")

    loaders = {
        name: make_loader(frame, vocab, args.max_len, args.batch_size, shuffle=(name == "train"))
        for name, frame in [("train", train), ("validation", validation), ("test", test)]
    }

    model = build_model(args.model, len(vocab), NUM_CLASSES, embeddings, **model_options(args)).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters={trainable:,}")

    weights = None
    if args.class_weights == "balanced":
        present = np.unique(train["label"])
        weights = np.ones(NUM_CLASSES, dtype=np.float32)
        weights[present] = compute_class_weight("balanced", classes=present, y=train["label"])
        weights = torch.tensor(weights, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    weights_path = RESULTS_DIR / f"{args.run_name}.pt"
    history, best_f1, stale = [], -1.0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        val_loss, val_true, val_probs = run_epoch(model, loaders["validation"], loss_fn, device)
        val_scores = score(val_true, val_probs.argmax(1))
        scheduler.step(val_scores["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": val_scores["accuracy"],
                "val_macro_f1": val_scores["macro_f1"],
                "seconds": round(time.time() - start, 1),
            }
        )
        print(
            f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_scores['accuracy']:.4f}  val_macro_f1={val_scores['macro_f1']:.4f}"
        )
        if val_scores["macro_f1"] > best_f1:
            best_f1, stale = val_scores["macro_f1"], 0
            torch.save(model.state_dict(), weights_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: no validation gain for {args.patience} epochs")
                break
    train_seconds = round(time.time() - start, 1)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    _, val_true, val_probs = run_epoch(model, loaders["validation"], loss_fn, device)
    _, test_true, test_probs = run_epoch(model, loaders["test"], loss_fn, device)
    test_pred = test_probs.argmax(1)

    history = pd.DataFrame(history)
    history.to_csv(RESULTS_DIR / f"{args.run_name}_history.csv", index=False)
    plot_history(history, FIGURES_DIR / f"{args.run_name}_training_curves.png", args.run_name)

    predictions = pd.DataFrame(
        {"row": np.arange(len(test_true)), "label": test_true, "prediction": test_pred}
    )
    for i, name in enumerate(LABEL_NAMES):
        predictions[f"prob_{name}"] = test_probs[:, i].round(5)
    predictions.to_csv(PREDICTIONS_DIR / f"{args.run_name}_test.csv", index=False)

    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "settings": {k: v for k, v in vars(args).items() if k != "run_name"},
        "vocab_size": len(vocab),
        "fasttext_coverage": None if coverage is None else round(coverage, 4),
        "trainable_parameters": trainable,
        "best_epoch": int(history.loc[history["val_macro_f1"].idxmax(), "epoch"]),
        "epochs_run": len(history),
        "train_seconds": train_seconds,
        "device": str(device),
        "validation": score(val_true, val_probs.argmax(1)),
        "test": score(test_true, test_pred),
    }
    with open(RESULTS_DIR / f"{args.run_name}_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"validation": summary["validation"], "test": summary["test"]}, indent=2))


if __name__ == "__main__":
    main()
