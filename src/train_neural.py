"""Train a word-level neural classifier and save its metrics, history and test predictions.

Examples (run from the project root):
    python src/train_neural.py --model bilstm
    python src/train_neural.py --model bilstm --embeddings random --run-name bilstm_random \
        --change "random embeddings instead of fastText" --hypothesis "pretrained vectors help with little data"
    python src/train_neural.py --model bilstm --limit 500 --epochs 1      # quick smoke test

Outputs, named after --run-name:
    results/<run>_metrics.json        settings, validation and test scores
    results/<run>_history.csv         loss and scores per epoch
    results/predictions/<run>_test.csv  true label, prediction and class probabilities
    figures/<run>_training_curves.png
    results/<run>.pt                  best weights (not committed)
    results/experiment_log.csv        one row appended per run (skipped with --limit)
"""

import argparse
import json
import time

import numpy as np
<<<<<<< HEAD
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from config import NUM_CLASSES, RESULTS_DIR, get_device, set_seed
from data import load_splits
from metrics import (
    compute_metrics,
    figure_path,
    log_experiment,
    plot_confusion_matrix,
    plot_learning_curves,
    save_metrics,
    save_predictions,
)
from neural_models import BiLSTMClassifier, TextCNN
from preprocess import clean_splits
from sequence_data import Vocabulary, load_fasttext_matrix, make_loader

FINAL_SEEDS = [42, 7, 123]

PLANS = {
    "bilstm": {
        "owner": "Rene",
        "name": "BiLSTM + attention",
        "prefix": "R",
        "start": {"pooling": "last", "max_length": 256, "embeddings": "random", "class_weights": False,
                  "hidden_size": 128, "dropout": 0.3, "learning_rate": 1e-3, "epochs": 8, "batch_size": 64},
        "first": ("BiLSTM, random embeddings, final hidden state, first 256 words",
                  "A recurrent model that reads word order should match the linear baselines."),
        "steps": [
            ({"pooling": "attention"}, "Attention pooling instead of the final hidden state",
             "Final states forget the start of long articles; attention can weight the key words wherever they are."),
            ({"max_length": 512}, "Read the first 512 words instead of 256",
             "54% of articles are longer than 256 words, so truncation drops evidence."),
            ({"embeddings": "fasttext"}, "Initialise with fastText Swahili vectors",
             "Pretrained vectors give rare words a useful starting point (52% of the vocabulary appears once)."),
            ({"class_weights": True}, "Class-weighted cross-entropy",
             "Weighting minority classes should raise recall on 'afya' and 'uchumi'."),
        ],
    },
    "textcnn": {
        "owner": "Thierry",
        "name": "TextCNN",
        "prefix": "C",
        "start": {"max_length": 256, "embeddings": "random", "class_weights": False, "num_filters": 100,
                  "kernel_sizes": [3, 4, 5], "dropout": 0.5, "learning_rate": 1e-3, "epochs": 8, "batch_size": 64},
        "first": ("TextCNN, random embeddings, kernels 3/4/5, first 256 words",
                  "Topic is signalled by local key phrases, which convolutions detect regardless of position."),
        "steps": [
            ({"max_length": 512}, "Read the first 512 words instead of 256",
             "Max-pooling ignores position, so more text should only add evidence."),
            ({"embeddings": "fasttext"}, "Initialise with fastText Swahili vectors",
             "Pretrained vectors should help most for a model that sees each phrase only locally."),
            ({"class_weights": True}, "Class-weighted cross-entropy",
             "Weighting minority classes should raise recall on 'afya' and 'uchumi'."),
            ({"num_filters": 200, "kernel_sizes": [2, 3, 4, 5]}, "200 filters and kernel sizes 2 to 5",
             "Two-word phrases such as 'benki kuu' and more filters give the model more phrase detectors."),
        ],
    },
}


def build_model(model_type, config, vocabulary, embedding_cache):
    embeddings = None
    if config["embeddings"] == "fasttext":
        if "matrix" not in embedding_cache:
            embedding_cache["matrix"], embedding_cache["coverage"] = load_fasttext_matrix(vocabulary)
        embeddings = embedding_cache["matrix"]
    if model_type == "bilstm":
        return BiLSTMClassifier(len(vocabulary), NUM_CLASSES, hidden_size=config["hidden_size"],
                                dropout=config["dropout"], pooling=config["pooling"], embeddings=embeddings)
    return TextCNN(len(vocabulary), NUM_CLASSES, num_filters=config["num_filters"],
                   kernel_sizes=config["kernel_sizes"], dropout=config["dropout"], embeddings=embeddings)


def predict(model, loader, device, loss_function):
    model.eval()
    probabilities, total_loss = [], 0.0
    with torch.no_grad():
        for ids, labels in loader:
            logits = model(ids.to(device))
            total_loss += loss_function(logits, labels.to(device)).item() * len(labels)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    in_length_order = np.concatenate(probabilities)
    original_order = np.empty_like(in_length_order)
    original_order[loader.batch_sampler.order] = in_length_order
    return original_order, total_loss / len(loader.dataset)


def train_one(model_type, config, splits, vocabulary, embedding_cache, seed, device):
    """Train with early stopping on validation macro-F1 and return the best model's outputs."""
    set_seed(seed)
    train, validation, test = splits
    train_loader = make_loader(train, vocabulary, config["max_length"], config["batch_size"], shuffle=True)
    val_loader = make_loader(validation, vocabulary, config["max_length"], 256, shuffle=False)
    test_loader = make_loader(test, vocabulary, config["max_length"], 256, shuffle=False)

    model = build_model(model_type, config, vocabulary, embedding_cache).to(device)
    weights = None
    if config["class_weights"]:
        weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train["label"])
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    train_loss_function = nn.CrossEntropyLoss(weight=weights)
    eval_loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

    history, best_f1, best_state, patience = [], -1, None, 0
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = train_loss_function(model(ids), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * len(labels)

        val_probabilities, val_loss = predict(model, val_loader, device, eval_loss_function)
        val_f1 = compute_metrics(validation["label"], val_probabilities)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": running_loss / len(train), "val_loss": val_loss,
                        "val_macro_f1": val_f1})
        print(f"  epoch {epoch}: train_loss={history[-1]['train_loss']:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                break

    model.load_state_dict(best_state)
    val_probabilities, _ = predict(model, val_loader, device, eval_loss_function)
    test_probabilities, _ = predict(model, test_loader, device, eval_loss_function)
    return {
        "model": model,
        "history": history,
        "val_probabilities": val_probabilities,
        "test_probabilities": test_probabilities,
        "val_metrics": compute_metrics(validation["label"], val_probabilities),
        "seconds": round(time.time() - start, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def run_plan(model_type, device):
    plan = PLANS[model_type]
    train, validation, test = load_splits()
    train, validation, test, _ = clean_splits(train, validation, test)
    splits = (train, validation, test)
    vocabulary = Vocabulary(train["clean_text"])
    print(f"vocabulary size: {len(vocabulary)}")
    embedding_cache, step_results = {}, {}
    history_dir = RESULTS_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    def run_step(experiment_id, config, change, hypothesis):
        print(f"[{experiment_id}] {change}")
        result = train_one(model_type, config, splits, vocabulary, embedding_cache, 42, device)
        step_results[json.dumps(config, sort_keys=True)] = result
        extra = {"config": config, "epochs_run": len(result["history"]), "seconds": result["seconds"],
                 "parameters": result["parameters"]}
        log_experiment(experiment_id, plan["name"], plan["owner"], change, hypothesis, result["val_metrics"],
                       extra=extra)
        with open(history_dir / f"{experiment_id}.json", "w") as f:
            json.dump(result["history"], f)
        return result["val_metrics"]["macro_f1"]

    best_config = dict(plan["start"])
    best_f1 = run_step(f"{plan['prefix']}01", best_config, *plan["first"])
    for number, (change_values, change, hypothesis) in enumerate(plan["steps"], start=2):
        candidate = {**best_config, **change_values}
        score = run_step(f"{plan['prefix']}{number:02d}", candidate, change, hypothesis)
        if score > best_f1:
            best_config, best_f1 = candidate, score
            print(f"  kept (val macro-F1 {score:.4f})")
        else:
            print(f"  not kept ({score:.4f} <= {best_f1:.4f})")

    final_id = f"{plan['prefix']}{len(plan['steps']) + 2:02d}"
    seed_scores, main_run = [], None
    for seed in FINAL_SEEDS:
        print(f"[{final_id}] final configuration, seed {seed}")
        if seed == 42:
            result = step_results[json.dumps(best_config, sort_keys=True)]
        else:
            result = train_one(model_type, best_config, splits, vocabulary, embedding_cache, seed, device)
        test_metrics = compute_metrics(test["label"], result["test_probabilities"])
        seed_scores.append({"seed": seed, "val_macro_f1": result["val_metrics"]["macro_f1"],
                            "test_macro_f1": test_metrics["macro_f1"], "test_accuracy": test_metrics["accuracy"]})
        if main_run is None:
            main_run = (result, test_metrics)

    result, test_metrics = main_run
    test_scores = [s["test_macro_f1"] for s in seed_scores]
    summary = {"config": best_config, "seeds": seed_scores, "test_macro_f1_mean": round(float(np.mean(test_scores)), 4),
               "test_macro_f1_std": round(float(np.std(test_scores)), 4), "seconds": result["seconds"],
               "parameters": result["parameters"], "fasttext_coverage": embedding_cache.get("coverage")}
    log_experiment(final_id, plan["name"], plan["owner"],
                   f"Final configuration, {len(FINAL_SEEDS)} seeds (test macro-F1 {summary['test_macro_f1_mean']} "
                   f"+/- {summary['test_macro_f1_std']})",
                   "Selected configuration, evaluated once on test.", result["val_metrics"], test_metrics, extra=summary)

    save_predictions(model_type, "validation", validation, result["val_probabilities"])
    save_predictions(model_type, "test", test, result["test_probabilities"])
    save_metrics(model_type, {"validation": result["val_metrics"], "test": test_metrics, "summary": summary})
    with open(history_dir / f"{final_id}.json", "w") as f:
        json.dump(result["history"], f)
    plot_learning_curves(result["history"], f"{plan['name']} learning curves",
                         figure_path(f"learning_curves_{model_type}.png"))
    plot_confusion_matrix(test["label"], result["test_probabilities"].argmax(axis=1), f"{plan['name']} (test)",
                          figure_path(f"confusion_{model_type}.png"))
    torch.save({"state_dict": result["model"].state_dict(), "config": best_config, "words": vocabulary.words},
               RESULTS_DIR / f"{model_type}_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bilstm", "textcnn"], required=True)
    parser.add_argument("--device", default=None, help="cpu, cuda or mps; defaults to the fastest available")
    args = parser.parse_args()
    run_plan(args.model, torch.device(args.device) if args.device else get_device())
=======
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
from experiment_log import log_experiment
from neural_models import build_model
from preprocess import clean_splits
from sequence_data import build_vocab, load_fasttext, make_loader

plt = plot_style.plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="bilstm")
    parser.add_argument("--run-name", default=None, help="defaults to the model name")
    parser.add_argument("--change", default="", help="what this run changes, for the experiment log")
    parser.add_argument("--hypothesis", default="", help="why the change might help")
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
        "change": args.change,
        "hypothesis": args.hypothesis,
        "settings": {k: v for k, v in vars(args).items() if k not in ("run_name", "change", "hypothesis")},
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
    if not args.limit:
        log_experiment(
            args.run_name,
            args.model,
            args.change,
            args.hypothesis,
            summary["validation"],
            summary["test"],
            summary["settings"],
            trainable_parameters=trainable,
            epochs_run=summary["epochs_run"],
            train_seconds=train_seconds,
            device=str(device),
        )
    print(json.dumps({"validation": summary["validation"], "test": summary["test"]}, indent=2))


if __name__ == "__main__":
    main()
>>>>>>> 3bc0998f9527cbe0c046ce4f14e8cf014cd38e2b
