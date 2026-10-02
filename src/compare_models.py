"""Which models are compared, and how to load their test predictions."""

import pandas as pd

from config import PREDICTIONS_DIR

# prediction file prefix -> name used in tables and figures (names match plot_style.MODEL_COLORS)
MODELS = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "bilstm": "BiLSTM + attention",
    "textcnn": "TextCNN",
    "afriberta": "AfriBERTa",
}


def load_predictions(split="test"):
    """Return {model name: DataFrame with label, predicted and prob_<class> columns} for models that have results."""
    predictions = {}
    for key, name in MODELS.items():
        path = PREDICTIONS_DIR / f"{key}_{split}.csv"
        if not path.exists():
            print(f"no {split} predictions for {name} ({path.name}), leaving it out")
            continue
        frame = pd.read_csv(path)
        # train_neural.py writes "prediction"; the baselines and AfriBERTa write "predicted".
        frame = frame.rename(columns={"prediction": "predicted"}).drop(columns=["row"], errors="ignore")
        predictions[name] = frame
    return predictions
