"""Append one row per training run to results/experiment_log.csv.

Every model logs here, and the comparison script reads the file, so the columns are fixed.
A lock file keeps runs that finish at the same time from writing over each other.
"""

import csv
import fcntl
import json
from datetime import datetime, timezone

from config import RESULTS_DIR, make_output_dirs

LOG_PATH = RESULTS_DIR / "experiment_log.csv"
LOCK_PATH = RESULTS_DIR / ".experiment_log.lock"

COLUMNS = [
    "timestamp",
    "run_name",
    "model",
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "trainable_parameters",
    "epochs_run",
    "train_seconds",
    "device",
    "settings",
]


def log_experiment(run_name, model, validation, test, settings, **extra):
    """validation and test are score dicts with accuracy, macro_f1 and weighted_f1."""
    make_output_dirs()
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "model": model,
        "val_accuracy": validation["accuracy"],
        "val_macro_f1": validation["macro_f1"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_weighted_f1": test["weighted_f1"],
        "settings": json.dumps(settings, sort_keys=True),
        **{k: v for k, v in extra.items() if k in COLUMNS},
    }
    with open(LOCK_PATH, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        is_new = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        fcntl.flock(lock, fcntl.LOCK_UN)
    print(f"logged {run_name} to {LOG_PATH}")
