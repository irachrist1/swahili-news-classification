"""Append one row per training run to results/experiment_log.csv.

Every model logs here, and the comparison script reads the file, so the columns are fixed.
A lock file keeps runs that finish at the same time from writing over each other.
"""

import csv
import json
from datetime import datetime, timezone

from config import RESULTS_DIR, make_output_dirs

try:
    import fcntl
except ImportError:  # Windows has no fcntl; msvcrt provides an equivalent lock.
    fcntl = None
    import msvcrt

LOG_PATH = RESULTS_DIR / "experiment_log.csv"
LOCK_PATH = RESULTS_DIR / ".experiment_log.lock"

COLUMNS = [
    "timestamp",
    "run_name",
    "model",
    "change",
    "hypothesis",
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


def log_experiment(run_name, model, change, hypothesis, validation, test, settings, **extra):
    """Log one run. change says what differs from the reference run, hypothesis why it might help.

    validation and test are score dicts with accuracy, macro_f1 and weighted_f1.
    """
    make_output_dirs()
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "model": model,
        "change": change,
        "hypothesis": hypothesis,
        "val_accuracy": validation["accuracy"],
        "val_macro_f1": validation["macro_f1"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_weighted_f1": test["weighted_f1"],
        "settings": json.dumps(settings, sort_keys=True),
        **{k: v for k, v in extra.items() if k in COLUMNS},
    }
    with open(LOCK_PATH, "w") as lock:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        else:
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        is_new = not LOG_PATH.exists()
        if not is_new:
            with open(LOG_PATH, newline="") as f:
                header = next(csv.reader(f), [])
            if header != COLUMNS:
                raise ValueError(f"{LOG_PATH} has columns {header}, expected {COLUMNS}")
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_UN)
    print(f"logged {run_name} to {LOG_PATH}")
