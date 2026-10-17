"""
Helpers to measure a model and to keep a record of every experiment.

Each experiment is saved in three places:
  1. reports/experiment_log.csv   - the experiment table used in the report
  2. logs/<name>.log              - a plain text log (what ran, how long it took)
  3. logs/tensorboard/<run>/      - TensorBoard files (hyperparameters, metrics and curves)

To look at the TensorBoard files, run in a terminal:   tensorboard --logdir logs/tensorboard
"""
import logging
import os
import shutil
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # hide TensorFlow's long start-up messages

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from tensorboard.plugins.hparams import api as hp

# Columns of the experiment table
TABLE_COLUMNS = ["exp_id", "step", "model", "hyperparameters", "features", "split", "eval_set",
                 "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_seconds", "observations"]


def evaluate_binary(y_true, y_score, y_pred=None, threshold=0.5):
    """
    Compute the metrics used in the project. Precision, recall and F1 are for the
    positive class (diabetes). If no predictions are given, a score >= threshold means 'diabetes'.
    ROC-AUC and PR-AUC use the scores, so they do not depend on the threshold.
    """
    y_score = np.asarray(y_score)
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


class ExperimentTracker:
    """Records experiments in the CSV table, the text log and TensorBoard."""

    def __init__(self, csv_path, tensorboard_dir, log_file):
        self.csv_path = Path(csv_path)
        self.tb_dir = Path(tensorboard_dir)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # text logger: prints to the notebook and writes to the log file
        self.logger = logging.getLogger(f"tracker_{Path(log_file).stem}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
        for handler in (logging.FileHandler(log_file, mode="w"), logging.StreamHandler()):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _run_dir(self, run_name):
        """A fresh TensorBoard folder for this run (old files of the same run are removed first)."""
        run_dir = self.tb_dir / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        return run_dir

    def log_experiment(self, exp_id, step, model, params, features, split, eval_set, metrics, seconds, observations=""):
        """Add one experiment to the CSV table, the text log and TensorBoard."""
        # 1) CSV table (a row with the same exp_id is replaced, so re-running the notebook is safe)
        row = {"exp_id": exp_id, "step": step, "model": model,
               "hyperparameters": str(params), "features": features, "split": split, "eval_set": eval_set,
               **{k: round(float(v), 4) for k, v in metrics.items()},
               "train_seconds": round(seconds, 1), "observations": observations}
        table = self.read()
        table = table[table["exp_id"] != exp_id]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True).sort_values("exp_id")
        table[TABLE_COLUMNS].to_csv(self.csv_path, index=False)

        # 2) TensorBoard: hyperparameters + final metrics of this run (shows in the HParams tab)
        run_name = f"{exp_id}_{model}".replace(" ", "_").replace("/", "-")
        hparams = {"model": model, "features": features}
        hparams.update({k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()})
        with tf.summary.create_file_writer(str(self._run_dir(run_name))).as_default():
            hp.hparams(hparams)
            for name, value in metrics.items():
                tf.summary.scalar(name, float(value), step=1)
            tf.summary.scalar("train_seconds", seconds, step=1)

        # 3) text log
        self.logger.info("%s | %s | acc %.3f | precision %.3f | recall %.3f | F1 %.3f | ROC-AUC %.3f | PR-AUC %.3f | %.1fs",
                         exp_id, model, metrics["accuracy"], metrics["precision"], metrics["recall"],
                         metrics["f1"], metrics["roc_auc"], metrics["pr_auc"], seconds)

    def log_curve(self, run_name, curves, steps):
        """
        Write curves to TensorBoard so they can be viewed as line charts.
        curves: {"tag name": list of values}, steps: the x values (for example the boosting round).
        """
        with tf.summary.create_file_writer(str(self._run_dir(run_name))).as_default():
            for tag, values in curves.items():
                for step, value in zip(steps, values):
                    tf.summary.scalar(tag, float(value), step=int(step))
        self.logger.info("curve logged to TensorBoard run '%s': %s", run_name, ", ".join(curves))

    def read(self):
        """Return the current experiment table (empty if there is none yet)."""
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path, keep_default_na=False)
        return pd.DataFrame(columns=TABLE_COLUMNS)

    def set_observations(self, notes):
        """Write the 'observations' text for experiments, e.g. {"T01": "text"}."""
        table = self.read()
        for exp_id, text in notes.items():
            table.loc[table["exp_id"] == exp_id, "observations"] = text
        table[TABLE_COLUMNS].to_csv(self.csv_path, index=False)
