"""Shared by the MAMMAL benchmark methods: paths, the model, batching, scoring and run bookkeeping.

The scoring itself lives in ../../serving/inference_mammal.py (the code behind the API), so a benchmark score
and an API score for the same molecule and model come from the same place. This module re-exports what the
methods use and adds the training-side helpers.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AIRCHECK_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = AIRCHECK_DIR.parent
sys.path.insert(0, str(AIRCHECK_DIR / "serving"))  # the serving modules import each other by plain name
import inference_mammal  # noqa: E402

# The prompt, the tokenizer, the batching and the scoring: one copy, in serving/
from inference_mammal import (  # noqa: E402,F401
    ANSWER_POSITION, BASE_MODEL, ENCODER_INPUTS_ATTENTION_MASK, ENCODER_INPUTS_TOKENS, IGNORE_INDEX,
    MAX_PROMPT_TOKENS, MAX_SMILES_TOKENS, SCORES, TOKEN_WEIGHTS_KEY, answer_scores, encoder_prompt,
    make_collate, make_loader, model_source, prepare_sample, tokenizer,
)
from inference_mammal import MoleculeDataset  # noqa: E402,F401
from inference_mammal import clean_smiles as valid_smiles  # noqa: E402,F401
from inference_mammal import load_mammal as load_model  # noqa: E402,F401

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = min(24, os.cpu_count())  # all CPUs, up to 24 (run.py --num-workers changes it)


def set_num_workers(n: int):
    """How many processes prepare the batches (0 = the main process, least memory)."""
    global NUM_WORKERS
    NUM_WORKERS = n
    inference_mammal.set_num_workers(n)


set_num_workers(NUM_WORKERS)  # serving defaults to 0 workers; training wants more


def score_with(model, smiles, batch_size=16, source=None, log=None) -> np.ndarray:
    """Scores for these molecules (NaN for a SMILES RDKit cannot read), with the serving code.

    `model` is a plain MAMMAL model, as the methods hold it while training; `source` is the folder its
    tokenizer comes from (None = the pretrained one).
    """
    loaded = inference_mammal.LoadedModel(models=[model.to(DEVICE).eval()], path=None, device=DEVICE,
                                          source=source or BASE_MODEL)
    if log:
        log(f"  scoring {len(smiles):,} molecules")
    return inference_mammal.score_smiles(loaded, smiles, batch_size)


class Log:
    """Print a line and append it to progress.log in the run folder (follow it with `tail -f`)."""

    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        with open(self.path, "a") as handle:
            handle.write(line + "\n")


def positive_weight(labels, balance: bool) -> float:
    """How much more a label-1 sample counts in the loss: (number of 0s) / (number of 1s), or 1.0."""
    if not balance:
        return 1.0
    labels = np.asarray(labels).astype(int)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    assert n_pos and n_neg, "training data needs both labels to balance the classes"
    return round(n_neg / n_pos, 4)


def predictions_frame(split: str, rows, smiles, labels, scores) -> pd.DataFrame:
    """Rows of predictions.csv, in the format aircheck-benchmark-core reads."""
    return pd.DataFrame({"split": split, "row": np.asarray(rows), "SMILES": np.asarray(smiles),
                         "LABEL": np.asarray(labels).astype(int), "score": np.asarray(scores, dtype=float)})


def fork_commit() -> str:
    """This repo's commit, with "-dirty" if tracked files have uncommitted changes."""
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({os.cpu_count()} cores)"
