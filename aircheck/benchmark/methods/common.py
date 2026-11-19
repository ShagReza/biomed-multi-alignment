"""Shared by the MAMMAL benchmark methods: paths, the tokenizer, the prompt, batching and scoring.

Moved from the AIRCHECK notebooks of this fork (`encoder_prompt`, `prepare_sample`, `MoleculeDataset`,
`make_collate`, `score_molecules`), which had a copy in every notebook. The prompt and the score are exactly
theirs: the score of a molecule is the probability the model gives to `<1>` against `<0>` at the answer
position.
"""

import os
import subprocess
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp  # noqa: E402
from fuse.data.utils.collates import CollateDefault  # noqa: E402
from mammal.keys import (  # noqa: E402
    DECODER_INPUTS_ATTENTION_MASK, DECODER_INPUTS_STR, DECODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_STR, ENCODER_INPUTS_TOKENS, LABELS_ATTENTION_MASK, LABELS_STR, LABELS_TOKENS, SCORES,
)
from mammal.model import Mammal  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

AIRCHECK_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = AIRCHECK_DIR.parent
BASE_MODEL = "ibm/biomed.omics.bl.sm.ma-ted-458m"  # Hugging Face model ID
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = min(24, os.cpu_count())  # data-loading processes (run.py --num-workers changes it)

MAX_SMILES_TOKENS = 300   # tokens kept of the SMILES itself
MAX_PROMPT_TOKENS = 320   # tokens kept of the whole prompt
MAX_ANSWER_TOKENS = 4
ANSWER_POSITION = 1       # the decoder writes <SENTINEL_ID_0> first, then the answer
IGNORE_INDEX = -100
TOKEN_WEIGHTS_KEY = "data.token_weights"

_tokenizers = {}


def tokenizer(source=None):
    """The MAMMAL tokenizer of a model (the base model by default); loaded once per source."""
    source = str(source or BASE_MODEL)
    if source not in _tokenizers:
        op = ModularTokenizerOp.from_pretrained(source)
        _tokenizers[source] = (op, op.get_token_id("<0>"), op.get_token_id("<1>"), op.get_token_id("<PAD>"))
    return _tokenizers[source]


def encoder_prompt(smiles: str) -> str:
    """The text MAMMAL reads for one molecule."""
    return (
        f"<@TOKENIZER-TYPE=SMILES><SENTINEL_ID_0>"
        f"<MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE>"
        f"<@TOKENIZER-TYPE=SMILES@MAX-LEN={MAX_SMILES_TOKENS}>"
        f"<SEQUENCE_NATURAL_START>{smiles}<SEQUENCE_NATURAL_END><EOS>"
    )


def prepare_sample(sample_dict: dict, tokenizer_op, pad_token_id: int, pos_weight: float = 1.0,
                   with_label: bool = True) -> dict:
    """Tokenize one sample: the prompt, and (for training) the answer and its loss weight."""
    sample_dict[ENCODER_INPUTS_STR] = encoder_prompt(sample_dict["data.smiles"])
    tokenizer_op(sample_dict=sample_dict, key_in=ENCODER_INPUTS_STR, key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
                 key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK, max_seq_len=MAX_PROMPT_TOKENS)
    for key in (ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK):
        sample_dict[key] = torch.tensor(sample_dict[key])
    if not with_label:
        return sample_dict

    label = int(sample_dict["data.label"])
    # What the model should write
    sample_dict[LABELS_STR] = f"<@TOKENIZER-TYPE=SMILES><SENTINEL_ID_0><{label}><EOS>"
    tokenizer_op(sample_dict=sample_dict, key_in=LABELS_STR, key_out_tokens_ids=LABELS_TOKENS,
                 key_out_attention_mask=LABELS_ATTENTION_MASK, max_seq_len=MAX_ANSWER_TOKENS)
    for key in (LABELS_TOKENS, LABELS_ATTENTION_MASK):
        sample_dict[key] = torch.tensor(sample_dict[key])
    sample_dict[LABELS_TOKENS][sample_dict[LABELS_TOKENS] == pad_token_id] = IGNORE_INDEX  # padding: no loss

    # What the decoder is fed during training (teacher forcing)
    sample_dict[DECODER_INPUTS_STR] = f"<@TOKENIZER-TYPE=SMILES><DECODER_START><SENTINEL_ID_0><{label}><EOS>"
    tokenizer_op(sample_dict=sample_dict, key_in=DECODER_INPUTS_STR, key_out_tokens_ids=DECODER_INPUTS_TOKENS,
                 key_out_attention_mask=DECODER_INPUTS_ATTENTION_MASK, max_seq_len=MAX_ANSWER_TOKENS)
    for key in (DECODER_INPUTS_TOKENS, DECODER_INPUTS_ATTENTION_MASK):
        sample_dict[key] = torch.tensor(sample_dict[key])

    # Class balancing: weight every counted position of this sample, 0 where the loss is ignored
    weight = pos_weight if label == 1 else 1.0
    sample_dict[TOKEN_WEIGHTS_KEY] = weight * (sample_dict[LABELS_TOKENS] != IGNORE_INDEX).float()
    return sample_dict


class MoleculeDataset(Dataset):
    """One row per molecule, from a frame with a 'smiles' and (for training) a 'label' column."""

    def __init__(self, frame, tokenizer_op, pad_token_id, pos_weight: float = 1.0, with_label: bool = True):
        self.smiles = list(frame["smiles"])
        self.labels = list(frame["label"]) if with_label else [0] * len(self.smiles)
        self.tokenizer_op, self.pad_token_id = tokenizer_op, pad_token_id
        self.pos_weight, self.with_label = pos_weight, with_label

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        sample = {"data.sample_id": idx, "data.smiles": self.smiles[idx], "data.label": self.labels[idx]}
        return prepare_sample(sample, self.tokenizer_op, self.pad_token_id, self.pos_weight, self.with_label)


def make_collate(pad_token_id):
    """Trim the padding each batch does not need (10-20% faster)."""
    return CollateDefault(special_handlers_keys={
        ENCODER_INPUTS_TOKENS: partial(CollateDefault.crop_padding, pad_token_id=pad_token_id),
        ENCODER_INPUTS_ATTENTION_MASK: partial(CollateDefault.crop_padding, pad_token_id=False),
    })


def make_loader(frame, source=None, batch_size=16, shuffle=False, pos_weight=1.0, with_label=True) -> DataLoader:
    op, _, _, pad = tokenizer(source)
    dataset = MoleculeDataset(frame, op, pad, pos_weight=pos_weight, with_label=with_label)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=make_collate(pad),
                      num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)


def answer_scores(probs, pos_token_id, neg_token_id) -> np.ndarray:
    """The probability of <1> against <0>, from the scores of the answer position."""
    pos, neg = probs[:, pos_token_id], probs[:, neg_token_id]
    return (pos / (pos + neg + 1e-8)).detach().float().cpu().numpy()


def load_model(path=None) -> Mammal:
    """A MAMMAL model: the pretrained base model (path None), a saved model folder, or a run's best_model."""
    if path is None:
        return Mammal.from_pretrained(BASE_MODEL)
    path = Path(path).expanduser()
    if not (path / "config.json").exists() and (path / "best_model").is_dir():
        path = path / "best_model"  # a fold or run folder was given
    assert (path / "config.json").exists(), f"No MAMMAL model in {path}"
    return Mammal.from_pretrained(str(path))


def model_source(path=None) -> str:
    """How a model is written down in a config or run record."""
    return BASE_MODEL if path is None else str(Path(path).expanduser().resolve())


def valid_smiles(smiles) -> tuple:
    """Stripped SMILES and a valid flag for each (RDKit can read it)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    cleaned = [s.strip() if isinstance(s, str) else "" for s in smiles]
    return cleaned, [bool(s) and Chem.MolFromSmiles(s) is not None for s in cleaned]


def score_with(model, smiles, batch_size=16, source=None, log=None) -> np.ndarray:
    """Scores for these molecules (NaN for a SMILES RDKit cannot read), as the inference notebooks compute them."""
    op, neg_id, pos_id, pad = tokenizer(source)
    cleaned, valid = valid_smiles(smiles)
    scores = np.full(len(cleaned), np.nan)
    kept = [s for s, ok in zip(cleaned, valid) if ok]
    if not kept:
        return scores
    model = model.to(DEVICE).eval()
    loader = make_loader(pd.DataFrame({"smiles": kept}), source, batch_size, with_label=False)
    out = []
    with torch.no_grad():
        for done, batch in enumerate(loader, start=1):
            for key in (ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK):
                batch[key] = batch[key].to(DEVICE)
            result = model.generate(batch, output_scores=True, return_dict_in_generate=True, max_new_tokens=5)
            out.append(answer_scores(result[SCORES][:, ANSWER_POSITION, :], pos_id, neg_id))
            if log and done % 20 == 0:
                log(f"  scored {sum(len(part) for part in out):,} of {len(kept):,} molecules")
    scores[np.array(valid)] = np.concatenate(out)
    return scores


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
