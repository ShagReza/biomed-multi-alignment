"""Score molecules with a fine-tuned MAMMAL model.

This is the one place MAMMAL scoring is written down: the benchmark imports it too
(../benchmark/methods/common.py), so an API score and a benchmark score for the same molecule and model come
from the same code. Moved here from the AIRCHECK notebooks (`encoder_prompt`, `prepare_sample`,
`MoleculeDataset`, `make_collate`, `score_molecules`).

    from inference_mammal import load_model, predict

    model = load_model("path/to/<run>/models/final")                  # the final model (default)
    model = load_model("path/to/<run>/models", models="folds")        # the CV fold models, averaged
    results = predict(model, ["CCO", "c1ccccc1", "not_a_smiles"])
    # [{"smiles": "CCO", "score": 0.41, "valid": True}, ..., {"smiles": "not_a_smiles", "score": None, "valid": False}]

`score` is the probability the model gives to `<1>` against `<0>` at the answer position, between 0 and 1.
"""

import os
import warnings
from dataclasses import dataclass
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
from rdkit import Chem, RDLogger  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

RDLogger.DisableLog("rdApp.*")

BASE_MODEL = "ibm/biomed.omics.bl.sm.ma-ted-458m"  # Hugging Face model ID
MAX_SMILES_TOKENS = 300   # tokens kept of the SMILES itself
MAX_PROMPT_TOKENS = 320   # tokens kept of the whole prompt
MAX_ANSWER_TOKENS = 4
ANSWER_POSITION = 1       # the decoder writes <SENTINEL_ID_0> first, then the answer
IGNORE_INDEX = -100
TOKEN_WEIGHTS_KEY = "data.token_weights"
NUM_WORKERS = 0  # data-loading processes: 0 suits an API; the benchmark raises it with set_num_workers()

_tokenizers = {}


def set_num_workers(n: int):
    """How many processes prepare the batches (0 = the main process, least memory)."""
    global NUM_WORKERS
    NUM_WORKERS = n


def tokenizer(source=None):
    """(tokenizer, <0> id, <1> id, <PAD> id) for a model; loaded once per source."""
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
    """A DataLoader over molecules, ready for MAMMAL."""
    op, _, _, pad = tokenizer(source)
    dataset = MoleculeDataset(frame, op, pad, pos_weight=pos_weight, with_label=with_label)
    # persistent workers would stay alive for every fold at once during cross-validation
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=make_collate(pad),
                      num_workers=NUM_WORKERS)


def clean_smiles(smiles: list):
    """Strip spaces and check each SMILES with RDKit; returns the cleaned SMILES and a valid flag for each."""
    cleaned = [s.strip() if isinstance(s, str) else "" for s in smiles]
    return cleaned, [bool(s) and Chem.MolFromSmiles(s) is not None for s in cleaned]


def answer_scores(probs, pos_token_id: int, neg_token_id: int) -> np.ndarray:
    """The probability of <1> against <0>, from the scores of the answer position."""
    pos, neg = probs[:, pos_token_id], probs[:, neg_token_id]
    return (pos / (pos + neg + 1e-8)).detach().float().cpu().numpy()


def model_source(path=None) -> str:
    """How a model is written down in a config or run record."""
    return BASE_MODEL if path is None else str(Path(path).expanduser().resolve())


def _model_dir(path) -> Path:
    """A saved MAMMAL model folder, given it or a folder holding it."""
    path = Path(path).expanduser()
    if not (path / "config.json").exists() and (path / "best_model").is_dir():
        path = path / "best_model"  # a fold or final folder was given
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"No MAMMAL model in {path}")
    return path


def load_mammal(path=None) -> Mammal:
    """One MAMMAL model: the pretrained one (path None), or a saved model folder."""
    return Mammal.from_pretrained(BASE_MODEL if path is None else str(_model_dir(path)))


@dataclass
class LoadedModel:
    models: list   # one model, or one per CV fold (their scores are averaged)
    path: Path
    device: str
    source: str    # the tokenizer source: a model folder, or the pretrained model ID


def load_model(path, device=None, models="full") -> LoadedModel:
    """Load a fine-tuned MAMMAL model.

    models: "full" (default): `path` is a saved model folder, or a folder holding `best_model/`
            (a benchmark run's `models/final`).
            "folds": `path` holds `fold_<i>/best_model/` (a benchmark run's `models/`); every fold model is
            loaded and their scores averaged.
    device: "cpu" or "cuda"; None uses a GPU if there is one.
    """
    path = Path(path).expanduser()
    if models == "full":
        folders = [_model_dir(path)]
    elif models == "folds":
        folders = sorted(path.glob("fold_*/best_model"), key=lambda p: int(p.parent.name.split("_")[1]))
        if not folders:
            raise FileNotFoundError(f"No fold_*/best_model in {path} (models='folds' needs a run's models/ folder)")
    else:
        raise ValueError(f"models must be 'full' or 'folds', not {models!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    loaded = [Mammal.from_pretrained(str(f)).to(device).eval() for f in folders]
    # A saved model keeps its tokenizer in <model>/tokenizer; for a local path fuse reads that
    # folder as given (it only appends "tokenizer" when downloading from the hub)
    source = str(folders[0] / "tokenizer") if (folders[0] / "tokenizer").is_dir() else BASE_MODEL
    return LoadedModel(models=loaded, path=path, device=device, source=source)


def _scores(model: LoadedModel, smiles: list, batch_size: int) -> np.ndarray:
    op, neg_id, pos_id, pad = tokenizer(model.source)
    loader = make_loader(pd.DataFrame({"smiles": smiles}), model.source, batch_size, with_label=False)
    per_model = []
    with torch.no_grad():
        for one in model.models:
            parts = []
            for batch in loader:
                for key in (ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK):
                    batch[key] = batch[key].to(model.device)
                result = one.generate(batch, output_scores=True, return_dict_in_generate=True, max_new_tokens=5)
                parts.append(answer_scores(result[SCORES][:, ANSWER_POSITION, :], pos_id, neg_id))
            per_model.append(np.concatenate(parts))
    scores = np.mean(per_model, axis=0)  # one model, or the average over folds
    if len(scores) != len(smiles):
        raise RuntimeError(f"Got {len(scores)} scores for {len(smiles)} molecules")
    return scores


def score_smiles(model: LoadedModel, smiles, batch_size: int = 16) -> np.ndarray:
    """Scores in the order given, NaN for a SMILES RDKit cannot read."""
    cleaned, valid = clean_smiles(list(smiles))
    scores = np.full(len(cleaned), np.nan)
    kept = [s for s, ok in zip(cleaned, valid) if ok]
    if kept:
        scores[np.array(valid)] = _scores(model, kept, batch_size)
    return scores


def predict(model: LoadedModel, smiles: list, batch_size: int = 16) -> list:
    """Score a list of SMILES. Returns one result per input, in the same order:
    {"smiles": ..., "score": float between 0 and 1, "valid": True}, or score None and valid False for a SMILES
    that RDKit cannot read."""
    scores = score_smiles(model, smiles, batch_size)
    return [{"smiles": original, "score": None if np.isnan(score) else float(score), "valid": not np.isnan(score)}
            for original, score in zip(smiles, scores)]
