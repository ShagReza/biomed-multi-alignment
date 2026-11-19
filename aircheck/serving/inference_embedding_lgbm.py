"""Score molecules with LightGBM on MAMMAL embeddings (from the benchmark's embed_lgbm method).

Each SMILES is turned into a MAMMAL embedding the same way as the benchmark does, then scored with the run's
final LightGBM model (models/full.txt, trained on all training rows) or, with models="folds", the average of
its fold models.

    from inference_embedding_lgbm import load_model, predict

    model = load_model("path/to/<experiment>/lgbm_pretrained")        # load once
    results = predict(model, ["CCO", "not_a_smiles"])
    # [{"smiles": "CCO", "score": 0.12, "valid": True}, {"smiles": "not_a_smiles", "score": None, "valid": False}]

encoder_path is the MAMMAL model the embeddings came from: a saved model folder (a run's `models/final`), or
None for the pretrained model. None takes it from the run's config.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from inference_mammal import (
    ENCODER_INPUTS_ATTENTION_MASK, ENCODER_INPUTS_TOKENS, clean_smiles, load_mammal, make_loader,
)

VIEW = "mammal"  # MAMMAL gives one embedding per molecule


@dataclass
class LoadedLgbmModel:
    encoder: torch.nn.Module  # the MAMMAL model that makes the embeddings
    view: str                 # always "mammal"; kept so the API can report it like MMELON's
    boosters: list            # the final model, or one LightGBM model per fold (their scores are averaged)
    path: Path
    device: str
    source: str               # the tokenizer source of the encoder


def embed_batch(model, batch) -> np.ndarray:
    """One vector per molecule: the encoder's token vectors, averaged over the real tokens."""
    mask = batch[ENCODER_INPUTS_ATTENTION_MASK]
    hidden = model.t5_model.encoder(
        inputs_embeds=model._calculate_inputs_embeddings(batch), attention_mask=mask,
    )["last_hidden_state"]                         # (batch, tokens, 768)
    weights = mask.unsqueeze(-1).to(hidden.dtype)  # 0 for padding, 1 for real tokens
    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
    return pooled.float().cpu().numpy()


def load_model(lgbm_run_dir, encoder_path=None, device=None, models="full") -> LoadedLgbmModel:
    """Load a LightGBM run and the MAMMAL model its embeddings came from.

    models: "full" (default) uses models/full.txt; "folds" averages the fold models."""
    run = Path(lgbm_run_dir).expanduser()
    config_path = run / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Not a LightGBM run folder (no config.json): {run}")
    config = json.loads(config_path.read_text())
    if models == "full":
        paths = [run / "models" / "full.txt"]
        if not paths[0].exists():
            raise FileNotFoundError(f"No final model {paths[0]} (use models='folds' for the fold models)")
    elif models == "folds":
        paths = sorted((run / "models").glob("fold_*.txt"), key=lambda p: int(p.stem.split("_")[1]))
        if not paths:
            raise FileNotFoundError(f"No models/fold_*.txt in {run}")
    else:
        raise ValueError(f"models must be 'full' or 'folds', not {models!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if encoder_path is None:  # from the run's config: a model folder, or the pretrained model's ID
        recorded = config["embedding_model"]
        encoder_path = None if recorded.startswith(("base:", "ibm/")) else recorded
    encoder = load_mammal(encoder_path).to(device).eval()
    source = str(Path(encoder_path).expanduser()) if encoder_path else None

    boosters = [lgb.Booster(model_file=str(p)) for p in paths]
    return LoadedLgbmModel(encoder=encoder, view=config.get("view", VIEW), boosters=boosters, path=run,
                           device=device, source=source)


def _embeddings(model: LoadedLgbmModel, smiles: list, batch_size: int) -> np.ndarray:
    loader = make_loader(pd.DataFrame({"smiles": smiles}), model.source, batch_size, with_label=False)
    parts = []
    with torch.no_grad():
        for batch in loader:
            for key in (ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK):
                batch[key] = batch[key].to(model.device)
            parts.append(embed_batch(model.encoder, batch))
    X = np.concatenate(parts)
    if len(X) != len(smiles):
        raise RuntimeError(f"Got {len(X)} embeddings for {len(smiles)} molecules")
    return X


def predict(model: LoadedLgbmModel, smiles: list, batch_size: int = 16) -> list:
    """Score a list of SMILES. Returns one result per input, in the same order:
    {"smiles": ..., "score": float between 0 and 1, "valid": True}, or score None and valid False for a SMILES
    that RDKit cannot read."""
    cleaned, valid = clean_smiles(smiles)
    kept = [s for s, ok in zip(cleaned, valid) if ok]
    scores = iter([])
    if kept:
        X = _embeddings(model, kept, batch_size)
        scores = iter(np.mean([b.predict(X) for b in model.boosters], axis=0).tolist())  # one model, or the fold average
    return [{"smiles": original, "score": next(scores) if ok else None, "valid": ok}
            for original, ok in zip(smiles, valid)]
