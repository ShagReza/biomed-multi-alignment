"""LightGBM on MAMMAL embeddings.

The same steps as ../../notebooks/MAMMAL_embedding_extraction.ipynb, MAMMAL_embedding_lgbm_cv.ipynb and
MAMMAL_embedding_lgbm_inference.ipynb:
1. Embed the training and test molecules with MAMMAL: the pretrained model by default, or a fine-tuned one
   (`encoder`). The embeddings are saved as parquet next to the runs (see embeddings.py) and reused.
2. One LightGBM model per fold (split foldN = that fold's test rows), plus one on all training rows.
3. The test file scored by the fold models averaged (test_ensemble) and by the model on all rows (test).

The LightGBM part itself is `aircheck_benchmark_core.lgbm`, shared with every other model in the benchmark.

With a fine-tuned encoder that has seen the training labels, the CV numbers are over-optimistic; only the
test numbers are valid then. The pretrained model (the default) gives a clean CV.
"""

import gc
import json
from pathlib import Path

import pandas as pd

from aircheck_benchmark_core import lgbm

from . import common, embeddings

DEFAULTS = dict(view=embeddings.VIEW, encoder=None, lgbm_params="suggested", balance_classes=True, seed=42,
                batch_size=32, embeddings_dir=None)
LGBM_PARAMS = lgbm.PARAMS  # so the notebooks and run.py can show them


def encoder_label(encoder) -> str:
    """Folder name for a model's embeddings: "pretrained", or "finetuned-<run name>"."""
    if encoder is None:
        return "pretrained"
    path = Path(encoder).expanduser().resolve()
    skip = {"best_model", "models", "final"}
    run = next((p.name for p in path.parents if p.name not in skip and not p.name.startswith("fold_")), "encoder")
    return f"finetuned-{run}"


def run(data: pd.DataFrame, folds: list, test: pd.DataFrame, s: dict, out_dir: Path, log) -> tuple:
    """Returns (predictions, report extras, settings used). `data` and `test` have columns smiles, label."""
    s = {**DEFAULTS, **{k: v for k, v in s.items() if v is not None}}
    assert s["view"] in embeddings.VIEWS, f"view must be one of {embeddings.VIEWS} for embed_lgbm"
    assert s["lgbm_params"] in LGBM_PARAMS, f"lgbm_params must be one of {list(LGBM_PARAMS)}"

    label = encoder_label(s["encoder"])
    root = Path(s["embeddings_dir"]).expanduser() if s["embeddings_dir"] else out_dir.parent / "embeddings" / label
    source = common.model_source(s["encoder"])
    log(f"Embeddings ({label}, model: {source}) in {root}")
    model = common.load_model(s["encoder"])
    for name, frame in (("train", data), ("test", test)):
        embeddings.extract(model, frame["smiles"], root / name, source, batch_size=s["batch_size"], log=log)
    del model
    gc.collect()

    X_train, valid_train = embeddings.load(root / "train", s["view"], data["smiles"])
    X_test, valid_test = embeddings.load(root / "test", s["view"], test["smiles"])
    for name, valid in (("training", valid_train), ("test", valid_test)):
        if not valid.all():
            log(f"{(~valid).sum():,} {name} molecules have invalid SMILES: no embedding, empty score")

    log(f"Training LightGBM ({s['lgbm_params']} parameters) on {len(folds)} folds and on all rows, "
        f"{X_train.shape[1]} dimensions")
    scores = lgbm.train_and_score(X_train, valid_train, data["label"], folds, X_test, valid_test,
                                  out_dir / "models", choice=s["lgbm_params"],
                                  balance_classes=s["balance_classes"], seed=s["seed"])

    parts = [common.predictions_frame(f"fold{i}", test_idx, data["smiles"].iloc[test_idx],
                                      data["label"].iloc[test_idx], scores[f"fold{i}"])
             for i, (_, test_idx) in enumerate(folds)]
    parts += [common.predictions_frame(split, test.index, test["smiles"], test["label"], scores[split])
              for split in ("test_ensemble", "test")]
    (out_dir / "config.json").write_text(json.dumps(
        {"view": s["view"], "embedding_model": source, "lgbm_params_choice": s["lgbm_params"],
         "lgbm_params": LGBM_PARAMS[s["lgbm_params"]], "embeddings_dir": str(root)}, indent=2) + "\n")
    s = {k: v for k, v in s.items() if k != "embeddings_dir"}
    return pd.concat(parts, ignore_index=True), {}, {**s, "paths": {"embeddings": str(root)}}
