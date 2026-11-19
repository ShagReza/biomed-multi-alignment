"""MAMMAL embeddings, saved as parquet and shared between runs.

The same steps and file layout as ../../archive-to-be-deleted/notebooks/MAMMAL_embedding_extraction.ipynb: one vector per molecule,
the encoder's token vectors averaged over the real tokens, written as parquet parts of `CHUNK_SIZE` molecules:

    <dir>/config.json                     model, number of molecules, view
    <dir>/<view>/part-00000.parquet       row_index (row in the input), smiles, embedding
    <dir>/invalid_smiles.csv              rows RDKit cannot read (skipped, no embedding)

Finished parts are reused, so an interrupted extraction resumes and a later run costs nothing. The
embeddings of a fine-tuned model go in their own folder, never mixed with the pretrained ones.
"""

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import common  # puts ../../serving on sys.path
from .common import ENCODER_INPUTS_ATTENTION_MASK, ENCODER_INPUTS_TOKENS

import inference_embedding_lgbm  # noqa: E402  (from ../../serving)

CHUNK_SIZE = 1000   # molecules per part file, as in the extraction notebook
VIEW = "mammal"     # MAMMAL gives one embedding per molecule
VIEWS = (VIEW,)


def available_views(model=None) -> tuple:
    return VIEWS


def _part(out_dir: Path, view: str, chunk: int) -> Path:
    return out_dir / view / f"part-{chunk:05d}.parquet"


# The embedding itself lives in ../../serving/inference_embedding_lgbm.py, so the API and the benchmark
# make the same vectors
embed_batch = inference_embedding_lgbm.embed_batch


def extract(model, smiles, out_dir: Path, source: str, data_path=None, views=None, batch_size=32,
            log=print) -> Path:
    """Write the embeddings of `smiles` to `out_dir` (reusing finished parts). `source` names the model."""
    out_dir = Path(out_dir)
    cleaned, valid = common.valid_smiles(smiles)
    molecules = pd.DataFrame({"row_index": np.where(valid)[0], "smiles": np.array(cleaned, dtype=object)[valid]})
    config = {"encoder": source, "data_path": str(data_path) if data_path else None,
              "n_rows": len(cleaned), "n_valid": len(molecules), "views": [VIEW], "chunk_size": CHUNK_SIZE,
              "max_smiles_tokens": common.MAX_SMILES_TOKENS, "max_prompt_tokens": common.MAX_PROMPT_TOKENS}
    config_path = out_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        same = {k: previous.get(k) for k in ("encoder", "n_rows", "n_valid")}
        assert same == {k: config[k] for k in same}, (
            f"{out_dir} holds embeddings of other molecules or another encoder:\n{previous}\nUse another folder.")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / VIEW).mkdir(exist_ok=True)
    if not len(molecules):
        return out_dir
    if not all(valid):
        pd.DataFrame({"row_index": np.where(~np.array(valid))[0],
                      "smiles": np.array(smiles, dtype=object)[~np.array(valid)]}).to_csv(
            out_dir / "invalid_smiles.csv", index=False)

    n_chunks = (len(molecules) + CHUNK_SIZE - 1) // CHUNK_SIZE
    todo = [c for c in range(n_chunks) if not _part(out_dir, VIEW, c).exists()]
    if not todo:
        log(f"Embeddings ready in {out_dir} ({len(molecules):,} molecules)")
        return out_dir
    log(f"Embedding {len(molecules):,} molecules in {len(todo)} of {n_chunks} chunk(s)")

    model = model.to(common.DEVICE).eval()
    for done, chunk in enumerate(todo, start=1):
        rows = molecules.iloc[chunk * CHUNK_SIZE:(chunk + 1) * CHUNK_SIZE]
        loader = common.make_loader(rows, batch_size=batch_size, with_label=False)
        parts = []
        with torch.no_grad():
            for batch in loader:
                for key in (ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK):
                    batch[key] = batch[key].to(common.DEVICE)
                parts.append(embed_batch(model, batch))
        matrix = np.concatenate(parts)
        assert len(matrix) == len(rows), f"chunk {chunk}: {len(matrix)} embeddings for {len(rows)} molecules"
        frame = rows.assign(embedding=list(matrix))
        # Write to a hidden temp file first, so an interrupted write never looks like a finished part
        final = _part(out_dir, VIEW, chunk)
        tmp = final.with_name(f".{final.name}.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.rename(final)
        del loader, parts
        gc.collect()
        log(f"  chunk {done}/{len(todo)} done ({len(rows):,} molecules)")
    return out_dir


def load(out_dir: Path, view: str, smiles) -> tuple:
    """(embeddings of the valid molecules, valid flag per input row), checked against `smiles` row by row."""
    folder = Path(out_dir) / view
    assert folder.is_dir() and any(folder.glob("part-*.parquet")), f"No {view!r} embeddings in {out_dir}"
    frame = pd.read_parquet(folder).sort_values("row_index").reset_index(drop=True)
    assert frame["row_index"].is_unique, f"Duplicate row_index in {folder}"

    cleaned, valid = common.valid_smiles(smiles)
    valid = np.array(valid, dtype=bool)
    expected = np.where(valid)[0]
    assert np.array_equal(frame["row_index"].to_numpy(), expected), (
        f"{folder} holds {len(frame)} embeddings for rows that are not the {len(expected)} valid molecules here")
    mismatch = int((frame["smiles"].to_numpy() != np.array(cleaned, dtype=object)[valid]).sum())
    assert mismatch == 0, f"{mismatch} rows have different SMILES in {folder} and the input"
    return np.stack(frame["embedding"].to_numpy()).astype(np.float32), valid
