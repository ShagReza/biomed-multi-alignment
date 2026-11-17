"""Build a small mock multi-target dataset for MAMMAL_multitarget_finetuning.ipynb.

The multi-target notebooks need rows of (protein sequence, SMILES, label). We have no real multi-target
screen yet, so this writes a synthetic one, only to check that the notebooks run end to end.

- **Molecules** are real: SMILES taken from the MMELON fork's sample data (a WDR91 ASMS subset).
- **Proteins are not real.** Three random amino-acid sequences are generated from a fixed seed, so the file
  is reproducible but the sequences mean nothing biologically.
- **Labels are synthetic.** Target A keeps the molecule's real label; targets B and C use a deterministic
  rule over the SMILES, so each target has a different answer for the same molecule and the model has
  something target-specific to learn.

Nothing here should be used to judge how well the model works: it only exercises the code.

Usage, from the repository root:
    python aircheck/tools/make_mock_multitarget.py
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
OUT_DIR = REPO_ROOT / "aircheck" / "sample_data"
MMELON_SAMPLE = REPO_ROOT.parent / "biomed-multi-view" / "aircheck" / "sample_data"

SEED = 42
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"   # the 20 standard amino acids, one letter each
TARGETS = {"MOCK_TARGET_A": 180, "MOCK_TARGET_B": 140, "MOCK_TARGET_C": 220}  # name -> length
N_PER_TARGET = {"train": 300, "test": 100}
POSITIVE_RATE = 0.05                   # roughly, for the synthetic targets


def mock_sequences(rng) -> dict:
    """A random amino-acid sequence per target. Not real proteins."""
    return {name: "".join(rng.choice(list(AMINO_ACIDS), size=length))
            for name, length in TARGETS.items()}


def synthetic_label(target: str, smiles: str) -> int:
    """A deterministic 0/1 that depends on both the target and the molecule."""
    digest = hashlib.sha256(f"{target}|{smiles}".encode()).digest()
    return int(int.from_bytes(digest[:4], "big") / 2**32 < POSITIVE_RATE)


def build(split: str, source: Path, sequences: dict, rng) -> pd.DataFrame:
    molecules = pd.read_parquet(source, columns=["SMILES", "LABEL"]).dropna()
    rows = []
    for target, sequence in sequences.items():
        take = molecules.sample(n=min(N_PER_TARGET[split], len(molecules)), random_state=rng.integers(1 << 30))
        for smiles, real_label in zip(take["SMILES"], take["LABEL"]):
            label = int(real_label) if target == "MOCK_TARGET_A" else synthetic_label(target, smiles)
            rows.append({"TARGET": target, "PROTEIN": sequence, "SMILES": smiles, "LABEL": label})
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def main() -> None:
    assert MMELON_SAMPLE.exists(), (
        f"MMELON sample data not found at {MMELON_SAMPLE}; clone the biomed-multi-view fork next to this one")
    rng = np.random.default_rng(SEED)
    sequences = mock_sequences(rng)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, source in [("train", MMELON_SAMPLE / "train.parquet"), ("test", MMELON_SAMPLE / "test.parquet")]:
        df = build(split, source, sequences, rng)
        out = OUT_DIR / f"mock_multitarget_{split}.csv"
        df.to_csv(out, index=False)
        counts = df.groupby("TARGET")["LABEL"].agg(["size", "sum"]).rename(columns={"size": "rows", "sum": "positives"})
        print(f"{out.name}: {len(df)} rows")
        print(counts.to_string(), "\n")


if __name__ == "__main__":
    main()
