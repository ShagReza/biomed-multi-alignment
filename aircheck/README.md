# AIRCHECK notebooks for MAMMAL

This folder is the AIRCHECK team's addition to IBM's [biomed-multi-alignment](../README.md) (MAMMAL) repository.
Everything outside `aircheck/` is the original repository, unchanged.

The goal is the same as for our MMELON work (the `aircheck/` folder of our biomed-multi-view fork): fine-tune and
evaluate a foundation model for **hit identification**, binary classification of molecules, ranked and scored with
**hit@K** (how many of the top K predicted molecules are actives), so the two models can be compared on the same data.

Status: setup and planning. The notebooks are not written yet.

## MAMMAL in short

- **Model:** `ibm/biomed.omics.bl.sm.ma-ted-458m`, a 458M-parameter encoder-decoder (T5-like) trained on proteins,
  small molecules and gene expression.
- **Input:** one text prompt. Special tokens mark what follows, e.g. `<MOLECULAR_ENTITY_SMALL_MOLECULE>` before a
  SMILES or `<MOLECULAR_ENTITY_GENERAL_PROTEIN>` before an amino-acid sequence. A prompt can hold a molecule, a
  protein, or both; the task decides.
- **Output:** the model writes an answer token. For classification it writes `<0>` or `<1>`, and the score is the
  probability of `<1>`.
- **Fine-tuning** teaches it one prompt format. Inference must use the same format.

### Two ways to use it for hit identification

| | Molecule only | Protein + molecule |
|---|---|---|
| Prompt | SMILES | target's amino-acid sequence + SMILES |
| Models | one per target | one for many targets |
| Example in this repo | `mammal/examples/molnet` | `mammal/examples/dti_bindingdb_kd` (binding affinity) |
| Use | direct comparison with MMELON | sharing what is learned across targets; needs many targets to generalize to new proteins |

We start with **molecule only**, the same setup as MMELON.

### How it differs from MMELON

| | MMELON | MAMMAL |
|---|---|---|
| Molecule input | SMILES text, 2D image and graph, fused | SMILES text only |
| Size | ~84M parameters | 458M parameters |
| Prediction | classification head | generated `<0>` / `<1>` token |
| Library | `bmfm_sm` (fuse-med-ml 0.3.0, torch 2.1, transformers 4) | `mammal` (fuse-med-ml 0.4.1, torch 2.14, transformers 5) |

The two need **separate Python environments**; see [INSTALL.md](INSTALL.md).

## Folder layout (planned)

```
aircheck/
├── README.md                    # this file
├── INSTALL.md                   # environment setup
├── requirements-aircheck.txt    # extra packages on top of the repository's own
├── notebooks/                   # fine-tuning, CV, inference (to come)
└── output/                      # created by the notebooks; not tracked by git
```

## Data

The notebooks will read the same files as the MMELON notebooks (`.csv` or `.parquet`, with a SMILES and a label
column), so both models train and test on identical data. The sample data is the MMELON fork's
`aircheck/sample_data/` (WDR91 ASMS subset: 10,221 train molecules with 221 actives, 1,031 test with 31 actives).

## Plan

Following the MMELON notebooks, as far as it makes sense for MAMMAL:

1. **Fine-tuning** on one train/validation split, best epoch chosen by the same metrics as MMELON (hit@K, AUROC).
2. **Inference** on a test file: `predictions.csv` (every molecule with its score and rank) and metrics.
3. **Cross-validation** on the same cluster folds as MMELON, with early stopping and class balancing.
4. **Fold ensemble** inference.
5. Later, optionally: embeddings + LightGBM, the report, a whole-recipe runner, and the protein + molecule format.

Starting points: the earlier notebooks `MAMMAL_finetune.ipynb` and `MAMMAL_inference.ipynb` (from IBM's tutorial),
which use one fixed epoch, no CV, and hard-coded paths.

## Notes

- **Speed:** MAMMAL is about 5 times larger than MMELON, so expect slower training and scoring; a GPU is needed
  for real datasets.
- **GPU:** the environment's PyTorch is built for CUDA 13, which supports T4, L4, A100 and H100 but not V100.
