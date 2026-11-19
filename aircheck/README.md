# AIRCHECK benchmark for MAMMAL

This folder is the AIRCHECK team's addition to IBM's [biomed-multi-alignment](../README.md) (MAMMAL) repository.
Everything outside `aircheck/` is the original repository, unchanged.

The goal is the same as for our MMELON work (the `aircheck/` folder of our biomed-multi-view fork): fine-tune and
evaluate a foundation model for **hit identification**, binary classification of molecules, ranked and scored with
**hit@K** (how many of the top K predicted molecules are actives), so the two models can be compared on the same data.

Status: [benchmark/](benchmark/) holds the code that produces the benchmark results. Its LightGBM method and
LoRA fine-tuning have been checked end to end on a small CPU run; full fine-tuning needs about 15 GB of memory
and is still untested. Expect to fix small things on the first GPU run.

The earlier notebooks are in [archive-to-be-deleted/](archive-to-be-deleted/) and are no longer used.

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

## Folder layout

```
aircheck/
├── README.md                    # this file
├── INSTALL.md                   # environment setup
├── requirements-aircheck.txt    # extra packages on top of the repository's own
├── benchmark/                   # benchmark runs: run_all.sh, run.py, methods/ and thin notebooks/
├── serving/                     # prediction API and its Docker image
├── sample_output/               # mock-up of what an experiment produces (made-up numbers)
├── sample_data/                 # mock multi-target data for the protein + molecule notebooks
├── tools/                       # make_mock_multitarget.py, which builds that mock data
├── archive-to-be-deleted/       # the old notebooks; replaced by benchmark/
└── output/                      # experiments and runs; not tracked by git
```

## Data

The notebooks read the same files as the MMELON notebooks (`.csv` or `.parquet`, with a SMILES and a label
column), so both models train and test on identical data. The sample data is the MMELON fork's
`aircheck/sample_data/` (WDR91 ASMS subset: 10,221 train molecules with 221 actives, 1,031 test with 31 actives).

## How to run it

```bash
aircheck/benchmark/run_all.sh --experiment wdr91-cluster \
    --train <train file> --test <test file> --only finetune_lora,lgbm_pretrained
```
Everything goes to `aircheck/output/<experiment>/`, including `report.md`. See
[benchmark/README.md](benchmark/README.md) for the methods and options,
[benchmark/notebooks/](benchmark/notebooks/) for the same steps one at a time, and
[sample_output/](sample_output/) for a mock-up of the folder a run produces.

To serve a trained model over HTTP, see [serving/](serving/): the same API and Docker setup as the MMELON
fork, reusing the scoring code the benchmark uses.

## The archived notebooks

All in `archive-to-be-deleted/notebooks/`, kept only for reference. The three ways of using MAMMAL each have a
training and an inference notebook, and the molecule-only route also has cross-validation and a fold ensemble,
matching the MMELON set.

| Mode | File | What it does |
|---|---|---|
| Folds | `cluster_cv_folds.ipynb` | Cluster-held-out CV folds. **Copied unchanged from the MMELON fork**, so both models use the same splits. |
| 1. Molecule only | `MAMMAL_finetuning.ipynb` | One model on a train file, best epoch picked on a second file |
| | `MAMMAL_finetuning_cv.ipynb` | Cross-validation on those folds; picks the epoch to train the final model for |
| | `MAMMAL_inference.ipynb` | Scores a file with one model |
| | `MAMMAL_inference_cv.ipynb` | Scores a file with the fold models averaged |
| 2. Embeddings | `MAMMAL_embedding_extraction.ipynb` | One 768-number vector per molecule, from the encoder (pretrained or fine-tuned) |
| + LightGBM | `MAMMAL_embedding_lgbm_cv.ipynb` | CV LightGBM on those embeddings. **Copied from the MMELON fork.** |
| | `MAMMAL_embedding_lgbm_inference.ipynb` | Scores test embeddings with the LightGBM models. **Copied from the MMELON fork.** |
| 3. Protein + molecule | `MAMMAL_multitarget_finetuning.ipynb` | One model across several targets; the prompt holds the protein and the molecule |
| | `MAMMAL_multitarget_inference.ipynb` | Scores (protein, molecule) pairs, with metrics per target |

The settings, metrics (hit@K, AUROC, PR AUC) and output layout match the MMELON notebooks, so the results of
the two models line up side by side.

### LoRA

Every fine-tuning notebook has a `USE_LORA` setting:

- **`False`** (the default) trains all 458M weights. Closest to the MMELON runs, and needs about 15 GB of GPU
  memory.
- **`True`** freezes the model and trains small added matrices instead, about 0.2% of the weights. Much less
  memory and faster, usually with similar results. A higher `LEARNING_RATE` (1e-4 to 3e-4) suits it.

Either way the saved model is a plain MAMMAL model: with LoRA the matrices are merged into the weights before
saving, so the inference notebooks do not care which was used.

### Mock multi-target data

The protein + molecule notebooks need rows of (protein, molecule, label), which no AIRCHECK screen gives us
yet. `tools/make_mock_multitarget.py` writes a small stand-in to `sample_data/`:

```bash
python aircheck/tools/make_mock_multitarget.py
```

The molecules are real (from the MMELON sample), but the **protein sequences are randomly generated and the
labels are synthetic**. It exists only to check that the notebooks run; its metrics mean nothing.

## Notes

- **Speed:** MAMMAL is about 5 times larger than MMELON, so expect slower training and scoring; a GPU is needed
  for real datasets.
- **GPU:** the environment's PyTorch is built for CUDA 13, which supports T4, L4, A100 and H100 but not V100.
- **Memory:** full fine-tuning needs roughly 15 GB (weights, gradients and the optimizer's state). LoRA needs
  about 4 GB. Cross-validation holds one fold at a time, but keeps each fold's training state on disk, so it
  needs about 5.5 GB of disk per fold while it runs.
