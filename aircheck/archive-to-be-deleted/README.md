# Archive: to be deleted

The AIRCHECK notebooks as they were before the benchmark code. **Nothing here is used any more** and the whole
folder will be removed once a full benchmark run on a GPU has replaced them.

It is kept for now so the old way can be compared with the new one, and so nothing is lost by accident.

| Here | Replaced by |
|---|---|
| `notebooks/` (single-target fine-tuning, inference, embeddings, LightGBM, folds) | [../benchmark/](../benchmark/): `run_all.sh`, `run.py`, `methods/` and 5 thin notebooks |
| `notebooks/MAMMAL_multitarget_*.ipynb` | nothing: the benchmark compares models on one target at a time, so multi-target work stays here |

Two things to know if you open them:

- `MAMMAL_finetuning_cv.ipynb` **cannot run as written**: its trainer passes `logger=False` while fuse's module
  always adds a learning-rate monitor, so Lightning stops with
  `Cannot use LearningRateMonitor callback with Trainer that has no logger`.
  [../benchmark/methods/finetune.py](../benchmark/methods/finetune.py) gives each trainer a `CSVLogger` instead.
- They write to `../output/<DATASET>/` in the old layout, and their results are **not** comparable with
  benchmark results in one report: the benchmark reads `run.json` files, which these do not write.
