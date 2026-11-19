# Benchmark notebooks

The same work as `run_all.sh`, one step at a time, for exploring and explaining. They **define no model code**:
every cell calls [../methods/](../methods/), [../run.py](../run.py) or aircheck-benchmark-core, so a notebook
and a benchmark run can never drift apart.

| Notebook | Step | Replaces, in [../../archive-to-be-deleted/notebooks/](../../archive-to-be-deleted/notebooks/) |
|---|---|---|
| [01_folds.ipynb](01_folds.ipynb) | the experiment's cross-validation folds | `cluster_cv_folds.ipynb` |
| [02_finetune.ipynb](02_finetune.ipynb) | fine-tune MAMMAL (LoRA or all weights), score the test set | `MAMMAL_finetuning*`, `MAMMAL_inference*` |
| [03_embeddings_lgbm.ipynb](03_embeddings_lgbm.ipynb) | embeddings, then LightGBM | `MAMMAL_embedding_extraction`, `MAMMAL_embedding_lgbm_*` |
| [04_inference.ipynb](04_inference.ipynb) | score your own molecules with a trained run | the "score my own molecules" part of `*_inference*` |
| [05_report.ipynb](05_report.ipynb) | the report for the whole experiment | — (new; MMELON's `collect_results.ipynb`) |

**Before you start:** the `biomed-mammal` kernel with aircheck-benchmark-core installed — see
[../README.md](../README.md).

**How to use them:** every notebook starts with a settings cell; the first setting is `EXPERIMENT`, the folder
under `aircheck/output/` that holds the folds, the runs and the report. Use the same name in all of them, run
`01_folds` first (or reuse another model's fold files), then any of `02`/`03`, and `05` at the end. Run them
from this folder.

The sample paths point at the single-target sample files in the MMELON fork next to this repo, because
MAMMAL's own `sample_data/` is multi-target mock data without fingerprints. Point them at your own data.

Nothing here is needed for a benchmark run: `run_all.sh` does the same steps without opening a notebook.
Multi-target work stays in [../../archive-to-be-deleted/notebooks/](../../archive-to-be-deleted/notebooks/); the benchmark compares models on one target.
