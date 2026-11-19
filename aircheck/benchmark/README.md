# MAMMAL benchmark runs

`run.py` runs one MAMMAL method on one dataset from the command line and writes its predictions in the
standard format of [aircheck-benchmark-core](https://github.com/ShagReza/aircheck-benchmark-core). Core then
computes the metrics and the report, the same way for every model in the benchmark, so MAMMAL's numbers can be
compared directly with MMELON's and with the models added later.

This folder is plain Python: `run_all.sh` → `run.py` → `methods/`. The notebooks in [notebooks/](notebooks/)
do the same steps one at a time for exploring, and define no code of their own. The old notebooks in
[../notebooks/](../notebooks/) are untouched and still work on their own.

## Setup

1. Set up the MAMMAL environment with [../INSTALL.md](../INSTALL.md).
2. **Install aircheck-benchmark-core into that environment** (`run.py` imports it):
   ```bash
   git clone git@github.com:ShagReza/aircheck-benchmark-core.git ~/repos/aircheck-benchmark-core
   conda activate ~/envs/biomed-mammal
   pip install -e ~/repos/aircheck-benchmark-core
   aircheck-bench --version   # check
   ```

## An experiment

An **experiment** is one dataset with one set of conditions, and everything it produces stays in its own
folder under `aircheck/output/`, so this repo is self-contained: whoever is interested in MAMMAL finds the
folds, the models, the predictions and the report in one place.

```
aircheck/output/<experiment>/
├── experiment.json      the train/test files and fold settings this experiment was made with
├── folds/               the fold files (made here the first time, then reused)
├── <run name>/          predictions.csv, run.json, progress.log, models/
├── embeddings/          shared by the LightGBM runs of the same model
├── report.md            every run in this experiment, with report_summary.csv next to it
└── run_all.log
```
Use a new name for other conditions, for example `wdr91-cluster` and `wdr91-stratified`. `experiment.json` is
checked on every run, so one name cannot end up holding results from two different datasets.

**Use the same fold files as the other models.** Point `--folds` at the fold `.npz` another repo's experiment
already made, or make them once outside the repos and pass that path to both. Identical folds are what makes
the comparison fair; core checks that every run's fold predictions are exactly that fold's test rows.

## Run everything (the recipe)

```bash
aircheck/benchmark/run_all.sh --experiment wdr91-cluster \
    --train ~/Datasets/wdr91/train.parquet --test ~/Datasets/wdr91/test.parquet
```
That one command makes the folds (if they are not there yet), does the runs below one after another, and ends
with `aircheck-bench report`, which writes **`aircheck/output/wdr91-cluster/report.md`** and
`report_summary.csv` next to it. `aircheck/benchmark/run_all.sh --help` lists every option.

| Run | Method |
|---|---|
| `finetune_lora` | fine-tune LoRA adapters |
| `finetune_full` | fine-tune every weight (a large GPU and ~7 GB of disk per fold) |
| `lgbm_pretrained` | LightGBM on embeddings from the pretrained model |
| `lgbm_finetuned` | LightGBM on embeddings from the `finetune_lora` final model |

- **Start cheap:** `--only finetune_lora,lgbm_pretrained`.
- **Resumable:** a run that already has `run.json` is skipped, so starting the script again after an
  interruption (or after the VM shut down) continues where it stopped. Delete a run's folder to redo it.
- **`--only a,b` / `--skip a,b`** pick runs (`--list` prints the names), and any further options go to every
  `run.py` call, e.g. `--num-workers 2` or `--epochs 3` for a quick try. `--only none` just rewrites the report.
- **Fold method:** `--cv-method cluster` (default), `stratified` or `random`, with `--n-folds`; or `--folds
  <npz>` to use fold files from somewhere else.
- **Long runs:** `nohup aircheck/benchmark/run_all.sh --experiment wdr91-cluster ... > /dev/null 2>&1 &`,
  then follow `aircheck/output/wdr91-cluster/run_all.log`.

## Run one method

```bash
python aircheck/benchmark/run.py --method finetune --lora \
    --folds aircheck/output/wdr91-cluster/folds/train_cluster_cv_folds.npz \
    --test ~/Datasets/wdr91/test.parquet --out aircheck/output/wdr91-cluster/finetune_lora
```
The training molecules are the folds CSV next to `--folds` (`<name>_cv_data.csv`), so the folds and the rows
they refer to always match. Follow a run in `<out>/progress.log`.

## Methods

| `--method` | What it does | Same steps as |
|---|---|---|
| `finetune` | Fine-tunes MAMMAL on every fold side by side, picks the epoch by the fold average, then trains a final model on all training rows for that many epochs. `--lora` trains LoRA adapters and merges them into the saved model, so scoring is the same either way | `MAMMAL_finetuning_cv`, `MAMMAL_finetuning`, `MAMMAL_inference*` |
| `embed_lgbm` | Embeds the molecules with MAMMAL (the pretrained model by default, or `--encoder <model folder>`), then LightGBM per fold and on all rows | `MAMMAL_embedding_extraction`, `MAMMAL_embedding_lgbm_cv`, `MAMMAL_embedding_lgbm_inference` |

Settings (left out = the default in `methods/<method>.py`, the same as the notebooks):

| Option | Method | Default |
|---|---|---|
| `--lora`, `--lora-r`, `--lora-alpha`, `--lora-dropout` | finetune | off, 8, 16, 0.05 (with LoRA a learning rate of 1e-4 to 3e-4 usually works better) |
| `--epochs`, `--patience` | finetune | 50, 5 |
| `--batch-size` | both | 16 (finetune), 32 (embed_lgbm) |
| `--learning-rate`, `--weight-decay`, `--warmup-fraction` | finetune | 1e-5, 0.01, 0.1 |
| `--select-metrics` | finetune | hit@200 hit@100 hit@50 rocauc |
| `--no-final` | finetune | final model on (skip it to save time; then only `test_ensemble`) |
| `--encoder` | embed_lgbm | the pretrained model |
| `--lgbm-params` | embed_lgbm | `suggested` (or `default`) |
| `--embeddings-dir` | embed_lgbm | `<out>/../embeddings/<model>/` |
| `--no-balance` | both | classes balanced (label 1 weighted by the class counts) |
| `--seed` | both | 42 |
| `--num-workers` | both | all CPUs, up to 24; use 0–2 on a small machine |
| `--name` | both | display name in the report |

## Output

In `--out`:
```
predictions.csv   split, row, SMILES, LABEL, score: one row per scored molecule
                  split = fold0, fold1, ... (each fold's test rows), test_ensemble (test file, fold models
                  averaged), test (test file, final model / model on all rows)
run.json          model, method, display name, settings, selected epoch, hardware, this repo's commit, core version
progress.log      what happened, with times
models/           fold_<i>/best_model/ and final/best_model/ (plain MAMMAL models) for finetune;
                  fold_<i>.txt and full.txt for embed_lgbm
config.json       embed_lgbm only: the model and view its LightGBM models were trained on
```
`embed_lgbm` writes the embeddings one folder up, so every rerun with the same model shares them:
```
<out>/../embeddings/<pretrained | finetuned-<run name>>/<train | test>/
├── config.json                     model, number of molecules, token limits
├── mammal/part-00000.parquet       row_index, smiles, embedding — the same layout as the extraction notebook
└── invalid_smiles.csv              rows RDKit cannot read (no embedding, empty score)
```
`run.py` prints the scores at the end (computed by core, which also checks that every fold's predictions are
exactly that fold's test rows). Molecules with a SMILES that RDKit cannot read get an empty score; core ranks
them last.

## Differences from the notebooks

- **Final model:** `MAMMAL_finetuning.ipynb` keeps the best epoch as measured on its test file. Here the final
  model trains for a fixed number of epochs (the CV-selected epoch + 1), so the test file is never used to
  choose anything.
- **Fold predictions:** the notebooks keep only fold metrics. Here each fold's test rows are scored with that
  fold's model and saved, with the scoring code of the inference notebooks (the probability of `<1>` against
  `<0>` at the answer position). The epoch is still chosen from the metrics logged during training.
- **Folds and metrics** come from aircheck-benchmark-core instead of `cluster_cv_folds.ipynb` and the
  notebooks' `compute_metrics`; both give the same results. The LightGBM part is core's `lgbm.py`, shared with
  the other models.
- **A logger for training:** fuse's module logs the learning rate, which Lightning allows only when the
  trainer has a logger, so each fold gets a `CSVLogger` in `models/fold_<i>/logs/`.
  `MAMMAL_finetuning_cv.ipynb` passes `logger=False` and stops with
  `Cannot use LearningRateMonitor callback with Trainer that has no logger`.
- **One folder per experiment** instead of `output/<dataset>/<run name>/`.
- **Multi-target** (`MAMMAL_multitarget_*`) is not a benchmark method here: the benchmark compares models on
  one target at a time. Those notebooks are untouched in [../notebooks/](../notebooks/).

## A note on reproducing numbers

Embeddings computed on a GPU and on a CPU differ by about 1e-6 per value, which is enough to move LightGBM's
splits and change hit@50 by a molecule or two, so the metrics are **not** identical across machines. Keep the
embeddings an experiment produced instead of recomputing them on other hardware half-way through, and compare
models within one experiment, which is what the report does. `run.json` records the hardware, the fork commit
and the core version of every run.
