"""Fine-tune MAMMAL for binary hit classification, with or without LoRA.

The same training as ../../archive-to-be-deleted/notebooks/MAMMAL_finetuning_cv.ipynb:
1. Train all folds side by side, one epoch at a time (up to `epochs`), scoring each fold's test rows after
   every epoch; average the metrics over the folds.
2. Pick the epoch by `select_metrics` (first decides, later ones break ties); stop after `patience` epochs
   without improvement. Each fold's model at that epoch is saved as a plain MAMMAL model (LoRA matrices are
   merged in), so scoring is the same with and without LoRA.
3. Predictions: each fold's test rows scored by that fold's model (split foldN), the test file scored by the
   fold models averaged (test_ensemble) and, unless final=False, by one final model trained on all training
   rows for the selected number of epochs (test). The final model never sees the test file.
"""

import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from aircheck_benchmark_core.metrics import HIT_KS, compute_metrics

from . import common
from .common import ANSWER_POSITION, TOKEN_WEIGHTS_KEY

from fuse.dl.lightning.pl_module import LightningModuleDefault  # noqa: E402
from pytorch_lightning.loggers import CSVLogger  # noqa: E402
from mammal.keys import LABELS_TOKENS, SCORES  # noqa: E402
from mammal.losses import LossHead  # noqa: E402
from mammal.lr_schedulers import cosine_annealing_with_warmup_lr_scheduler  # noqa: E402

DEFAULTS = dict(use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_target_modules=["q", "v"],
                epochs=50, patience=5, batch_size=16, learning_rate=1e-5, weight_decay=0.01,
                warmup_fraction=0.1, seed=42, balance_classes=True,
                select_metrics=["hit@200", "hit@100", "hit@50", "rocauc"], final=True)


class HitAtKModule(LightningModuleDefault):
    """MAMMAL's training module, plus the benchmark metrics after every epoch.

    The score of a molecule is the probability the model gives to `<1>` against `<0>` at the answer
    position: the same number the inference notebooks compute."""

    def set_scoring(self, pos_token_id: int, neg_token_id: int):
        self._pos_token_id, self._neg_token_id = pos_token_id, neg_token_id

    @property
    def model(self):
        """The MAMMAL model inside the module (fuse stores it as `_model`)."""
        return self._model

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_scores, self._val_labels = [], []

    def validation_step(self, batch_dict, batch_idx, dataloader_idx=0):
        super().validation_step(batch_dict, batch_idx, dataloader_idx)
        probs = batch_dict[SCORES][:, ANSWER_POSITION, :]  # (batch, vocabulary)
        self._val_scores.append(common.answer_scores(probs, self._pos_token_id, self._neg_token_id))
        answers = batch_dict[LABELS_TOKENS][:, ANSWER_POSITION]
        self._val_labels.append((answers == self._pos_token_id).detach().int().cpu().numpy())

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        scores, labels = np.concatenate(self._val_scores), np.concatenate(self._val_labels)
        for name, value in compute_metrics(labels, scores).items():
            self.log(f"val_{name}", float(value))
        # The loss Lightning logged under its own name, copied to the val_ name the selection uses
        logged = self.trainer.callback_metrics.get("validation.losses.total_loss")
        self.log("val_loss", float(logged) if logged is not None else float("nan"))


def build_model(s: dict):
    """The pretrained model, wrapped in LoRA when use_lora is on."""
    model = common.load_model(None)
    if s["use_lora"]:
        from mammal.lora import get_lora_model

        model = get_lora_model(model, r=s["lora_r"], lora_alpha=s["lora_alpha"], lora_dropout=s["lora_dropout"],
                               target_modules=s["lora_target_modules"])
    return model


def make_module(fold: dict, s: dict, steps: int) -> HitAtKModule:
    """A fresh model and training module for one fold (the trainer restores its weights from disk)."""
    warmup = max(1, int(s["warmup_fraction"] * steps))

    def configure(module):
        opt = torch.optim.AdamW([p for p in module.trainer.model.parameters() if p.requires_grad],
                                lr=s["learning_rate"], weight_decay=s["weight_decay"])
        sched = cosine_annealing_with_warmup_lr_scheduler(
            opt, T_max=max(1, steps - warmup), num_warmup_steps=warmup, eta_min_factor=0.1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    module = HitAtKModule(model=build_model(s),
                          losses={"ce": LossHead(loss_type="ce", sample_token_weights_key=TOKEN_WEIGHTS_KEY)},
                          optimizers_and_lr_schs=configure, model_dir=str(fold["dir"]))
    _, neg_id, pos_id, _ = common.tokenizer()
    module.set_scoring(pos_id, neg_id)
    return module


def save_model(model, path: Path, s: dict, lora_state=None):
    """Save as a plain MAMMAL model, so scoring is the same with and without LoRA."""
    path.mkdir(parents=True, exist_ok=True)
    if s["use_lora"]:
        from peft import set_peft_model_state_dict

        if lora_state is not None:
            set_peft_model_state_dict(model, lora_state)
        model.merge_and_unload()._save_pretrained(str(path))  # gives back a plain Mammal
    else:
        model._save_pretrained(str(path))
    common.tokenizer()[0].save_pretrained(str(path / "tokenizer"))


def train_cv(data: pd.DataFrame, fold_list: list, s: dict, models_dir: Path, log) -> tuple:
    """Train the folds side by side; returns (selected epoch, epochs trained, per fold-epoch metrics)."""
    allowed = [f"hit@{k}" for k in HIT_KS] + ["rocauc", "pr_auc", "precision", "recall", "specificity",
                                              "accuracy", "loss"]
    assert set(s["select_metrics"]) <= set(allowed), f"select_metrics must come from {allowed}"

    folds = []
    for i, (train_idx, test_idx) in enumerate(fold_list):
        train_frame, test_frame = data.iloc[train_idx], data.iloc[test_idx]
        weight = common.positive_weight(train_frame["label"], s["balance_classes"])
        fold = {"fold": i, "dir": models_dir / f"fold_{i}", "pos_weight": weight,
                "test_rows": np.asarray(test_idx), "test": test_frame}
        fold["dir"].mkdir(parents=True, exist_ok=True)
        fold["loaders"] = (
            common.make_loader(train_frame, batch_size=s["batch_size"], shuffle=True, pos_weight=weight),
            common.make_loader(test_frame, batch_size=s["batch_size"], shuffle=False, pos_weight=weight),
        )
        folds.append(fold)

    pl.seed_everything(s["seed"], workers=True)
    rows, best_key, best_epoch, wait = [], None, None, 0
    log(f"Training {len(folds)} folds side by side, up to {s['epochs']} epochs (patience {s['patience']}), "
        f"{'LoRA' if s['use_lora'] else 'all weights'}")
    for epoch in range(s["epochs"]):
        for fold in folds:
            state_path = fold["dir"] / "training_state.ckpt"
            pl.seed_everything(s["seed"] + 1000 * fold["fold"] + epoch, verbose=False)
            # fuse's module logs the learning rate, which Lightning only allows with a logger
            trainer = pl.Trainer(max_epochs=epoch + 1, logger=CSVLogger(str(fold["dir"]), name="logs"),
                                 enable_checkpointing=False, default_root_dir=fold["dir"], accelerator="auto",
                                 devices=1, num_sanity_val_steps=0, enable_model_summary=False,
                                 enable_progress_bar=False)
            module = make_module(fold, s, len(fold["loaders"][0]) * s["epochs"])
            # Continue this fold from its previous epoch (weights and optimizer state are restored)
            trainer.fit(module, *fold["loaders"], ckpt_path=state_path if epoch > 0 else None)
            trainer.save_checkpoint(state_path)
            rows.append(dict(fold=fold["fold"], epoch=epoch, **{k.removeprefix("val_"): float(v) for k, v
                                                                in trainer.callback_metrics.items()
                                                                if k.startswith("val_")}))
            if s["use_lora"]:
                from peft import get_peft_model_state_dict

                fold["last_lora"] = {k: v.detach().cpu().clone()
                                     for k, v in get_peft_model_state_dict(module.model).items()}
            log(f"epoch {epoch}, fold {fold['fold'] + 1}/{len(folds)} done")
            del trainer, module
            gc.collect()
            if common.DEVICE == "cuda":
                torch.cuda.empty_cache()

        metrics = pd.DataFrame(rows)
        means = metrics[metrics["epoch"] == epoch].mean(numeric_only=True)
        # Larger is better; loss is negated so a lower loss ranks higher. Ties keep the earlier epoch.
        key = tuple(float(means[m]) * (-1 if m == "loss" else 1) for m in s["select_metrics"])
        if best_key is None or key > best_key:
            best_key, best_epoch, wait = key, epoch, 0
            for fold in folds:
                if s["use_lora"]:
                    fold["best_lora"] = fold["last_lora"]  # a few MB, kept in memory
                else:
                    shutil.copy2(fold["dir"] / "training_state.ckpt", fold["dir"] / "best_state.ckpt")
            status = "best so far, state kept"
        else:
            wait += 1
            status = f"no improvement ({wait}/{s['patience']})"
        log(f"epoch {epoch:3d} mean over folds:  "
            + "  ".join(f"{m} {means[m]:.4f}" for m in s["select_metrics"]) + f"  -> {status}")
        metrics.to_csv(models_dir / "cv_metrics_per_epoch.csv", index=False)
        if wait >= s["patience"]:
            log(f"Stopping early: no improvement of the fold average for {s['patience']} epochs")
            break

    log(f"Saving each fold's model from epoch {best_epoch}")
    for fold in folds:
        model = build_model(s)
        if not s["use_lora"]:
            state = torch.load(fold["dir"] / "best_state.ckpt", map_location="cpu")["state_dict"]
            # Lightning stores the module's weights under "_model."; strip that to get the model's own keys
            state = {k.split(".", 1)[1] if k.startswith("_model.") else k: v for k, v in state.items()}
            result = model.load_state_dict(state, strict=False)
            assert not result.missing_keys, f"fold {fold['fold']}: missing weights {result.missing_keys[:5]}"
        save_model(model, fold["dir"] / "best_model", s, lora_state=fold.get("best_lora"))
        del model
        gc.collect()
    for fold in folds:
        for name in ("training_state.ckpt", "best_state.ckpt"):
            path = fold["dir"] / name
            if path.exists():
                path.unlink()
    return best_epoch, epoch + 1, pd.DataFrame(rows), folds


def train_final(data: pd.DataFrame, epochs: int, s: dict, models_dir: Path, log) -> Path:
    """One model on all training rows for a fixed number of epochs (no test file involved)."""
    final_dir = models_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    fold = {"fold": "final", "dir": final_dir}
    weight = common.positive_weight(data["label"], s["balance_classes"])
    loader = common.make_loader(data, batch_size=s["batch_size"], shuffle=True, pos_weight=weight)
    pl.seed_everything(s["seed"], workers=True)
    module = make_module(fold, s, len(loader) * epochs)
    trainer = pl.Trainer(max_epochs=epochs, logger=CSVLogger(str(final_dir), name="logs"),
                         enable_checkpointing=False, default_root_dir=final_dir, accelerator="auto", devices=1,
                         limit_val_batches=0, num_sanity_val_steps=0, enable_model_summary=False,
                         enable_progress_bar=False)
    log(f"Training the final model on all {len(data):,} training rows for {epochs} epoch(s)")
    trainer.fit(module, loader)
    model = module.model
    if s["use_lora"]:
        from peft import get_peft_model_state_dict

        lora_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        save_model(model, final_dir / "best_model", s, lora_state=lora_state)
    else:
        save_model(model, final_dir / "best_model", s)
    del trainer, module, model
    gc.collect()
    return final_dir / "best_model"


def run(data: pd.DataFrame, fold_list: list, test: pd.DataFrame, s: dict, out_dir: Path, log) -> tuple:
    """Returns (predictions, report extras, settings used). `data` and `test` have columns smiles, label."""
    s = {**DEFAULTS, **{k: v for k, v in s.items() if v is not None}}
    bad = data["smiles"].isna() | (data["smiles"].astype(str).str.strip() == "")
    assert not bad.any(), f"{bad.sum()} training rows have no SMILES, e.g. rows {data.index[bad][:5].tolist()}"
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    # A fold keeps its training state (~5.5 GB) while training, plus the saved model (~1.9 GB)
    needed_gb = (5.5 + 1.9) * len(fold_list) + (1.9 if s["final"] else 0)
    free_gb = shutil.disk_usage(models_dir).free / 1e9
    assert free_gb > needed_gb + 2, (
        f"Need ~{needed_gb:.0f} GB free for {len(fold_list)} folds, only {free_gb:.0f} GB available. "
        f"Use fewer folds, set use_lora, or free up space.")

    best_epoch, epochs_run, cv_metrics, folds = train_cv(data, fold_list, s, models_dir, log)

    parts = []
    log("Scoring each fold's test rows with that fold's model")
    for fold in folds:
        model = common.load_model(fold["dir"] / "best_model")
        scores = common.score_with(model, fold["test"]["smiles"], s["batch_size"], log=log)
        parts.append(common.predictions_frame(f"fold{fold['fold']}", fold["test_rows"],
                                              fold["test"]["smiles"], fold["test"]["label"], scores))
        del model
        gc.collect()

    log("Scoring the test file with each fold's model, then averaging")
    fold_scores = []
    for fold in folds:
        model = common.load_model(fold["dir"] / "best_model")
        fold_scores.append(common.score_with(model, test["smiles"], s["batch_size"], log=log))
        del model
        gc.collect()
    parts.append(common.predictions_frame("test_ensemble", test.index, test["smiles"], test["label"],
                                          np.mean(fold_scores, axis=0)))

    if s["final"]:
        final_model = train_final(data, best_epoch + 1, s, models_dir, log)
        log("Scoring the test file with the final model")
        model = common.load_model(final_model)
        parts.append(common.predictions_frame("test", test.index, test["smiles"], test["label"],
                                              common.score_with(model, test["smiles"], s["batch_size"], log=log)))
        del model
        gc.collect()

    (models_dir / "selection.json").write_text(json.dumps(
        {"select_metrics": s["select_metrics"], "best_epoch": best_epoch, "epochs_trained": epochs_run,
         "use_lora": s["use_lora"],
         "models": [str(f["dir"] / "best_model") for f in folds]}, indent=2) + "\n")
    extras = {"selected epoch": best_epoch, "epochs trained": epochs_run}
    return pd.concat(parts, ignore_index=True), extras, s
