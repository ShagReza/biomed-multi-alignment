# MAMMAL prediction API

A small web API that scores molecules (SMILES) with a fine-tuned MAMMAL model, so a website can request
predictions. No training code is included.

The two inference files here are the one place MAMMAL scoring is written down: the benchmark imports them
(see [../benchmark/methods/common.py](../benchmark/methods/common.py)), so a benchmark score and an API score
for the same molecule and model come from exactly the same code. It mirrors the MMELON fork's
`aircheck/serving/`, so both models are served the same way.

| File | Purpose |
|---|---|
| `inference_mammal.py` | Scores SMILES with a fine-tuned MAMMAL model: the final model, or the CV fold models averaged. Also holds the prompt, the tokenizer and the batching. |
| `inference_embedding_lgbm.py` | Scores SMILES with LightGBM on MAMMAL embeddings: the final model (`full.txt`) or the fold models. |
| `api.py` | The web API (FastAPI): endpoints, request and response formats, errors. |
| `requirements-serving.txt` | Extra packages for the API only (FastAPI, Uvicorn); not needed to run the benchmark. |
| `.env.example` | The settings the API reads (names with placeholder values). Copy to `.env` for local use. |
| `models.example.json` | Example `MODELS_CONFIG`: one entry per target, with its model type and path. |
| `Dockerfile` | Builds the CPU image that runs the API. |
| `Dockerfile.dockerignore` | What goes into the image: only the code, never models, data or settings. |

## Inference files

Both files work the same way: `load_model(...)` once, then `predict(model, smiles)`, which returns one result
per SMILES in the same order: `{"smiles": ..., "score": 0-1 or None, "valid": True/False}` (invalid SMILES get
no score). Run them from this folder.

Paths below are benchmark runs, `output/<experiment>/<run name>/` (see
[../benchmark/README.md](../benchmark/README.md)).

| | Final model (default, `models="full"`) | CV fold models, averaged (`models="folds"`) |
|---|---|---|
| `inference_mammal.py` | the run's `models/final` (or the `best_model` folder inside it) | the run's `models` folder, which holds `fold_<i>/best_model` |
| `inference_embedding_lgbm.py` | the LightGBM run folder (uses `models/full.txt`) | the same folder (uses `models/fold_*.txt`) |

```python
from inference_mammal import load_model, predict
model = load_model("../output/wdr91-cluster/finetune_lora/models/final")        # final model
model = load_model("../output/wdr91-cluster/finetune_lora/models", models="folds")  # fold ensemble
predict(model, ["CCO", "c1ccccc1"])

from inference_embedding_lgbm import load_model, predict
model = load_model("../output/wdr91-cluster/lgbm_pretrained")                   # pretrained embeddings
model = load_model("../output/wdr91-cluster/lgbm_finetuned",                    # fine-tuned embeddings
                   encoder_path="../output/wdr91-cluster/finetune_lora/models/final")
predict(model, ["CCO", "c1ccccc1"])
```

- `score` is the probability the model gives to `<1>` against `<0>` at the answer position.
- LoRA runs save a plain MAMMAL model (the adapters are merged in), so loading is the same either way.
- For LightGBM on **pretrained** embeddings, `encoder_path` can be left out: the run's `config.json` says
  which model the embeddings came from, and the pretrained one is downloaded from Hugging Face if needed.
- Folds are more accurate but K times slower and use K times the memory; the API uses the final model.

## API

All models in `MODELS_CONFIG` are loaded once at startup.

Run it locally:

```bash
cd aircheck/serving
cp models.example.json models.json        # then edit the paths to your models
MODELS_CONFIG=models.json ~/envs/biomed-mammal/bin/uvicorn api:app --port 8090
```

- Use the `biomed-mammal` env's `uvicorn` (or `micromamba activate ~/envs/biomed-mammal` first).
- Startup loads every model (about 1.9 GB each), so give it a minute; then open http://localhost:8090/docs to
  try the endpoints (`/` itself shows "Not Found"; that is normal).
- Port 8080 is used by Jupyter on the VM, so use another port such as 8090. Stop the server with Ctrl+C.
- Optional settings: `MAX_SMILES` (default 50), `BATCH_SIZE` (16) and `DEVICE` (`cpu` or `cuda`), see
  `.env.example`.

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /targets` | each target with its model type and number of models |
| `POST /predict` | body `{"target": "WDR91", "smiles": ["CCO", ...]}` (at most `MAX_SMILES`, default 50) → `{"target", "model", "results": [{"smiles", "score", "valid", "rank"}]}`, same order as sent; rank 1 = highest score |

Errors: 404 for an unknown target, 422 for an empty or too long list.

**Speed and memory.** MAMMAL is a 458M-parameter model: on CPU it takes roughly 0.5 s per molecule, so a
50-molecule request takes about 25 s, and each loaded model needs about 2 GB of memory. Size the machine for
the number of targets you serve, or use a GPU (`DEVICE=cuda`).

## Docker

The image holds the code and its packages (CPU-only PyTorch, plus TensorFlow which `fuse-med-ml` requires:
about 5-6 GB). **Models are not in it**: they are mounted when the container runs, so a model can change
without rebuilding.

**Build**, from the repository root:

```bash
cd <repository root>
docker build -f aircheck/serving/Dockerfile -t mammal-api .
```

If docker says "permission denied", add yourself to the docker group once (`sudo usermod -aG docker $USER`),
then run `newgrp docker` or open a new terminal.

**Run**, giving it the models folder and the config:

```bash
docker run -p 8090:8080 \
  -v /path/to/models:/models:ro \
  -v /path/to/models-docker.json:/config/models.json:ro \
  -v $HOME/.cache/huggingface:/cache/huggingface:ro \
  -e MODELS_CONFIG=/config/models.json \
  mammal-api
# then open http://localhost:8090/docs
```

- The paths inside that `models.json` must be the paths **inside** the container (`/models/...`), not the
  host's, so keep a separate copy from the one used without Docker.
- The Hugging Face cache mount is optional but saves downloading the tokenizer (and, for a pretrained-embedding
  model, the 1.8 GB base model) on the first request. Without it the container needs internet.
- The container serves port **8080** (what Cloud Run expects); `-p 8090:8080` maps it to 8090 on your machine.
  It reads `PORT`, `MODELS_CONFIG`, `MAX_SMILES`, `BATCH_SIZE` and `DEVICE`.
- `:ro` mounts the models read-only.

**Push to Artifact Registry** (the AIRCHECK project's repository, as for MMELON):

```bash
gcloud auth configure-docker northamerica-northeast2-docker.pkg.dev
docker tag mammal-api \
  northamerica-northeast2-docker.pkg.dev/aircheck-398319/ml-benchmarking-images/mammal-api:v1
docker push \
  northamerica-northeast2-docker.pkg.dev/aircheck-398319/ml-benchmarking-images/mammal-api:v1
```

**To deploy it** (Cloud Run or any other host), three things are needed: the image above, a bucket holding the
models and a `models.json` with `/models/...` paths, and the bucket mounted at `/models` with
`MODELS_CONFIG=/models/models.json`. The website then calls `POST /predict`.

Status: the code is written and the LightGBM path is tested end to end. The fine-tuned path and the Docker
image are **not yet tested**, because no MAMMAL fine-tuning run exists on a GPU yet.
