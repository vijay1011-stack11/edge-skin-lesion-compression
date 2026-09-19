# Edge AI Solo Project — HAM10000 Skin Lesion Classification + Compression

Plan A, locked. Research reference project for EPFL / US Master's applications.

## What this is

Train a MobileNetV2 classifier on the HAM10000 dermoscopic image dataset (7 skin
lesion classes), then run a systematic model-compression study (pruning, INT8
post-training quantization, quantization-aware training, and combinations),
producing a Pareto frontier of accuracy vs. model size vs. CPU latency —
with a specific focus on **per-class accuracy degradation** (not just aggregate
accuracy) since HAM10000 is badly imbalanced and the clinically dangerous
classes (`mel`, `akiec`) are the ones most likely to quietly break under
compression.

**Why this angle specifically**: see `literature/lit-scan.md`. Short version —
plain "train MobileNetV2 + quantize it" has been done before (papers 1, 4, 5
in that file). What's missing from the existing literature, and what this
project does instead: real measured CPU latency (not FLOPs-as-proxy),
per-class F1 degradation curves, and a full combined pruning+quantization
Pareto sweep on one consistent backbone/dataset.

## IMPORTANT: this code was written in a sandbox with no internet access

This code was built and smoke-tested (bugs fixed, confirmed to run end-to-end)
in an environment that cannot reach Kaggle, Harvard Dataverse, HuggingFace, or
`storage.googleapis.com` (where Keras fetches ImageNet weights from). That
means:
- **You cannot get the real HAM10000 dataset here.** You need to run this on
  Google Colab (which has direct, fast Kaggle access and a free GPU) or your
  own machine.
- The smoke test (`scripts/smoke_test.py`) only proves the *code* has no bugs —
  it trains on 42 fake random-noise images for 1 epoch with random (not
  ImageNet) weights. The accuracy numbers from it are meaningless. Real
  numbers only come from a real run on real data.

**Recommended: run this on Google Colab.**

## Setup (Google Colab)

1. Upload this whole folder to Colab, or `git clone`/copy it into your Colab
   working directory.
2. Get a Kaggle API token: kaggle.com → your account → "Create New API Token"
   → downloads `kaggle.json`.
3. In a Colab cell:
   ```python
   from google.colab import files
   files.upload()  # upload kaggle.json
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !pip install -q kaggle
   !kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/HAM10000 --unzip
   ```
   This gives you `data/HAM10000/HAM10000_metadata.csv` and the two image
   folders — exactly the layout `src/config.py` expects.
4. `pip install -r requirements.txt` (Colab already has TensorFlow; you mainly
   need `tensorflow-model-optimization`).
5. Make sure `MOBILENET_WEIGHTS = "imagenet"` in `src/config.py` (it is, by
   default — this only needs changing for offline smoke-testing).

## Running the pipeline

In order:

```bash
python -m src.train_baseline   # trains MobileNetV2, saves models/baseline_final.keras
                                # + reports/baseline_eval.json (per-class F1 etc.)
python -m src.compress         # produces every compressed variant in models/*.tflite:
                                #   ptq_dynamic, ptq_int8, pruned_{30,50,70}, pruned_{..}_int8, qat_int8
python -m src.benchmark        # benchmarks every .tflite file: accuracy, per-class F1,
                                # size (KB), CPU latency (ms) -> reports/pareto_results.csv
```

`reports/pareto_results.csv` is the table the write-up is built around — one
row per compression configuration.

## Repo layout

```
src/
  config.py       # all paths, hyperparameters, class names — edit DATA_DIR here
  data.py         # lesion_id-aware split (prevents data leakage), tf.data pipeline,
                   # class weights for imbalance, representative-dataset generator for PTQ
  model.py        # MobileNetV2 transfer-learning architecture
  train_baseline.py
  compress.py     # pruning / PTQ / QAT / combined
  evaluate.py     # per-class metrics, confusion matrix, critical-class F1
  benchmark.py    # latency + size + accuracy sweep across all .tflite variants
scripts/
  make_synthetic_data.py   # generates fake data for offline smoke-testing only
  smoke_test.py             # runs the full pipeline end-to-end on fake data, fast
literature/
  lit-scan.md      # verified literature scan + the novelty/differentiation argument
reports/           # eval reports + the final Pareto CSV land here
models/            # trained checkpoints + .tflite files land here
```

## Known correctness details already handled in the code (don't redo these wrong)

- **Lesion-level split, not image-level.** HAM10000 has multiple images per
  lesion (`lesion_id` repeats). `data.split_by_lesion()` uses `GroupShuffleSplit`
  so no lesion's images appear in more than one of train/val/test — asserted
  explicitly in code. Splitting by `image_id` instead silently inflates your
  reported accuracy through near-duplicate leakage. This is the single most
  common mistake in HAM10000 write-ups you'll find online — don't repeat it.
- **Class imbalance.** `nv` is ~67% of the dataset. `compute_class_weights()`
  applies inverse-frequency weighting during training; `evaluate.py` always
  reports macro-F1 and per-class F1 alongside plain accuracy, and separately
  tracks `mel`/`akiec` F1 as "critical class" metrics — a compressed model
  that keeps 90% overall accuracy while `mel` F1 collapses is a *failure*,
  and the write-up should say so if it happens.
- **QAT/nested-model gotcha (already fixed):** `model.py` builds off
  `base.input`/`base.output` directly rather than calling the MobileNetV2
  base as a nested layer. Calling it as `base(inputs)` works fine for normal
  training but breaks `tfmot.quantization.keras.quantize_model()` later with
  a "Quantizing a keras Model inside another keras Model" error. If you ever
  rewrite `model.py`, keep this pattern.

## Timeline (from the original brief)

- Now – early Oct: dataset acquired on Colab, literature scan done (this file),
  baseline training running
- Oct: baseline results finalized, compression experiments start
- Nov: full Pareto sweep + per-class degradation analysis
- Late Nov – early Dec: write up as a technical report / arXiv-style preprint

## Boundary

This project's code/data/experiments stay here. Only the finished technical
report crosses over into the "Masters" project as a research-reference
document once it's done.
