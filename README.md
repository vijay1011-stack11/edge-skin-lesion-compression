# Edge AI: Compression Trade-offs for Skin Lesion Classifiers

**Author:** Vijayabalaguroo S
**Report:** [Technical report (PDF)](reports/technical_report.pdf)

A measured, per-class study of how post-training quantization, pruning and
quantization-aware training affect a MobileNetV2 skin lesion classifier
trained on HAM10000. Every variant is evaluated on file size, real
single-thread CPU latency, accuracy, and per-class F1, with explicit tracking
of melanoma (`mel`) and actinic keratosis (`akiec`), the classes where a miss
costs the most.

## Motivation

Rural clinics and disaster zones are where image-based AI would help most and
where a reliable connection to a cloud model is least likely. Running a
compressed model on the device removes that dependency, but compression is
not free: it can cost accuracy unevenly across classes, and a smaller model is
not automatically a faster one on a given chip.

This project asks, for one concrete task, what each compression technique
costs in accuracy, class by class, and in real milliseconds on a CPU, instead
of relying on parameter counts or FLOPs. HAM10000 is strongly imbalanced
(about two thirds of images are benign nevi, `nv`), which makes aggregate
accuracy a misleading metric. The literature scan behind this framing is in
`literature/lit-scan.md`.

## Results

Test set: 1,527 images, lesion-level split. Latency is the mean of two
benchmark runs (100 timed single-image calls each, after 5 warm-up calls),
one thread, Google Colab CPU.

| Model | Size (KB) | Smaller | Latency (ms) | Accuracy | Macro-F1 | `mel` F1 | `akiec` F1 |
|---|---|---|---|---|---|---|---|
| **Baseline (float32)** | 8682.7 | 1.0× | 6.9 | 0.685 | 0.514 | 0.446 | 0.389 |
| PTQ dynamic-range | 2447.5 | 3.5× | 17.7 | 0.687 | 0.525 | 0.421 | 0.421 |
| PTQ full-INT8 | 2645.6 | 3.3× | 10.7 | 0.713 | 0.517 | 0.431 | 0.408 |
| QAT (5 epochs) → INT8 | 2653.0 | 3.3× | 10.5 | 0.567 | 0.359 | 0.239 | 0.236 |
| Pruned 30% (dyn-range) | 2447.5 | 3.5× | 22.2 | 0.560 | 0.404 | 0.334 | 0.276 |
| Pruned 30% + INT8 | 2645.6 | 3.3× | 10.9 | 0.597 | 0.419 | 0.333 | 0.263 |
| Pruned 50% (dyn-range) | 2447.5 | 3.5× | 21.9 | 0.623 | 0.377 | 0.229 | 0.036 |
| Pruned 50% + INT8 | 2645.6 | 3.3× | 11.5 | 0.635 | 0.404 | 0.322 | 0.036 |
| Pruned 70% (dyn-range) | 2447.5 | 3.5× | 19.6 | 0.126 | 0.037 | 0.000 | 0.000 |
| Pruned 70% + INT8 | 2645.6 | 3.3× | 10.5 | 0.121 | 0.034 | 0.000 | 0.000 |

Pruned rows without "+ INT8" are dynamic-range INT8 models.
Full results: [`reports/pareto_results.csv`](reports/pareto_results.csv) · Baseline evaluation with confusion matrix: [`reports/baseline_v3_eval.json`](reports/baseline_v3_eval.json)

### Key findings

1. **INT8 post-training quantization lost nothing measurable.** Full-INT8 is
   3.3× smaller (8,683 KB → 2,646 KB) at accuracy 0.713 and macro-F1 0.517;
   dynamic-range is 3.5× smaller at macro-F1 0.525. These differences from the
   baseline are within sampling noise (about ±2.3 accuracy points), so the
   result is "no loss", not "an improvement".
2. **Aggregate accuracy hid a class-level collapse under pruning.** From 30%
   to 50% sparsity, accuracy *rose* (0.560 → 0.623) while `akiec` F1 fell
   from 0.276 to 0.036, as the model drifted toward predicting `nv`. A pruning
   level chosen from an accuracy target would have passed 50% and silently
   stopped detecting `akiec`.
3. **70% sparsity produced a failed model** (accuracy 0.126, macro-F1 0.037),
   worse than always predicting `nv`.
4. **Smaller was not faster on this CPU.** Float32 baseline: 6.9 ms.
   Full-INT8: 10.5–11.5 ms. Dynamic-range: 17.7–22.2 ms. The ordering was
   identical in both benchmark runs. Unstructured pruning changed neither
   file size nor latency, because the stored arrays and kernels are dense.
5. **QAT underperformed** (macro-F1 0.359), most likely because 5 epochs
   was not enough training.

On latency vs macro-F1, only the baseline, PTQ full-INT8 and PTQ
dynamic-range are non-dominated. The practical case for quantization here is
memory and storage, not speed.

### What these results do not show

- **That pruning is intrinsically harmful.** The pruning recovery recipe
  (8 epochs at lr 1e-5, class weights without the `mel`/`akiec` boost,
  uniform sparsity, final-epoch weights) is a likely cause and was not
  isolated.
- **That INT8 is slow in general.** One shared x86 Colab CPU, no profiling.
  A plausible explanation (hypothesis, not measured) is that float32 runs
  on TFLite's optimised XNNPACK path while INT8 on x86 adds overhead; the
  ranking could reverse on ARM CPUs with integer instructions.
- **Anything clinical.** One dataset, one seed, one split. These models must
  not be used for diagnosis.

## Methodology

- **Lesion-level split, not image-level.** HAM10000 contains multiple images
  of the same lesion. `data.split_by_lesion()` uses `GroupShuffleSplit`
  (seed 42, 70/15/15) with an assertion that no lesion appears in two splits.
  Splitting by image leaks near-duplicates across splits and inflates
  accuracy.
- **Class imbalance.** Inverse-frequency class weights, with an extra boost
  for `mel` (×2.0) and `akiec` (×1.5) in the final baseline. Evaluation
  always reports macro-F1 and per-class F1 alongside accuracy.
- **Baseline.** MobileNetV2 (ImageNet weights, 224×224), global average
  pooling, dropout 0.3, 7-way softmax. Head trained for 10 epochs at lr 1e-3,
  then backbone unfrozen from layer 60 at lr 1e-5 for up to 25 epochs with
  early stopping.
- **Compression variants.** TFLite PTQ (dynamic-range, and full-INT8
  calibrated on 100 training images); tfmot unstructured magnitude pruning
  at 30/50/70%, each also converted to full-INT8; QAT via
  `tfmot.quantization.keras.quantize_model`.
- **Real latency.** Every `.tflite` file is benchmarked with
  `tf.lite.Interpreter` on one thread, not estimated from FLOPs. The float32
  TFLite baseline reproduced the Keras evaluation exactly, so conversion
  itself did not change the model.
- **QAT-compatible model construction.** `model.py` builds directly from
  `base.input` / `base.output` instead of nesting MobileNetV2 as a layer,
  which otherwise breaks `quantize_model()`.

## Setup

Tested on Google Colab (GPU for training, CPU for benchmarking).

1. Clone the repo into your Colab working directory.
2. Download a Kaggle API token (kaggle.com → Account → Create New API Token).
3. In a Colab cell:
   ```python
   from google.colab import files
   files.upload()  # upload kaggle.json
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !pip install -q kaggle
   !kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/HAM10000 --unzip
   ```
4. `pip install -r requirements.txt`
5. Set `TF_USE_LEGACY_KERAS=1` before importing TensorFlow (tfmot requires Keras 2):
   ```python
   import os
   os.environ["TF_USE_LEGACY_KERAS"] = "1"
   ```
6. Confirm `MOBILENET_WEIGHTS = "imagenet"` in `src/config.py` (the default).

## Running the pipeline

```bash
python -m src.train_baseline   # trains MobileNetV2 -> models/baseline_final.keras
                               #   + reports/baseline_eval.json (per-class F1 etc.)
python -m src.compress         # builds every compressed variant in models/*.tflite:
                               #   ptq_dynamic, ptq_int8, pruned_{30,50,70},
                               #   pruned_{..}_int8, qat_int8
python -m src.benchmark        # benchmarks every .tflite: accuracy, per-class F1,
                               #   size, CPU latency -> reports/pareto_results.csv
```

A quick end-to-end check on synthetic data (no dataset download needed):

```bash
python scripts/make_synthetic_data.py
python scripts/smoke_test.py
```

This only verifies the pipeline runs. Its accuracy numbers are meaningless.

## Repo layout

```
src/
  config.py          # paths, hyperparameters, class names
  data.py            # lesion-level split, tf.data pipeline, class weights,
                     #   representative dataset for PTQ
  model.py           # MobileNetV2 transfer-learning architecture
  train_baseline.py
  compress.py        # pruning / PTQ / QAT / combined
  evaluate.py        # per-class metrics, confusion matrix, critical-class F1
  benchmark.py       # size + latency + accuracy sweep across .tflite variants
scripts/
  make_synthetic_data.py   # synthetic data for the pipeline check
  smoke_test.py            # fast end-to-end pipeline check
literature/
  lit-scan.md        # literature review and novelty argument
reports/             # technical report, eval reports, results CSV
models/              # trained checkpoints and .tflite files (generated)
```

## Dataset

HAM10000: 10,015 dermoscopic images across 7 diagnostic classes, licensed
CC BY-NC 4.0.

> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large
> collection of multi-source dermatoscopic images of common pigmented skin
> lesions. *Scientific Data* 5, 180161 (2018).

## License

Code: MIT. The dataset is subject to its own CC BY-NC 4.0 license.
