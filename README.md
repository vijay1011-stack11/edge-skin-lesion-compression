# Edge AI: Compression Trade-offs for Skin Lesion Classifiers

**Author:** Vijayabalaguroo S
**Report:** [Technical report (PDF)](reports/technical_report.pdf)

A systematic study of how model compression affects a MobileNetV2 skin lesion
classifier trained on HAM10000, measured on what matters for on-device
deployment: model size, real CPU inference latency, and per-class accuracy on
the clinically critical classes.

## Motivation

Compressing a MobileNetV2 classifier with quantization has been studied
before, but most published work reports aggregate accuracy and uses FLOPs as
a proxy for speed. That hides two things that matter on real devices:

1. **Per-class degradation.** HAM10000 is heavily imbalanced (`nv` is ~67% of
   the data). A compressed model can keep high overall accuracy while F1 on
   melanoma (`mel`) or actinic keratosis (`akiec`), the dangerous classes,
   quietly collapses.
2. **Measured latency.** Smaller does not automatically mean faster. Actual
   speed depends on the hardware's support for the compressed format.

This project measures both, across a full sweep of pruning, quantization and
combined configurations on one consistent backbone and dataset. The
literature review behind this framing is in `literature/lit-scan.md`.

## Results

| Configuration | Size (MB) | Accuracy | Macro-F1 | `mel` F1 | `akiec` F1 | CPU latency (ms) |
|---|---|---|---|---|---|---|
| Float32 baseline | 8.5 | 68.5% | [ ] | [ ] | [ ] | 6.9 |
| PTQ dynamic range | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| PTQ full INT8 | 2.6 | 71.3% | [ ] | [ ] | [ ] | 10.7 |
| Pruned 30% | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pruned 50% | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pruned 70% | [ ] | [ ] | 0.037 | [ ] | [ ] | [ ] |
| Pruned + INT8 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| QAT INT8 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

Full results: [`reports/pareto_results.csv`](reports/pareto_results.csv)

### Key findings

- **Full INT8 quantization cut model size 3.3× (8.5 MB → 2.6 MB) with no loss
  in accuracy.**
- **70% pruning collapsed the model** (macro-F1 0.037). Moderate sparsity
  is the practical limit for this backbone.
- **INT8 ran slower than float32 on an x86 CPU (10.7 ms vs 6.9 ms).** Size
  reduction does not translate into speed without native INT8 hardware
  support. Latency has to be measured on the target device, not assumed.

## Methodology

- **Lesion-level split, not image-level.** HAM10000 contains multiple images
  of the same lesion (`lesion_id` repeats). `data.split_by_lesion()` uses
  `GroupShuffleSplit` so no lesion appears in more than one of
  train/val/test, and this is asserted in code. Splitting by `image_id`
  leaks near-duplicates across splits and inflates reported accuracy, a
  common flaw in published HAM10000 results.
- **Class imbalance.** Training uses inverse-frequency class weights.
  Evaluation always reports macro-F1 and per-class F1 alongside accuracy, and
  tracks `mel` and `akiec` F1 separately as critical-class metrics. A
  compressed model that holds overall accuracy while `mel` F1 drops is
  counted as a failure.
- **Real latency.** Every `.tflite` variant is benchmarked for measured CPU
  inference time, not estimated from FLOPs.
- **QAT-compatible model construction.** `model.py` builds directly from
  `base.input` / `base.output` instead of nesting MobileNetV2 as a layer.
  Nesting breaks `tfmot.quantization.keras.quantize_model()` with a
  "Quantizing a keras Model inside another keras Model" error.

## Setup

Tested on Google Colab (free GPU, direct Kaggle access).

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
5. Confirm `MOBILENET_WEIGHTS = "imagenet"` in `src/config.py` (the default).

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
reports/             # technical report, eval reports, Pareto results CSV
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
