# Literature Scan — HAM10000 / Lightweight CNN / Compression (Sept 2026)

Verified via arXiv/Springer/MDPI/PubMed. 6 confirmed, 2 flagged low-confidence (included for awareness only, not to be cited without re-checking).

## Directly relevant papers

1. **Chaturvedi, Gupta, Prasad (2019)** — "Skin Lesion Analyser: An Efficient Seven-Way Multi-Class Skin Cancer Classification Using MobileNet." *Advances in Intelligent Systems and Computing* (Springer) / arXiv:1907.03220.
   MobileNet(v1) transfer learning on HAM10000. Acc 83.15%, weighted F1 83%. https://arxiv.org/pdf/1907.03220

2. **Baig, Abbas, Almakki, Ibrahim, AlSuwaidan, Ahmed (2023)** — "Light-Dermo: A Lightweight Pretrained CNN for the Diagnosis of Multiclass Skin Lesions." *Diagnostics* 13(3):385 (MDPI).
   ShuffleNet + SE blocks + depthwise-separable convs, trained on HAM10000+ISIC2019+ISIC2020. Acc 99.14%, F1 98.1%. https://doi.org/10.3390/diagnostics13030385

3. **Al Mamun et al. (2025)** — "Optimizing Deep Learning for Skin Cancer Classification: A Computationally Efficient CNN with Minimal Accuracy Trade-Off." arXiv:2505.21597.
   Custom lightweight CNN vs ResNet50 on HAM10000: 96.7% param reduction, 99.2% FLOPs reduction. Acc 87.05% (custom) vs 89.08% (ResNet50). https://arxiv.org/abs/2505.21597

4. **Wang et al. (2025)** — "Quantization-Aware Neuromorphic Architecture for Efficient Skin Disease Classification on Resource-Constrained Devices." arXiv:2507.15958.
   Ghost modules + SE/ECA attention, QAT → spiking net on BrainChip Akida neuromorphic hardware. HAM10000: 91.6% top-1, 82.4% macro-F1, 1.5ms/image on Akida. https://arxiv.org/html/2507.15958v1

5. **Winata, Andryani, Gunawan, Lumban Gaol (2026)** — "Efficiency Analysis of AI Model Compression for Edge Teledermatology." Studies in Computational Intelligence vol. 1227 (Springer).
   Compares compression strategies on HAM10000/ISIC/BCN20000 for edge teledermatology. Knowledge distillation won (+2.65% over baseline), fastest mobile inference 109ms. Exact baseline architecture/full metrics not confirmable from abstract alone. https://link.springer.com/chapter/10.1007/978-3-032-01133-6_9

6. **Paxton, Aslansefat, Thakker, Papadopoulos, Maslekar (2025)** — "Enhancing Fairness in Skin Lesion Classification for Medical Diagnosis Using Prune Learning." arXiv:2509.00745.
   Skewness-based structured pruning (channels/patches/attention heads) on VGG11 and ViT-B16, ISIC2019 + Fitzpatrick skin-type labels — pruning for bias reduction, not just compression. VGG11: 79% acc / 65% F1 post-pruning. https://arxiv.org/abs/2509.00745

## Low-confidence / unverified (do not cite without re-checking)

- Lavaei et al. (2025), arXiv:2512.17515 — PTQ/QAT on unspecified "medical imaging datasets"; could not confirm HAM10000 is included or any numeric results.
- ScienceDirect (2024) "Skin cancer detection using lightweight model souping and ensembling knowledge distillation for memory-constrained devices" — on-topic title, but paywalled; authors/metrics not confirmed.

## Positioning of this project

Training MobileNetV2 on HAM10000 and applying TFLite PTQ is not new on its own; papers 1, 4 and 5 cover closely related ground. The project therefore focuses on three things that are absent or only partly covered in the papers above:

1. **Measured CPU latency on a reproducible setup.** Most papers report parameters or FLOPs as a proxy for efficiency. Paper 4 reports measured latency, but on specialised neuromorphic hardware (BrainChip Akida), and paper 5 reports a measured mobile latency (109 ms). This project measures single-thread CPU latency for every variant with the TFLite interpreter.
2. **Per-class F1 under compression.** HAM10000 is heavily imbalanced (~67% `nv`), so aggregate accuracy can hide a collapse on the rare, clinically important classes (`mel`, `akiec`). None of the papers above report per-class degradation across compression levels; for paper 5 this needs to be checked against the full text, since only the abstract was available.
3. **One backbone, one split, one benchmark for all techniques.** Pruning (30/50/70%), INT8 PTQ (dynamic-range and full-integer), pruning combined with INT8, and QAT are compared on the same MobileNetV2 baseline and the same lesion-level split.

Papers 1, 4 and 5 are cited as the closest prior work in the technical report's related-work section.
