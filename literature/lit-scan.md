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

## Direct verdict: is this project novel?

**No, not in its plain form.** "Train MobileNetV2 on HAM10000, run TFLite PTQ, report accuracy drop" is already covered in spirit by papers 1, 4, and 5. That exact recipe alone is not a contribution — a reviewer (or an admissions reader who checks) would call it derivative.

**What's actually missing from the literature, and what this project will do instead:**

1. **No paper here measures real wall-clock CPU/edge-device latency.** All of them report FLOPs/params as a *proxy* for efficiency, or use exotic hardware (BrainChip Akida — not something anyone else can reproduce). None benchmark on a standard, reproducible target (a laptop CPU thread-limited to simulate a low-power device, or literally a Raspberry Pi).
2. **No paper reports per-class accuracy degradation under compression.** HAM10000 is heavily imbalanced (~67% `nv`/melanocytic nevi). Aggregate accuracy can look fine while the compressed model silently gets much worse at the rare, clinically important classes (`mel`, `akiec`). Nobody in this list checks that.
3. **No paper here runs a full combined ablation** (pruning + INT8 PTQ + QAT on the *same* backbone, same data splits) to produce a real accuracy-vs-size-vs-latency Pareto frontier. Each paper picks one technique.

**Locked differentiation angle for this project (all three, since they're one experiment pipeline anyway):**
- Real CPU latency benchmarking (TFLite runtime, thread-limited to 1 core as a low-power-device proxy; Raspberry Pi run is a stretch goal, not a requirement) instead of FLOPs-as-proxy.
- Per-class F1 degradation curve across compression levels (baseline → pruned → INT8 PTQ → QAT), highlighting `mel`/`akiec` specifically — the clinically-dangerous classes most likely to break silently.
- Full Pareto frontier across 2x/4x/8x/16x effective compression, not a single checkpoint.

Cite papers 1, 4, and 5 explicitly in the report's related-work section as the closest prior art, and state up front why this project's contribution is the latency+per-class-degradation angle they don't cover.
