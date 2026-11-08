"""
Runs the full pipeline (train -> compress -> benchmark) on the tiny synthetic
dataset with epoch counts slashed to 1, purely to catch bugs before you point
this at the real HAM10000 data. Not a real training run -- accuracy numbers
from this are meaningless.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HAM10000_DIR"] = os.path.join(PROJECT_ROOT, "data", "HAM10000_synthetic")
sys.path.insert(0, PROJECT_ROOT)

from src import config

# Slash everything down for a fast smoke test.
config.BASELINE_EPOCHS_FROZEN = 1
config.BASELINE_EPOCHS_FINETUNE = 1
config.PRUNE_EPOCHS = 1
config.QAT_EPOCHS = 1
config.PRUNING_TARGET_SPARSITY = [0.5]
config.BENCHMARK_NUM_RUNS = 5
config.BATCH_SIZE = 4
config.FINETUNE_UNFREEZE_FROM = 140  # unfreeze very few layers, keep it fast
config.MOBILENET_WEIGHTS = None  # skip the ImageNet download for a fast offline check;
                                  # real runs must use "imagenet"

from src import train_baseline, compress, benchmark

print("\n===== SMOKE TEST: train_baseline =====")
train_baseline.main()

print("\n===== SMOKE TEST: compress =====")
compress.main()

print("\n===== SMOKE TEST: benchmark =====")
benchmark.main()

print("\n===== SMOKE TEST PASSED: pipeline ran end-to-end with no errors =====")
