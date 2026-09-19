"""
Central config for the HAM10000 edge-compression project.
Edit DATA_DIR to point at wherever you've put the dataset.
"""
import os

# ---- Paths ----
# Expected layout (this is exactly how HAM10000 ships from Kaggle / Harvard Dataverse):
#   DATA_DIR/HAM10000_metadata.csv
#   DATA_DIR/HAM10000_images_part_1/*.jpg
#   DATA_DIR/HAM10000_images_part_2/*.jpg
DATA_DIR = os.environ.get("HAM10000_DIR", "/home/claude/edge-ai-skin-lesion/data/HAM10000")
METADATA_CSV = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
IMAGE_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]

OUTPUT_DIR = "/home/claude/edge-ai-skin-lesion/models"
REPORTS_DIR = "/home/claude/edge-ai-skin-lesion/reports"

# ---- Classes ----
# HAM10000's 7 diagnostic categories (dx column), in a fixed order used everywhere.
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
NUM_CLASSES = len(CLASS_NAMES)

# Clinically-critical classes to track separately in every report (malignant / pre-malignant,
# and rare -- these are the ones a compressed model is most likely to quietly get worse at).
CRITICAL_CLASSES = ["mel", "akiec"]

# ---- Image / training params ----
# "imagenet" for real training. Set to None only for offline smoke-testing the code
# (e.g. this sandbox, which has no internet access to fetch pretrained weights).
MOBILENET_WEIGHTS = "imagenet"
IMG_SIZE = 224          # MobileNetV2 default input size
BATCH_SIZE = 32
SEED = 42

BASELINE_EPOCHS_FROZEN = 10     # phase 1: frozen backbone, train head only
BASELINE_EPOCHS_FINETUNE = 15   # phase 2: unfreeze top layers, fine-tune
FINETUNE_UNFREEZE_FROM = 100    # unfreeze MobileNetV2 layers from this index onward

# ---- Compression params ----
PRUNING_TARGET_SPARSITY = [0.3, 0.5, 0.7]   # multiple pruning levels for the Pareto sweep
PRUNE_EPOCHS = 8
QAT_EPOCHS = 5

# ---- Benchmarking ----
BENCHMARK_NUM_RUNS = 100        # inferences per model when measuring latency
BENCHMARK_THREADS = 1           # pinned to 1 thread to simulate a low-power edge device
