"""
Smart Examiner — Centralized Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Disable PaddlePaddle 3.0 PIR engine and oneDNN which are causing C++ crashes
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle_trocr")  # "paddle_trocr", "gemma", or "qwen"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMMA_MODEL_NAME = "gemini-3.5-flash-lite"  # Fast, cheap, vision-capable
MATCHING_MODEL_NAME = "gemini-3.5-flash-lite"
BLUEPRINT_MODEL_NAME = "gemini-3.5-flash-lite"  # Model for blueprint/allocation inference

# Reproducibility metadata & versioning
MATCHING_PROMPT_VERSION = "5.0"
BLUEPRINT_VERSION = "3.0"
SCORING_ENGINE_VERSION = "2.0"
GRADING_POLICY_VERSION = "2.0"
GRADING_INCREMENT = 0.5  # Minimum mark step (0.5 = half marks allowed)

# ---------------------------------------------------------------------------
# Default Grading Policy (per requirement type)
# Maps evidence_state → fraction of max_marks
# Teacher-defined partial-credit bands override these defaults.
# ---------------------------------------------------------------------------
DEFAULT_GRADING_POLICY = {
    "identification": {
        "present": 1.0, "partial": 0.5, "absent": 0.0,
        "incorrect": 0.0, "contradicted": 0.0, "uncertain": 0.0,
    },
    "explanation": {
        "present": 1.0, "partial": 0.5, "absent": 0.0,
        "incorrect": 0.0, "contradicted": 0.0, "uncertain": 0.0,
    },
    "definition": {
        "present": 1.0, "partial": 0.5, "absent": 0.0,
        "incorrect": 0.0, "contradicted": 0.0, "uncertain": 0.0,
    },
    "example": {
        "present": 1.0, "partial": 0.5, "absent": 0.0,
        "incorrect": 0.0, "contradicted": 0.0, "uncertain": 0.0,
    },
    "mandatory": {
        "present": 1.0, "partial": 0.5, "absent": 0.0,
        "incorrect": 0.0, "contradicted": 0.0, "uncertain": 0.0,
    },
    "comparison": {
        "present": 1.0, "partial": 0.5, "absent": 0.0,
        "incorrect": 0.0, "contradicted": 0.0, "uncertain": 0.0,
    },
    # choose_any_n is scored differently (by concept counting in scoring.py)
    # Other/unknown types fall back to "default"
    "default": {
        "present": 1.0, "partial": 0.5, "absent": 0.0,
        "incorrect": 0.0, "contradicted": 0.0, "uncertain": 0.0,
    },
}

# Canonical evidence states (authoritative list)
EVIDENCE_STATES = {"present", "partial", "absent", "incorrect", "contradicted", "uncertain"}

QWEN_MODEL_NAME = "qwen2.5vl:3b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "trocr-large-handwritten"
DEBUG_OUTPUT_DIR = BASE_DIR / "debug_output"
TEST_IMAGES_DIR = BASE_DIR / "tests" / "sample_images"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
MAX_IMAGE_DIMENSION = 4000          # Resize if any side exceeds this (pixels)
MIN_IMAGE_DIMENSION = 100           # Reject images smaller than this
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".pdf"}
MAX_FILE_SIZE_MB = 20

# ---------------------------------------------------------------------------
# Detection (PaddleOCR)
# ---------------------------------------------------------------------------
DET_MODEL_LANG = "en"
DET_USE_ANGLE_CLS = False           # We handle rotation in preprocessing
DET_USE_GPU = False                 # CPU-only for now

# ---------------------------------------------------------------------------
# Recognition (TrOCR)
# ---------------------------------------------------------------------------
TROCR_MODEL_NAME = str(MODEL_DIR)
TROCR_MAX_NEW_TOKENS = 256
TROCR_DEVICE = "cpu"                # Switch to "cuda" if GPU available

# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------
LINE_GROUP_THRESHOLD = 0.5          # Fraction of avg line height for grouping
MIN_CONFIDENCE = 0.10               # Discard lines below this confidence
LOW_CONF_THRESHOLD = 0.40           # Flag lines below this confidence
CROP_PADDING_RATIO = 0.15           # Padding as a fraction of line height

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000

# Available engines: "qwen", "gemma", "paddle_trocr", "kraken_trocr"
