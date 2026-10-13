"""Central configuration for paths and hyperparameters."""
import os

# --- Paths ---
BASE_DIR    = r"D:\Vijay\MovieLens"
DATA_DIR    = os.path.join(BASE_DIR, "ml-25m")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
MODEL_DIR   = os.path.join(OUT_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Data filtering ---
MIN_USER_RATINGS   = 20
MIN_MOVIE_RATINGS  = 5
TEST_YEAR_FRACTION = 0.2

# --- Evaluation ---
TOP_K          = 10
LIKE_THRESHOLD = 3.5
ACC_TOLERANCE  = 0.5

# --- Hybrid ---
HYBRID_ALPHA   = 0.85

# --- Reproducibility ---
RANDOM_STATE   = 42