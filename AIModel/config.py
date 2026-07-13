"""
Central configuration for the GSR + HR stress detection pipeline.
Adjust the paths/hyperparameters here before running main.py.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Point this at wherever your cleaned_data_*.csv files actually live.
# discover_volunteer_files() searches recursively, so this can be the
# top-level project folder OR the cleaned_data/ folder itself.
DATA_ROOT = "../cleaned_data"
OUTPUT_DIR = "outputs"

# ---------------------------------------------------------------------------
# Resistance -> Conductance fix
# ---------------------------------------------------------------------------
# Volunteers in this numeric range (non-"V" prefixed files) were recorded as
# resistance but mislabeled as conductance. conductance = 1 / resistance.
RESISTANCE_ID_MIN = 35
RESISTANCE_ID_MAX = 144

# ---------------------------------------------------------------------------
# Volunteers to exclude entirely (known bad / incomplete data)
# ---------------------------------------------------------------------------
EXCLUDED_VOLUNTEERS = {
    70, 102, 121, 153, 154, 155,           # sensor disconnect / incomplete session
    110, 112, 113, 115, 116, 120, 124,     # RR_Interval_ms ~99% missing
    135, 136, 137, 143, 144, 81, 158,      # RR_Interval_ms ~99% missing
}

# ---------------------------------------------------------------------------
# Protocol stage labels (must match the Stage column exactly)
# ---------------------------------------------------------------------------
STAGE_CALIBRATION = "Calibration"
STAGE_RELAXED = "Normal"
STAGE_STRESSED = "Stress"
STAGE_RELAXATION = "Relaxation"

# Stages used to compute each volunteer's own baseline mean/std.
# Using baseline-only stats (not the whole session) avoids diluting the
# stress signal when we normalize.
BASELINE_STAGES = {STAGE_CALIBRATION, STAGE_RELAXED}

# Stages ignored entirely for windowing/training
IGNORED_STAGES = {STAGE_CALIBRATION, STAGE_RELAXATION}

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
COL_TIME = "Time_sec"
COL_COND = "Conductance_microS"
COL_STAGE = "Stage"
COL_RR = "RR_Interval_ms"

# ---------------------------------------------------------------------------
# Windowing (1 sample/sec assumed)
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 1
WINDOW_SEC = 30
STEP_SEC = 15
WINDOW_SIZE = WINDOW_SEC * SAMPLE_RATE_HZ   # 30 samples
STEP_SIZE = STEP_SEC * SAMPLE_RATE_HZ       # 15 samples

# Sequence channels fed to the CNN+LSTM: GSR(z), RR(z), GSR(z) first-difference
N_CHANNELS = 3

# Window-level engineered scalar features fed to a small MLP branch:
# [rr_rmssd, rr_sdnn, gsr_slope, gsr_range, gsr_peak_count]
N_SCALAR_FEATURES = 5

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
TEST_FRACTION = 0.20      # fraction of volunteers held out for final test
VAL_FRACTION = 0.20       # fraction of REMAINING (train) volunteers used for val
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4
EARLY_STOP_PATIENCE = 10

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_stress.pt")
GLOBAL_MEAN_PATH = os.path.join(OUTPUT_DIR, "global_mean.npy")
GLOBAL_STD_PATH = os.path.join(OUTPUT_DIR, "global_std.npy")
GLOBAL_FEAT_MEAN_PATH = os.path.join(OUTPUT_DIR, "global_feat_mean.npy")
GLOBAL_FEAT_STD_PATH = os.path.join(OUTPUT_DIR, "global_feat_std.npy")
TRAINING_CURVES_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")