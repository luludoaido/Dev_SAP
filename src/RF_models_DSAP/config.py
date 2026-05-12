from pathlib import Path

# ── File Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(".")
EXPRESSION_FILE = BASE_DIR / "data" / "TCGA-STAD_gene_expression_cpm.csv"
SUBTYPE_FILE = BASE_DIR / "data" / "TCGA-STAD_subtypes.csv"

# ── Train/Test Split ───────────────────────────────────────────────
TRAIN_SIZE = 0.7
TRAIN_TEST_RANDOM_STATE = 1

# ── Model ──────────────────────────────────────────────────────────
MODEL_RANDOM_STATE = 42

# ── Feature Selection ──────────────────────────────────────────────
MIN_NON_ZERO_FRACTION = 0.1
VARIANCE_THRESHOLD = 0.01
TOP_FEATURE_COUNT = 20
TOP_IMPORTANCE_COUNT = 10

# ── Hyperparameter Grid ────────────────────────────────────────────
RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [None, 20, 50, 100],
    "max_features": ["sqrt"],
    "min_samples_leaf": [1, 2, 5, 10],
}

# ── Final Model Parameters ─────────────────────────────────────────
FINAL_MULTICLASS_PARAMS = {
    "n_estimators": 300,
    "bootstrap": True,
    "min_samples_leaf": 2,
    "max_depth": None,
    "max_features": "sqrt",
    "criterion": "gini",
    "random_state": MODEL_RANDOM_STATE,
}

FINAL_BINARY_PARAMS = {
    "n_estimators": 100,
    "bootstrap": True,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 10,
    "criterion": "gini",
    "random_state": MODEL_RANDOM_STATE,
}
