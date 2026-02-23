from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Standard directories
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Ensure results/models exist
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)