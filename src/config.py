"""Configuration settings for the project."""

import os
from pathlib import Path

# Random seed for reproducibility
RANDOM_SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Target variable
TARGET_COLUMN = "Watch time(Minutes)"  # Will be determined after data exploration
