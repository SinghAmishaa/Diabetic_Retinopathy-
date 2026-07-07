"""
Project Configuration
---------------------
This file stores all configurable settings used across the project.
"""

from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent



# Dataset paths
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_CSV = DATASET_DIR / "train.csv"
TRAIN_IMAGES = DATASET_DIR / "train_images"
PROCESSED_IMAGES = DATASET_DIR / "processed_images"   # <-- Add this 
TRAIN_SPLIT = DATASET_DIR / "train_split.csv"
VAL_SPLIT = DATASET_DIR / "val_split.csv"

# Output folders
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Image settings
IMAGE_SIZE = (224, 224)

# Training settings
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4

# Number of classes
NUM_CLASSES = 5