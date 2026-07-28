"""
Central configuration for the anomaly detection project.
Edit these values instead of hardcoding paths/params in every script.
"""

# Root folder where your dataset lives (see README.md for expected structure)
DATA_ROOT = "data"

# Which product category to run. Options (per the project guide):
# "bottle", "hazelnut", "metal_nut", "transistor", "wood"
CATEGORY = "bottle"

# All images are resized to this size before modelling
IMG_SIZE = (64, 64)   # (width, height) — increase if you have time/compute budget

# Grayscale is simpler and usually enough for texture/shape defects.
# Set to False to keep RGB (3x more features -> slower PCA, may need more variance).
GRAYSCALE = True

# Fraction of variance PCA must keep (e.g. 0.90 = keep 90% of the information)
PCA_VARIANCE = 0.90

# Percentile of the TRAIN reconstruction error used as the alarm threshold.
# 99 means: only the top 1% "hardest" normal training images would be false alarms.
THRESHOLD_PERCENTILE = 99

MODEL_DIR = "models"
RESULTS_DIR = "results"
