"""Central configuration for the image anomaly-detection project.

The project keeps three methods:
1. raw-pixel PCA
2. basic convolutional autoencoder
3. pretrained ResNet18 features + PCA (CNN method)
"""

DATA_ROOT = "data"
CATEGORY = "metal_nut"  # bottle, hazelnut, metal_nut, transistor, wood

# PCA and ConvAE use exactly the same resized grayscale images.
# CNN keeps its original internal 224x224 RGB preprocessing.
IMG_SIZE = (128, 128)  # (width, height); square and divisible by 16
GRAYSCALE = True

# Fair raw-image PCA vs ConvAE comparison.
PCA_COMPONENTS = 64
AE_LATENT_DIM = 64

# CNN+PCA keeps the original explained-variance setting.
CNN_PCA_VARIANCE = 0.90

# All methods use the same deterministic normal-only fit/validation split.
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
THRESHOLD_PERCENTILE = 95.0

# Basic ConvAE settings: plain MSE + Adam, no denoising/perceptual/patch loss.
AE_EPOCHS = 50
AE_BATCH_SIZE = 16
AE_LEARNING_RATE = 1e-3
AE_PATIENCE = 8

MODEL_DIR = "models"
RESULTS_DIR = "results"
