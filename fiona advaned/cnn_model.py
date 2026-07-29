"""
CNN feature-based anomaly detector.

Idea: instead of comparing raw pixels (like PCA/autoencoder do), extract
DEEP features from a CNN pretrained on ImageNet (ResNet18). Those features
already encode textures, edges and shapes learned from millions of photos,
so they carry far more useful information than raw pixel values. PCA is
then fit on those deep features (same reconstruction-error idea as before,
just in a much richer feature space instead of pixel space).

This is a simplified version of the family of methods (PaDiM, PatchCore,
SPADE) that currently perform best on the MVTec-AD benchmark.

Requires: torch, torchvision (pip install torch torchvision).
The first run needs internet access to download the pretrained ResNet18
weights (~45MB, one-time - cached locally afterwards).
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.decomposition import PCA
import joblib

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_feature_extractor(device="cpu", pretrained=True):
    """ResNet18 with its final classification layer removed - output is a
    512-d feature vector per image (after global average pooling)."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = models.resnet18(weights=weights)
    resnet.fc = nn.Identity()
    resnet.eval()
    return resnet.to(device)


def extract_features(paths, device="cpu", batch_size=16, pretrained=True):
    """Loads images from disk paths, runs them through the CNN, returns an
    (N, 512) array of deep feature vectors."""
    extractor = build_feature_extractor(device, pretrained=pretrained)
    features = []
    with torch.no_grad():
        batch = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            batch.append(_TRANSFORM(img))
            if len(batch) == batch_size:
                feats = extractor(torch.stack(batch).to(device))
                features.append(feats.cpu().numpy())
                batch = []
        if batch:
            feats = extractor(torch.stack(batch).to(device))
            features.append(feats.cpu().numpy())
    return np.vstack(features)


def load_thumbnail(path, size=(128, 128)):
    """Small RGB image just for display in visualize.py (not used for scoring)."""
    img = Image.open(path).convert("RGB").resize(size)
    return np.asarray(img, dtype=np.float32) / 255.0


class CNNFeatureAnomalyDetector:
    """PCA over deep CNN features (not raw pixels). Same reconstruction-error
    concept as PCAAnomalyDetector, just applied in feature space."""

    def __init__(self, variance=0.90):
        self.pca = PCA(n_components=variance, svd_solver="full")

    def fit(self, X):
        self.pca.fit(X)
        return self

    def reconstruction_error(self, X):
        X_proj = self.pca.transform(X)
        X_rec = self.pca.inverse_transform(X_proj)
        errors = np.mean((X - X_rec) ** 2, axis=1)
        return errors, X_rec

    def save(self, path):
        joblib.dump(self.pca, path)

    def load(self, path):
        self.pca = joblib.load(path)
        return self
