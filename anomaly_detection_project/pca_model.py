"""
PCA anomaly detector.

Idea: fit PCA on normal images only, so it learns the "normal" subspace.
A defective image won't project/reconstruct well through that subspace,
so its reconstruction error (original vs. rebuilt image) will be higher.
That error IS the anomaly score.
"""

import numpy as np
from sklearn.decomposition import PCA
import joblib


class PCAAnomalyDetector:
    def __init__(self, variance=0.90):
        self.pca = PCA(n_components=variance, svd_solver="full")

    def fit(self, X):
        self.pca.fit(X)
        return self

    def reconstruction_error(self, X):
        """Returns (per-image MSE error, reconstructed images) for an array X."""
        X_proj = self.pca.transform(X)
        X_rec = self.pca.inverse_transform(X_proj)
        errors = np.mean((X - X_rec) ** 2, axis=1)
        return errors, X_rec

    def save(self, path):
        joblib.dump(self.pca, path)

    def load(self, path):
        self.pca = joblib.load(path)
        return self
