"""Basic raw-pixel PCA anomaly detector."""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.decomposition import PCA


class PCAAnomalyDetector:
    def __init__(self, n_components: int = 64):
        self.requested_components = int(n_components)
        self.effective_components: int | None = None
        self.pca: PCA | None = None

    def fit(self, X: np.ndarray):
        if len(X) < 2:
            raise ValueError("PCA needs at least two normal fit images.")
        self.effective_components = min(
            self.requested_components,
            X.shape[0] - 1,
            X.shape[1],
        )
        if self.effective_components < 1:
            raise ValueError("No valid PCA component is available.")
        self.pca = PCA(
            n_components=self.effective_components,
            svd_solver="randomized",
            random_state=0,
        )
        self.pca.fit(X)
        return self

    def reconstruction_error(self, X: np.ndarray):
        if self.pca is None:
            raise RuntimeError("PCA model is not fitted or loaded.")
        X_rec = self.pca.inverse_transform(self.pca.transform(X)).astype(np.float32)
        errors = np.mean((X - X_rec) ** 2, axis=1)
        return errors.astype(np.float64), X_rec

    def save(self, path: str):
        if self.pca is None:
            raise RuntimeError("Cannot save an unfitted PCA model.")
        joblib.dump(
            {
                "pca": self.pca,
                "requested_components": self.requested_components,
                "effective_components": self.effective_components,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = joblib.load(path)
        # Also tolerate the old project format, which saved the PCA object alone.
        if isinstance(checkpoint, PCA):
            self.pca = checkpoint
            self.effective_components = int(checkpoint.n_components_)
            self.requested_components = self.effective_components
        else:
            self.pca = checkpoint["pca"]
            self.requested_components = int(checkpoint["requested_components"])
            self.effective_components = int(checkpoint["effective_components"])
        return self
