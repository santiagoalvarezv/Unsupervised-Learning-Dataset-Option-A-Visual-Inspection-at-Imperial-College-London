"""
Autoencoder anomaly detector (same idea as PCA, but a nonlinear model).

Idea: train a small neural network to reconstruct normal images through a
narrow "bottleneck" layer. It's forced to compress and rebuild what it sees,
so it becomes good at reconstructing normal patterns specifically. A
defective image won't reconstruct as well, so its reconstruction error is
the anomaly score - same concept as PCA, just a nonlinear model instead of
a linear one.

Optionally works as a DENOISING autoencoder: if noise_std > 0, Gaussian
noise is added to the input during TRAINING only, while the target stays
the original clean image. This forces the network to learn the underlying
normal texture/structure instead of just memorising exact pixel values,
which can widen the gap between normal and defective reconstruction error.
Evaluation always uses clean (noise-free) images - noise is a training
trick only.

Uses scikit-learn's MLPRegressor (no extra dependencies like PyTorch needed).
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib


class AEAnomalyDetector:
    def __init__(self, bottleneck=32, hidden=128, max_iter=300, random_state=0, noise_std=0.0):
        # Symmetric encoder/decoder: input -> hidden -> bottleneck -> hidden -> output
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden, bottleneck, hidden),
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=10,
        )
        self.noise_std = noise_std
        self.random_state = random_state

    def fit(self, X):
        # Denoising variant: feed a noisy version of X as input, but still
        # target the original clean X. Plain autoencoder if noise_std == 0.
        if self.noise_std > 0:
            rng = np.random.RandomState(self.random_state)
            X_input = X + rng.normal(0, self.noise_std, X.shape)
            X_input = np.clip(X_input, 0.0, 1.0)  # pixel values stay valid
        else:
            X_input = X
        self.model.fit(X_input, X)
        return self

    def reconstruction_error(self, X):
        """Returns (per-image MSE error, reconstructed images) for an array X.
        Always uses clean X - noise is only applied during training."""
        X_rec = self.model.predict(X)
        errors = np.mean((X - X_rec) ** 2, axis=1)
        return errors, X_rec

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self

