"""Basic convolutional autoencoder for image anomaly detection.

This replaces the original flattened-image MLPRegressor while keeping the same
high-level detector interface used by train.py/evaluate.py:
    fit(X_fit, X_validation)
    reconstruction_error(X)
    save(path)
    load(path)

The method is deliberately basic and comparable with raw-pixel PCA:
- same resized grayscale vectors
- convolutional encoder/decoder
- one latent vector
- plain pixel MSE
- Adam optimizer
- sigmoid output
- early stopping on normal validation MSE
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ConvAutoencoder(nn.Module):
    def __init__(self, channels: int, image_size: int, latent_dim: int):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("Image size must be divisible by 16, e.g. 64 or 128.")

        self.channels = channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.spatial_size = image_size // 16
        flattened_size = 64 * self.spatial_size * self.spatial_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(channels, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.to_latent = nn.Linear(flattened_size, latent_dim)

        self.from_latent = nn.Linear(latent_dim, flattened_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder_conv(x)
        latent = self.to_latent(features.flatten(start_dim=1))
        decoded = self.from_latent(latent)
        decoded = decoded.view(-1, 64, self.spatial_size, self.spatial_size)
        return self.decoder_conv(decoded)


@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int


class AEAnomalyDetector:
    def __init__(
        self,
        latent_dim: int = 64,
        img_size: tuple[int, int] = (128, 128),
        grayscale: bool = True,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        patience: int = 8,
        random_state: int = 42,
        device: str = "auto",
        # Backward-compatible aliases from the original MLP version.
        bottleneck: int | None = None,
        hidden: int | None = None,
        noise_std: float = 0.0,
        max_iter: int | None = None,
    ):
        del hidden  # no hidden-width flag is needed for this fixed basic ConvAE
        if bottleneck is not None:
            latent_dim = bottleneck
        if max_iter is not None:
            epochs = max_iter
        if noise_std != 0.0:
            raise ValueError(
                "This baseline is a plain ConvAE. Use noise_std=0 so the PCA comparison stays simple."
            )

        self.latent_dim = int(latent_dim)
        self.img_size = tuple(int(value) for value in img_size)
        self.grayscale = bool(grayscale)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.device_name = device
        self.device = self._resolve_device(device)
        self.model: ConvAutoencoder | None = None
        self.history = TrainingHistory([], [], 0)

        width, height = self.img_size
        if width != height or width % 16 != 0:
            raise ValueError("ConvAE requires a square image size divisible by 16.")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(device)

    @property
    def channels(self) -> int:
        return 1 if self.grayscale else 3

    def _set_seed(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _vectors_to_tensor(self, X: np.ndarray) -> torch.Tensor:
        width, height = self.img_size
        expected = width * height * self.channels
        if X.ndim != 2 or X.shape[1] != expected:
            raise ValueError(
                f"Expected flattened vectors with {expected} values, got shape {X.shape}."
            )
        if self.grayscale:
            images = X.reshape(-1, height, width)[:, None, :, :]
        else:
            images = X.reshape(-1, height, width, 3).transpose(0, 3, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(images)).float()

    def _tensor_to_vectors(self, tensor: torch.Tensor) -> np.ndarray:
        images = tensor.detach().cpu().numpy()
        if not self.grayscale:
            images = images.transpose(0, 2, 3, 1)
        return images.reshape(len(images), -1).astype(np.float32)

    def _build_model(self):
        self.model = ConvAutoencoder(self.channels, self.img_size[0], self.latent_dim).to(self.device)

    def fit(self, X: np.ndarray, X_validation: np.ndarray | None = None):
        if len(X) < 2:
            raise ValueError("ConvAE needs at least two normal fit images.")
        if X_validation is None or len(X_validation) < 1:
            raise ValueError("ConvAE needs an independent normal validation set.")

        self._set_seed()
        self._build_model()
        assert self.model is not None

        train_tensor = self._vectors_to_tensor(X)
        validation_tensor = self._vectors_to_tensor(X_validation)
        generator = torch.Generator().manual_seed(self.random_state)

        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=min(self.batch_size, len(train_tensor)),
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            TensorDataset(validation_tensor),
            batch_size=min(self.batch_size, len(validation_tensor)),
            shuffle=False,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        best_state = copy.deepcopy(self.model.state_dict())
        best_validation_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        train_losses: list[float] = []
        validation_losses: list[float] = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_sum = 0.0
            train_count = 0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = self.model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()
                train_sum += float(loss.item()) * len(batch)
                train_count += len(batch)

            self.model.eval()
            validation_sum = 0.0
            validation_count = 0
            with torch.no_grad():
                for (batch,) in validation_loader:
                    batch = batch.to(self.device)
                    reconstruction = self.model(batch)
                    loss = criterion(reconstruction, batch)
                    validation_sum += float(loss.item()) * len(batch)
                    validation_count += len(batch)

            train_loss = train_sum / train_count
            validation_loss = validation_sum / validation_count
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            print(
                f"  epoch {epoch:03d}/{self.epochs}: "
                f"train_mse={train_loss:.6f}, validation_mse={validation_loss:.6f}"
            )

            if validation_loss < best_validation_loss - 1e-7:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"  early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                    break

        self.model.load_state_dict(best_state)
        self.history = TrainingHistory(train_losses, validation_losses, best_epoch)
        return self

    def reconstruction_error(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("ConvAE model is not fitted or loaded.")
        tensor = self._vectors_to_tensor(X)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=min(self.batch_size, max(1, len(tensor))),
            shuffle=False,
        )

        self.model.eval()
        reconstructions: list[torch.Tensor] = []
        errors: list[torch.Tensor] = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstruction = self.model(batch)
                batch_errors = torch.mean((batch - reconstruction) ** 2, dim=(1, 2, 3))
                reconstructions.append(reconstruction.cpu())
                errors.append(batch_errors.cpu())

        X_rec = self._tensor_to_vectors(torch.cat(reconstructions, dim=0))
        score_array = torch.cat(errors, dim=0).numpy().astype(np.float64)
        return score_array, X_rec

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted ConvAE model.")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "latent_dim": self.latent_dim,
                "img_size": self.img_size,
                "grayscale": self.grayscale,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "patience": self.patience,
                "random_state": self.random_state,
                "history": {
                    "train_loss": self.history.train_loss,
                    "validation_loss": self.history.validation_loss,
                    "best_epoch": self.history.best_epoch,
                },
            },
            path,
        )

    def load(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.latent_dim = int(checkpoint["latent_dim"])
        self.img_size = tuple(checkpoint["img_size"])
        self.grayscale = bool(checkpoint["grayscale"])
        self.epochs = int(checkpoint["epochs"])
        self.batch_size = int(checkpoint["batch_size"])
        self.learning_rate = float(checkpoint["learning_rate"])
        self.patience = int(checkpoint["patience"])
        self.random_state = int(checkpoint["random_state"])
        self._build_model()
        assert self.model is not None
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        history = checkpoint.get("history", {})
        self.history = TrainingHistory(
            list(history.get("train_loss", [])),
            list(history.get("validation_loss", [])),
            int(history.get("best_epoch", 0)),
        )
        return self
