"""
Convolutional autoencoder anomaly detector (same idea as PCA, but a
nonlinear CNN model instead of a linear one, and convolutional instead of
fully-connected).

Idea: train a small convolutional encoder/decoder to reconstruct normal
images through a narrow "bottleneck" feature map. Convolutions preserve the
image's 2D spatial structure (unlike a fully-connected/MLP autoencoder,
which flattens everything into a 1D vector and loses that structure) - this
is the standard, more appropriate architecture for image data. A defective
image won't reconstruct as well through the bottleneck, so its
reconstruction error is the anomaly score - same concept as PCA, just a
convolutional nonlinear model instead of a linear one.

Optionally works as a DENOISING autoencoder: if noise_std > 0, Gaussian
noise is added to the input during TRAINING only, while the target stays
the original clean image. This forces the network to learn the underlying
normal texture/structure instead of just memorising exact pixel values.
Evaluation always uses clean (noise-free) images - noise is a training
trick only.

Requires: torch (pip install torch).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvAE(nn.Module):
    """Symmetric conv encoder/decoder. `hidden` and `bottleneck` are channel
    counts (not flat vector sizes, unlike the old MLP version)."""

    def __init__(self, in_channels, img_size, hidden=128, bottleneck=32):
        super().__init__()
        h1, h2 = max(hidden // 4, 4), max(hidden // 2, 8)
        self.img_size = img_size  # (W, H)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, h1, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(h1, h2, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(h2, bottleneck, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck, h2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h2, h1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(h1, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # pixel values are in [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        # Guarantees exact (H, W) match regardless of stride/rounding, so
        # this works for any --img-size, not just powers of 2.
        target_hw = (self.img_size[1], self.img_size[0])
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


class AEAnomalyDetector:
    def __init__(self, img_size=(64, 64), grayscale=True, bottleneck=32, hidden=128,
                 noise_std=0.0, epochs=60, lr=1e-3, batch_size=16, random_state=0, device="cpu"):
        self.img_size = img_size
        self.grayscale = grayscale
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.noise_std = noise_std
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self.in_channels = 1 if grayscale else 3
        self.net = _ConvAE(self.in_channels, img_size, hidden, bottleneck).to(device)

    def _to_tensor(self, X):
        """(N, D) flat pixel vectors -> (N, C, H, W) tensor, matching how
        data_loader.py flattens images (row-major, channel-last before flatten)."""
        n = X.shape[0]
        w, h = self.img_size
        if self.grayscale:
            imgs = X.reshape(n, h, w, 1)
        else:
            imgs = X.reshape(n, h, w, 3)
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # -> (N, C, H, W)
        return torch.tensor(imgs, dtype=torch.float32)

    def fit(self, X):
        torch.manual_seed(self.random_state)
        X_clean = self._to_tensor(X).to(self.device)

        if self.noise_std > 0:
            rng = torch.Generator().manual_seed(self.random_state)
            noise = torch.randn(X_clean.shape, generator=rng) * self.noise_std
            X_input = torch.clamp(X_clean + noise.to(self.device), 0.0, 1.0)
        else:
            X_input = X_clean

        dataset = torch.utils.data.TensorDataset(X_input, X_clean)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            for batch_input, batch_target in loader:
                optimizer.zero_grad()
                output = self.net(batch_input)
                loss = loss_fn(output, batch_target)
                loss.backward()
                optimizer.step()
        return self

    def reconstruction_error(self, X):
        """Returns (per-image MSE error, flattened reconstructed images) for
        an array X. Always uses clean X - noise is only applied during training."""
        self.net.eval()
        X_tensor = self._to_tensor(X).to(self.device)
        with torch.no_grad():
            X_rec_tensor = self.net(X_tensor)
        errors = torch.mean((X_tensor - X_rec_tensor) ** 2, dim=[1, 2, 3]).cpu().numpy()
        # Flatten back to (N, D) in the same layout data_loader.py uses, so
        # visualize.py's existing reshape logic keeps working unchanged.
        X_rec = X_rec_tensor.cpu().numpy().transpose(0, 2, 3, 1).reshape(X.shape[0], -1)
        return errors, X_rec

    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "img_size": self.img_size,
            "grayscale": self.grayscale,
            "bottleneck": self.bottleneck,
            "hidden": self.hidden,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.img_size = checkpoint["img_size"]
        self.grayscale = checkpoint["grayscale"]
        self.bottleneck = checkpoint["bottleneck"]
        self.hidden = checkpoint["hidden"]
        self.in_channels = 1 if self.grayscale else 3
        self.net = _ConvAE(self.in_channels, self.img_size, self.hidden, self.bottleneck).to(self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()
        return self
