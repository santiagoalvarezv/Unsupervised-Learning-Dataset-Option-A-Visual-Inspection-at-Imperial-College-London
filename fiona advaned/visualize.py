"""
Step 3: Generate the plots your submission needs:
  - score histogram (normal vs defective, with threshold line)
  - example grids for: true positives, false alarms, missed defects, correct normals
    each with original / reconstruction / error heatmap
    (for --method cnn, only the original image + score is shown - there's no
    pixel-level "reconstruction" concept in deep-feature space)

Usage:
    python visualize.py
    python visualize.py --category hazelnut
    python visualize.py --category bottle --method autoencoder
    python visualize.py --category bottle --method cnn
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless environments
import matplotlib.pyplot as plt

from config import RESULTS_DIR, CATEGORY


def to_image(vector, img_size, grayscale):
    if grayscale:
        return vector.reshape((img_size[1], img_size[0]))
    return vector.reshape((img_size[1], img_size[0], 3))


def plot_score_histogram(errors, labels, threshold, category, method):
    plt.figure(figsize=(7, 4))
    plt.hist(errors[labels == 0], bins=30, alpha=0.6, label="Normal (good)")
    plt.hist(errors[labels == 1], bins=30, alpha=0.6, label="Defective")
    plt.axvline(threshold, color="black", linestyle="--", label="Threshold")
    plt.xlabel("Anomaly score (reconstruction error)")
    plt.ylabel("Count")
    plt.title(f"Anomaly scores - {category} ({method})")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{category}_{method}_score_histogram.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def show_examples(X_test, X_rec, errors, labels, predictions, category, method, img_size, grayscale,
                   has_reconstruction, n_examples=3):
    groups = {
        "true_positive": (labels == 1) & (predictions == 1),   # defect correctly flagged
        "false_alarm": (labels == 0) & (predictions == 1),     # normal flagged as defective
        "missed_defect": (labels == 1) & (predictions == 0),   # defective not flagged
        "correct_normal": (labels == 0) & (predictions == 0),  # normal correctly passed
    }

    n_cols = 3 if has_reconstruction else 1

    for name, mask in groups.items():
        idx = np.where(mask)[0][:n_examples]
        if len(idx) == 0:
            print(f"(no examples for '{name}' - none in this test set)")
            continue

        fig, axes = plt.subplots(len(idx), n_cols, figsize=(3 * n_cols, 3 * len(idx)))
        if len(idx) == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif len(idx) == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        for row, i in enumerate(idx):
            original = to_image(X_test[i], img_size, grayscale)

            if has_reconstruction:
                recon = to_image(X_rec[i], img_size, grayscale)
                heatmap = np.abs(original - recon)
                if heatmap.ndim == 3:
                    heatmap = heatmap.mean(axis=-1)

                axes[row, 0].imshow(original, cmap="gray" if grayscale else None)
                axes[row, 0].set_title("Original")
                axes[row, 1].imshow(recon, cmap="gray" if grayscale else None)
                axes[row, 1].set_title("Reconstruction")
                axes[row, 2].imshow(heatmap, cmap="hot")
                axes[row, 2].set_title(f"Error map (score={errors[i]:.4f})")
                for ax in axes[row]:
                    ax.axis("off")
            else:
                axes[row, 0].imshow(original)
                axes[row, 0].set_title(f"score={errors[i]:.4f}")
                axes[row, 0].axis("off")

        fig.suptitle(f"{category} ({method}): {name.replace('_', ' ')}")
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"{category}_{method}_{name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")



def plot_autoencoder_loss(category, method):
    """Plot train/validation MSE saved by the ConvAE training step."""
    if method != "autoencoder":
        return
    meta_path = os.path.join("models", f"{category}_{method}_meta.json")
    if not os.path.exists(meta_path):
        return
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    train_loss = meta.get("train_loss_history", [])
    validation_loss = meta.get("validation_loss_history", [])
    if not train_loss or not validation_loss:
        return

    epochs = np.arange(1, len(train_loss) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="Train MSE")
    plt.plot(epochs, validation_loss, label="Normal validation MSE")
    best_epoch = int(meta.get("best_epoch", 0))
    if best_epoch > 0:
        plt.axvline(best_epoch, linestyle="--", label=f"Best epoch={best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(f"ConvAE training curve - {category}")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{category}_{method}_loss_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

def main(category, method):
    data = np.load(os.path.join(RESULTS_DIR, f"{category}_{method}_eval.npz"), allow_pickle=True)
    img_size = tuple(int(x) for x in data["img_size"])
    grayscale = bool(data["grayscale"])
    has_reconstruction = bool(data["has_reconstruction"]) if "has_reconstruction" in data else True
    plot_score_histogram(data["errors"], data["labels"], float(data["threshold"]), category, method)
    plot_autoencoder_loss(category, method)
    show_examples(data["X_test"], data["X_rec"], data["errors"], data["labels"], data["predictions"],
                  category, method, img_size, grayscale, has_reconstruction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="Override config.CATEGORY")
    parser.add_argument("--method", choices=["pca", "autoencoder", "cnn"], default="pca")
    args = parser.parse_args()
    main(args.category or CATEGORY, args.method)
