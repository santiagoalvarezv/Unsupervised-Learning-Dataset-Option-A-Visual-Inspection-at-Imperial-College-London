"""
Step 1: Train on normal images only.

Usage:
    python train.py                                  # uses config.py defaults (PCA)
    python train.py --category hazelnut              # override category
    python train.py --category metal_nut --img-size 128x128 --pca-variance 0.95
    python train.py --category bottle --method autoencoder
    python train.py --category bottle --method cnn   # deep-feature (ResNet18 + PCA)
"""

import os
import json
import argparse
import numpy as np

from config import DATA_ROOT, IMG_SIZE, GRAYSCALE, PCA_VARIANCE, THRESHOLD_PERCENTILE, MODEL_DIR, CATEGORY
from data_loader import load_train_normal
from pca_model import PCAAnomalyDetector
from autoencoder_model import AEAnomalyDetector


def parse_img_size(value):
    w, h = value.lower().split("x")
    return (int(w), int(h))


def build_detector(method, pca_variance, ae_bottleneck, ae_hidden, ae_noise_std):
    if method == "pca":
        return PCAAnomalyDetector(variance=pca_variance)
    elif method == "autoencoder":
        return AEAnomalyDetector(bottleneck=ae_bottleneck, hidden=ae_hidden, noise_std=ae_noise_std)
    elif method == "cnn":
        from cnn_model import CNNFeatureAnomalyDetector
        return CNNFeatureAnomalyDetector(variance=pca_variance)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pca', 'autoencoder', or 'cnn'.")


def main(category, img_size, method, pca_variance, threshold_percentile, ae_bottleneck, ae_hidden, ae_noise_std,
         cnn_pretrained):
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, paths = load_train_normal(DATA_ROOT, category, img_size, GRAYSCALE)

    if method == "cnn":
        # Deep features instead of raw pixel vectors - extracted straight from
        # the original image files (paths), ignoring the flattened pixel X_train.
        from cnn_model import extract_features
        print(f"[{category}/{method}] Extracting ResNet18 features for {len(paths)} training images "
              f"(first run downloads pretrained weights, ~45MB)...")
        X_train = extract_features(paths, pretrained=cnn_pretrained)

    print(f"[{category}/{method}] Loaded {len(X_train)} normal training images "
          f"({X_train.shape[1]} features each).")

    detector = build_detector(method, pca_variance, ae_bottleneck, ae_hidden, ae_noise_std)
    detector.fit(X_train)
    if method == "pca":
        print(f"[{category}/{method}] PCA kept {detector.pca.n_components_} components "
              f"to explain {pca_variance:.0%} of the variance.")
    elif method == "autoencoder":
        noise_note = f", denoising with noise_std={ae_noise_std}" if ae_noise_std > 0 else ""
        print(f"[{category}/{method}] Trained autoencoder (bottleneck={ae_bottleneck}, hidden={ae_hidden}{noise_note}).")
    else:
        print(f"[{category}/{method}] PCA (on ResNet18 features) kept {detector.pca.n_components_} components "
              f"to explain {pca_variance:.0%} of the variance.")

    # Threshold is chosen from the TRAINING data only (never from test data)
    train_errors, _ = detector.reconstruction_error(X_train)
    threshold = float(np.percentile(train_errors, threshold_percentile))

    model_path = os.path.join(MODEL_DIR, f"{category}_{method}.joblib")
    threshold_path = os.path.join(MODEL_DIR, f"{category}_{method}_threshold.txt")
    meta_path = os.path.join(MODEL_DIR, f"{category}_{method}_meta.json")

    detector.save(model_path)
    with open(threshold_path, "w") as f:
        f.write(str(threshold))
    # Saved so evaluate.py / visualize.py always match how THIS model was trained,
    # even if config.py or the CLI args change later.
    with open(meta_path, "w") as f:
        json.dump({"method": method, "img_size": list(img_size), "grayscale": GRAYSCALE,
                    "pca_variance": pca_variance, "ae_bottleneck": ae_bottleneck,
                    "ae_hidden": ae_hidden, "ae_noise_std": ae_noise_std,
                    "cnn_pretrained": cnn_pretrained}, f)

    print(f"[{category}/{method}] Saved model -> {model_path}")
    print(f"[{category}/{method}] Threshold ({threshold_percentile}th pct of train error) "
          f"= {threshold:.6f} -> {threshold_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="Override config.CATEGORY")
    parser.add_argument("--method", choices=["pca", "autoencoder", "cnn"], default="pca",
                         help="Which anomaly detector to train (default: pca)")
    parser.add_argument("--img-size", default=None, help="e.g. 128x128 (overrides config.IMG_SIZE)")
    parser.add_argument("--pca-variance", type=float, default=None,
                         help="Overrides config.PCA_VARIANCE (pca and cnn methods)")
    parser.add_argument("--threshold-percentile", type=float, default=None,
                         help="Overrides config.THRESHOLD_PERCENTILE")
    parser.add_argument("--ae-bottleneck", type=int, default=32, help="Autoencoder bottleneck size (autoencoder only)")
    parser.add_argument("--ae-hidden", type=int, default=128, help="Autoencoder hidden layer size (autoencoder only)")
    parser.add_argument("--noise-std", type=float, default=0.0,
                         help="Denoising autoencoder: Gaussian noise std added to training input only "
                              "(0 = plain autoencoder, e.g. try 0.1)")
    parser.add_argument("--no-pretrained", action="store_true",
                         help="cnn method only: use a randomly initialised ResNet18 instead of "
                              "ImageNet-pretrained weights (mainly for offline testing - much weaker results)")
    args = parser.parse_args()

    main(
        category=args.category or CATEGORY,
        img_size=parse_img_size(args.img_size) if args.img_size else IMG_SIZE,
        method=args.method,
        pca_variance=args.pca_variance or PCA_VARIANCE,
        threshold_percentile=args.threshold_percentile or THRESHOLD_PERCENTILE,
        ae_bottleneck=args.ae_bottleneck,
        ae_hidden=args.ae_hidden,
        ae_noise_std=args.noise_std,
        cnn_pretrained=not args.no_pretrained,
    )
