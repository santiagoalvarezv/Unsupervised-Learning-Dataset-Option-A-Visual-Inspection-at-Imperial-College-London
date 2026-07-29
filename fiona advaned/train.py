"""Train one of the three anomaly detectors on normal images only.

Examples:
    python train.py --category metal_nut --method pca
    python train.py --category metal_nut --method autoencoder
    python train.py --category metal_nut --method cnn
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from autoencoder_model import AEAnomalyDetector
from config import (
    AE_BATCH_SIZE,
    AE_EPOCHS,
    AE_LATENT_DIM,
    AE_LEARNING_RATE,
    AE_PATIENCE,
    CATEGORY,
    CNN_PCA_VARIANCE,
    DATA_ROOT,
    GRAYSCALE,
    IMG_SIZE,
    MODEL_DIR,
    PCA_COMPONENTS,
    RANDOM_SEED,
    THRESHOLD_PERCENTILE,
    VALIDATION_FRACTION,
)
from data_loader import load_train_normal, split_normal_data
from pca_model import PCAAnomalyDetector


def parse_img_size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x")
        size = (int(width), int(height))
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Image size must look like 128x128.") from exc
    if size[0] != size[1] or size[0] % 16 != 0:
        raise argparse.ArgumentTypeError("Use a square size divisible by 16, e.g. 64x64 or 128x128.")
    return size


def model_path(category: str, method: str) -> str:
    extension = ".pt" if method == "autoencoder" else ".joblib"
    return os.path.join(MODEL_DIR, f"{category}_{method}{extension}")


def build_detector(args: argparse.Namespace):
    if args.method == "pca":
        return PCAAnomalyDetector(n_components=args.pca_components)
    if args.method == "autoencoder":
        return AEAnomalyDetector(
            latent_dim=args.latent_dim,
            img_size=args.img_size,
            grayscale=GRAYSCALE,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            random_state=args.seed,
            device=args.device,
        )
    if args.method == "cnn":
        from cnn_model import CNNFeatureAnomalyDetector
        return CNNFeatureAnomalyDetector(variance=args.cnn_pca_variance)
    raise ValueError(f"Unknown method: {args.method}")


def main(args: argparse.Namespace):
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_all, all_paths = load_train_normal(DATA_ROOT, args.category, args.img_size, GRAYSCALE)
    X_fit, X_validation, fit_paths, validation_paths = split_normal_data(
        X_all,
        all_paths,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
    )

    detector = build_detector(args)
    if args.method == "cnn":
        from cnn_model import extract_features
        print(f"[{args.category}/cnn] Extracting ResNet18 features for normal fit images...")
        X_fit_model = extract_features(
            fit_paths,
            device=args.device_for_cnn,
            pretrained=not args.no_pretrained,
        )
        print(f"[{args.category}/cnn] Extracting ResNet18 features for normal validation images...")
        X_validation_model = extract_features(
            validation_paths,
            device=args.device_for_cnn,
            pretrained=not args.no_pretrained,
        )
        detector.fit(X_fit_model)
    elif args.method == "autoencoder":
        X_fit_model = X_fit
        X_validation_model = X_validation
        detector.fit(X_fit_model, X_validation_model)
    else:
        X_fit_model = X_fit
        X_validation_model = X_validation
        detector.fit(X_fit_model)

    fit_errors, _ = detector.reconstruction_error(X_fit_model)
    validation_errors, _ = detector.reconstruction_error(X_validation_model)
    threshold = float(np.percentile(validation_errors, args.threshold_percentile))

    saved_model_path = model_path(args.category, args.method)
    threshold_path = os.path.join(MODEL_DIR, f"{args.category}_{args.method}_threshold.txt")
    meta_path = os.path.join(MODEL_DIR, f"{args.category}_{args.method}_meta.json")

    detector.save(saved_model_path)
    Path(threshold_path).write_text(f"{threshold:.12g}\n", encoding="utf-8")

    metadata = {
        "category": args.category,
        "method": args.method,
        "img_size": list(args.img_size),
        "grayscale": GRAYSCALE,
        "validation_fraction": args.validation_fraction,
        "random_seed": args.seed,
        "threshold_percentile": args.threshold_percentile,
        "threshold_source": "normal_validation",
        "fit_paths": fit_paths,
        "validation_paths": validation_paths,
        "fit_error_mean": float(np.mean(fit_errors)),
        "validation_error_mean": float(np.mean(validation_errors)),
        "threshold": threshold,
        "pca_components": args.pca_components,
        "latent_dim": args.latent_dim,
        "cnn_pca_variance": args.cnn_pca_variance,
        "cnn_pretrained": not args.no_pretrained,
    }
    if args.method == "pca":
        metadata["effective_pca_components"] = detector.effective_components
        metadata["explained_variance_ratio_sum"] = float(detector.pca.explained_variance_ratio_.sum())
    elif args.method == "autoencoder":
        metadata.update(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "patience": args.patience,
                "best_epoch": detector.history.best_epoch,
                "train_loss_history": detector.history.train_loss,
                "validation_loss_history": detector.history.validation_loss,
            }
        )
    else:
        metadata["effective_cnn_pca_components"] = int(detector.pca.n_components_)

    Path(meta_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[{args.category}/{args.method}] normal fit images: {len(fit_paths)}")
    print(f"[{args.category}/{args.method}] normal validation images: {len(validation_paths)}")
    print(f"[{args.category}/{args.method}] fit mean score: {np.mean(fit_errors):.6f}")
    print(f"[{args.category}/{args.method}] validation mean score: {np.mean(validation_errors):.6f}")
    print(
        f"[{args.category}/{args.method}] threshold={threshold:.6f} "
        f"({args.threshold_percentile:g}th percentile of normal validation scores)"
    )
    print(f"[{args.category}/{args.method}] Saved model -> {saved_model_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--method", choices=["pca", "autoencoder", "cnn"], default="pca")
    parser.add_argument("--img-size", type=parse_img_size, default=IMG_SIZE)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    parser.add_argument("--threshold-percentile", type=float, default=THRESHOLD_PERCENTILE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    parser.add_argument("--pca-components", type=int, default=PCA_COMPONENTS)
    parser.add_argument("--cnn-pca-variance", type=float, default=CNN_PCA_VARIANCE)

    parser.add_argument("--latent-dim", type=int, default=AE_LATENT_DIM)
    parser.add_argument("--epochs", type=int, default=AE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=AE_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=AE_LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=AE_PATIENCE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="CNN only: use random ResNet18 weights. Do not use for final results.",
    )
    parser.add_argument(
        "--cnn-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for ResNet18 feature extraction.",
    )
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.cnn_device == "auto":
        try:
            import torch
            parsed.device_for_cnn = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            parsed.device_for_cnn = "cpu"
    else:
        parsed.device_for_cnn = parsed.cnn_device
    main(parsed)
