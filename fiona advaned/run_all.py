"""Run train -> evaluate -> visualize -> compare for selected methods.

Examples:
    python run_all.py --category metal_nut
    python run_all.py --category metal_nut --methods pca,autoencoder,cnn
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from config import (
    AE_BATCH_SIZE,
    AE_EPOCHS,
    AE_LATENT_DIM,
    AE_LEARNING_RATE,
    AE_PATIENCE,
    CATEGORY,
    CNN_PCA_VARIANCE,
    IMG_SIZE,
    PCA_COMPONENTS,
    RANDOM_SEED,
    THRESHOLD_PERCENTILE,
    VALIDATION_FRACTION,
)


def run(command: list[str]):
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main(args: argparse.Namespace):
    common_train = [
        "--category", args.category,
        "--img-size", args.img_size,
        "--validation-fraction", str(args.validation_fraction),
        "--threshold-percentile", str(args.threshold_percentile),
        "--seed", str(args.seed),
        "--pca-components", str(args.pca_components),
        "--cnn-pca-variance", str(args.cnn_pca_variance),
        "--latent-dim", str(args.latent_dim),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--patience", str(args.patience),
        "--device", args.device,
        "--cnn-device", args.cnn_device,
    ]
    if args.no_pretrained:
        common_train.append("--no-pretrained")

    for method in args.methods:
        run([sys.executable, "train.py", "--method", method, *common_train])
        eval_command = [
            sys.executable,
            "evaluate.py",
            "--category", args.category,
            "--method", method,
            "--device", args.device,
            "--cnn-device", "cpu" if args.cnn_device == "auto" else args.cnn_device,
        ]
        run(eval_command)
        run([sys.executable, "visualize.py", "--category", args.category, "--method", method])

    run(
        [
            sys.executable,
            "compare.py",
            "--category", args.category,
            "--methods", ",".join(args.methods),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--methods", default="pca,autoencoder")
    parser.add_argument("--img-size", default=f"{IMG_SIZE[0]}x{IMG_SIZE[1]}")
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
    parser.add_argument("--cnn-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parsed = parser.parse_args()
    parsed.methods = [method.strip() for method in parsed.methods.split(",") if method.strip()]
    allowed = {"pca", "autoencoder", "cnn"}
    if any(method not in allowed for method in parsed.methods):
        raise SystemExit("--methods must contain pca, autoencoder, and/or cnn.")
    main(parsed)
