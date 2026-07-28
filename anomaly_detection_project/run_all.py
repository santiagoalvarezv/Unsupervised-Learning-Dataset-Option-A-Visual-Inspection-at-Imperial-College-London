"""
Run the FULL pipeline for one category in a single command:
  train + evaluate + visualize for each method, then compare.py.

Usage:
    python run_all.py --category hazelnut                       # pca + autoencoder (default)
    python run_all.py --category bottle --methods pca,cnn        # pca + cnn only
    python run_all.py --category bottle --methods pca,autoencoder,cnn   # all three
    python run_all.py --category metal_nut --img-size 128x128    # optional overrides, same flags as train.py
"""

import argparse
import subprocess
import sys

from config import CATEGORY


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main(category, methods, extra_train_args):
    for method in methods:
        run([sys.executable, "train.py", "--category", category, "--method", method] + extra_train_args)
        run([sys.executable, "evaluate.py", "--category", category, "--method", method])
        run([sys.executable, "visualize.py", "--category", category, "--method", method])

    run([sys.executable, "compare.py", "--category", category, "--methods", ",".join(methods)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="Override config.CATEGORY")
    parser.add_argument("--methods", default="pca,autoencoder",
                         help="Comma-separated list of methods to run, e.g. pca,autoencoder,cnn "
                              "(cnn requires torch/torchvision installed and internet on first run)")
    parser.add_argument("--img-size", default=None, help="Passed through to train.py (all methods)")
    parser.add_argument("--pca-variance", default=None, help="Passed through to train.py (pca/cnn methods)")
    parser.add_argument("--threshold-percentile", default=None, help="Passed through to train.py (all methods)")
    args = parser.parse_args()

    extra = []
    if args.img_size:
        extra += ["--img-size", args.img_size]
    if args.pca_variance:
        extra += ["--pca-variance", args.pca_variance]
    if args.threshold_percentile:
        extra += ["--threshold-percentile", args.threshold_percentile]

    main(args.category or CATEGORY, [m.strip() for m in args.methods.split(",")], extra)
