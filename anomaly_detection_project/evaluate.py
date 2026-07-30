"""
Step 2: Score the test images (normal + defective) and report metrics.

Usage:
    python evaluate.py
    python evaluate.py --category hazelnut
    python evaluate.py --category bottle --method autoencoder
    python evaluate.py --category bottle --method cnn
"""

import os
import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

from config import DATA_ROOT, MODEL_DIR, RESULTS_DIR, CATEGORY
from data_loader import load_test_set
from pca_model import PCAAnomalyDetector
from autoencoder_model import AEAnomalyDetector


def build_detector_for_eval(method):
    if method == "pca":
        return PCAAnomalyDetector()
    elif method == "autoencoder":
        return AEAnomalyDetector()
    elif method == "cnn":
        from cnn_model import CNNFeatureAnomalyDetector
        return CNNFeatureAnomalyDetector()
    else:
        raise ValueError(f"Unknown method '{method}'.")


def main(category, method):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, f"{category}_{method}.joblib")
    threshold_path = os.path.join(MODEL_DIR, f"{category}_{method}_threshold.txt")
    meta_path = os.path.join(MODEL_DIR, f"{category}_{method}_meta.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Run train.py --method {method} first.")

    with open(meta_path) as f:
        meta = json.load(f)
    img_size = tuple(meta["img_size"])
    grayscale = meta["grayscale"]

    detector = build_detector_for_eval(method).load(model_path)
    with open(threshold_path) as f:
        threshold = float(f.read())

    X_test, paths, labels, defect_types = load_test_set(DATA_ROOT, category, img_size, grayscale)

    if method == "cnn":
        # Score using deep features (not raw pixels); keep small RGB thumbnails
        # separately just for display in visualize.py (no pixel-level
        # "reconstruction" concept exists in feature space).
        from cnn_model import extract_features, load_thumbnail
        print(f"[{category}/{method}] Extracting ResNet18 features for {len(paths)} test images...")
        X_features = extract_features(paths, pretrained=meta.get("cnn_pretrained", True))
        errors, _ = detector.reconstruction_error(X_features)
        display_size = (128, 128)
        display_images = np.array([load_thumbnail(p, display_size).flatten() for p in paths])
        X_rec = np.zeros_like(display_images)  # no true reconstruction for this method
        has_reconstruction = False
        out_img_size = display_size
        out_grayscale = False
        X_test_out = display_images
    else:
        errors, X_rec = detector.reconstruction_error(X_test)
        has_reconstruction = True
        out_img_size = img_size
        out_grayscale = grayscale
        X_test_out = X_test

    predictions = (errors > threshold).astype(int)

    auc = roc_auc_score(labels, errors)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    cm = confusion_matrix(labels, predictions)

    print(f"Category: {category}  Method: {method}")
    print(f"Test images: {len(labels)}  (normal={np.sum(labels == 0)}, defective={np.sum(labels == 1)})")
    print(f"AUC-ROC: {auc:.3f}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(cm)

    out_path = os.path.join(RESULTS_DIR, f"{category}_{method}_eval.npz")
    np.savez(
        out_path,
        errors=errors,
        labels=labels,
        defect_types=defect_types,
        predictions=predictions,
        threshold=threshold,
        paths=np.array(paths),
        X_rec=X_rec,
        X_test=X_test_out,
        img_size=np.array(out_img_size),
        grayscale=out_grayscale,
        has_reconstruction=has_reconstruction,
        method=method,
        auc=auc,
        precision=precision,
        recall=recall,
    )
    print(f"Saved raw results -> {out_path} (used by visualize.py / compare.py)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=None, help="Override config.CATEGORY")
    parser.add_argument("--method", choices=["pca", "autoencoder", "cnn"], default="pca")
    args = parser.parse_args()
    main(args.category or CATEGORY, args.method)
