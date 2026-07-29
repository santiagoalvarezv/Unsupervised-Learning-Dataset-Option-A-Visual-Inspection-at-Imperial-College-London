"""Score test images for PCA, ConvAE, or CNN+PCA and save common results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from autoencoder_model import AEAnomalyDetector
from config import CATEGORY, DATA_ROOT, MODEL_DIR, RESULTS_DIR
from data_loader import load_test_set_with_types
from pca_model import PCAAnomalyDetector


def model_path(category: str, method: str) -> str:
    extension = ".pt" if method == "autoencoder" else ".joblib"
    return os.path.join(MODEL_DIR, f"{category}_{method}{extension}")


def build_detector_for_eval(method: str, device: str):
    if method == "pca":
        return PCAAnomalyDetector()
    if method == "autoencoder":
        return AEAnomalyDetector(device=device)
    if method == "cnn":
        from cnn_model import CNNFeatureAnomalyDetector
        return CNNFeatureAnomalyDetector()
    raise ValueError(f"Unknown method: {method}")


def main(category: str, method: str, device: str, cnn_device: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    saved_model_path = model_path(category, method)
    threshold_path = os.path.join(MODEL_DIR, f"{category}_{method}_threshold.txt")
    meta_path = os.path.join(MODEL_DIR, f"{category}_{method}_meta.json")
    for required in [saved_model_path, threshold_path, meta_path]:
        if not os.path.exists(required):
            raise FileNotFoundError(f"Missing {required}. Train {method} first.")

    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    img_size = tuple(meta["img_size"])
    grayscale = bool(meta["grayscale"])
    threshold = float(Path(threshold_path).read_text(encoding="utf-8"))

    detector = build_detector_for_eval(method, device).load(saved_model_path)
    X_test, paths, labels, defect_types = load_test_set_with_types(
        DATA_ROOT, category, img_size, grayscale
    )

    if method == "cnn":
        from cnn_model import extract_features, load_thumbnail
        print(f"[{category}/cnn] Extracting ResNet18 features for {len(paths)} test images...")
        X_features = extract_features(
            paths,
            device=cnn_device,
            pretrained=meta.get("cnn_pretrained", True),
        )
        errors, _ = detector.reconstruction_error(X_features)
        display_size = (128, 128)
        X_test_out = np.asarray([load_thumbnail(path, display_size).reshape(-1) for path in paths])
        X_rec = np.zeros_like(X_test_out)
        out_img_size = display_size
        out_grayscale = False
        has_reconstruction = False
    else:
        errors, X_rec = detector.reconstruction_error(X_test)
        X_test_out = X_test
        out_img_size = img_size
        out_grayscale = grayscale
        has_reconstruction = True

    predictions = (errors > threshold).astype(np.int64)
    auc = float(roc_auc_score(labels, errors))
    average_precision = float(average_precision_score(labels, errors))
    precision = float(precision_score(labels, predictions, zero_division=0))
    recall = float(recall_score(labels, predictions, zero_division=0))
    f1 = float(f1_score(labels, predictions, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    print(f"Category: {category}  Method: {method}")
    print(f"Test images: {len(labels)} (normal={np.sum(labels == 0)}, defective={np.sum(labels == 1)})")
    print(f"AUC-ROC: {auc:.3f}  Average precision: {average_precision:.3f}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Confusion matrix [[TN FP] [FN TP]] = [[{tn} {fp}] [{fn} {tp}]]")

    for defect_type in sorted(set(defect_types.tolist())):
        mask = defect_types == defect_type
        rate = float(np.mean(predictions[mask] == 1))
        rate_name = "false-positive rate" if defect_type == "good" else "recall"
        print(f"  {defect_type}: n={int(np.sum(mask))}, {rate_name}={rate:.3f}")

    out_path = os.path.join(RESULTS_DIR, f"{category}_{method}_eval.npz")
    np.savez_compressed(
        out_path,
        errors=errors,
        labels=labels,
        predictions=predictions,
        threshold=threshold,
        paths=np.asarray(paths),
        defect_types=defect_types,
        X_rec=X_rec,
        X_test=X_test_out,
        img_size=np.asarray(out_img_size),
        grayscale=out_grayscale,
        has_reconstruction=has_reconstruction,
        method=method,
        auc=auc,
        average_precision=average_precision,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )
    print(f"Saved raw results -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--method", choices=["pca", "autoencoder", "cnn"], default="pca")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cnn-device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    main(args.category, args.method, args.device, args.cnn_device)
