"""Load images from an MVTec-AD style folder structure.

Expected layout:
    data/<category>/train/good/*.png
    data/<category>/test/good/*.png
    data/<category>/test/<defect_type>/*.png

Normal training images may also be placed directly inside train/.
Images are returned as flattened vectors so the original PCA/evaluation/
visualisation pipeline remains compatible. The ConvAE reshapes them back to
images internally.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from PIL import Image

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Expected folder not found: {os.path.abspath(folder)}")
    return sorted(
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.lower().endswith(VALID_EXTENSIONS)
    )


def load_image(path: str, img_size: tuple[int, int], grayscale: bool) -> np.ndarray:
    mode = "L" if grayscale else "RGB"
    with Image.open(path) as image:
        image = image.convert(mode)
        image = image.resize(img_size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def load_paths_as_vectors(
    paths: Sequence[str], img_size: tuple[int, int], grayscale: bool
) -> np.ndarray:
    vectors = [load_image(path, img_size, grayscale).reshape(-1) for path in paths]
    feature_count = img_size[0] * img_size[1] * (1 if grayscale else 3)
    if not vectors:
        return np.empty((0, feature_count), dtype=np.float32)
    return np.stack(vectors).astype(np.float32)


def load_folder_as_vectors(folder: str, img_size: tuple[int, int], grayscale: bool):
    paths = _list_images(folder)
    return load_paths_as_vectors(paths, img_size, grayscale), paths


def load_train_normal(data_root: str, category: str, img_size: tuple[int, int], grayscale: bool):
    train_dir = os.path.join(data_root, category, "train")
    train_good_dir = os.path.join(train_dir, "good")
    chosen_dir = train_good_dir if os.path.isdir(train_good_dir) else train_dir

    X, paths = load_folder_as_vectors(chosen_dir, img_size, grayscale)
    if len(X) < 2:
        raise RuntimeError(f"Need at least two normal training images in {os.path.abspath(chosen_dir)}")
    return X, paths


def split_normal_data(
    X: np.ndarray,
    paths: Sequence[str],
    validation_fraction: float,
    random_seed: int,
):
    """Deterministic split shared by PCA, ConvAE, and CNN.

    The same path indices are produced every time, so independently running
    the three methods still gives the same normal fit/validation images.
    """
    if len(X) != len(paths):
        raise ValueError("X and paths must have the same length.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(paths))
    n_validation = max(1, int(round(len(paths) * validation_fraction)))
    n_validation = min(n_validation, len(paths) - 1)

    validation_indices = indices[:n_validation]
    fit_indices = indices[n_validation:]
    paths_array = np.asarray(paths)

    return (
        X[fit_indices],
        X[validation_indices],
        paths_array[fit_indices].tolist(),
        paths_array[validation_indices].tolist(),
    )


def load_test_set_with_types(
    data_root: str, category: str, img_size: tuple[int, int], grayscale: bool
):
    """Return X, paths, binary labels, and defect-type names."""
    test_dir = os.path.join(data_root, category, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Expected folder not found: {os.path.abspath(test_dir)}")

    vectors_list: list[np.ndarray] = []
    paths: list[str] = []
    labels: list[int] = []
    defect_types: list[str] = []

    for subfolder in sorted(os.listdir(test_dir)):
        sub_path = os.path.join(test_dir, subfolder)
        if not os.path.isdir(sub_path):
            continue
        X_sub, paths_sub = load_folder_as_vectors(sub_path, img_size, grayscale)
        if len(X_sub) == 0:
            continue

        label = 0 if subfolder == "good" else 1
        vectors_list.append(X_sub)
        paths.extend(paths_sub)
        labels.extend([label] * len(paths_sub))
        defect_types.extend([subfolder] * len(paths_sub))

    if not vectors_list:
        raise RuntimeError(f"No test images found under {os.path.abspath(test_dir)}")

    return (
        np.vstack(vectors_list),
        paths,
        np.asarray(labels, dtype=np.int64),
        np.asarray(defect_types),
    )


def load_test_set(data_root: str, category: str, img_size: tuple[int, int], grayscale: bool):
    """Backward-compatible three-output version used by older code."""
    X, paths, labels, _ = load_test_set_with_types(data_root, category, img_size, grayscale)
    return X, paths, labels
