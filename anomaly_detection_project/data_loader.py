"""
Loads images from the standard MVTec-AD style folder layout:

data/<category>/train/good/*.png
data/<category>/test/good/*.png
data/<category>/test/<defect_type>/*.png   (one or more defect subfolders)

Every image is converted to grayscale (or kept RGB), resized, scaled to
[0, 1], and flattened into a 1D feature vector for PCA.
"""

import os
import numpy as np
from PIL import Image

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _list_images(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Expected folder not found: {folder}")
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXTENSIONS)
    )


def load_image(path, img_size, grayscale):
    img = Image.open(path).convert("L" if grayscale else "RGB")
    img = img.resize(img_size)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_folder_as_vectors(folder, img_size, grayscale):
    paths = _list_images(folder)
    vectors = [load_image(p, img_size, grayscale).flatten() for p in paths]
    return np.array(vectors), paths


def load_train_normal(data_root, category, img_size, grayscale):
    """
    Accepts either layout:
      data/<category>/train/good/*.png   (standard MVTec layout)
      data/<category>/train/*.png        (images directly in train/, no subfolder needed)
    Since train/ only ever contains normal images, both are treated the same way.
    """
    train_good_dir = os.path.join(data_root, category, "train", "good")
    train_dir = os.path.join(data_root, category, "train")

    if os.path.isdir(train_good_dir):
        chosen_dir = train_good_dir
    else:
        chosen_dir = train_dir

    X, paths = load_folder_as_vectors(chosen_dir, img_size, grayscale)
    if len(X) == 0:
        raise RuntimeError(f"No training images found in {chosen_dir}")
    return X, paths


def load_test_set(data_root, category, img_size, grayscale):
    """
    Returns:
        X: (N, D) array of all test image vectors
        paths: list of N file paths (same order as X)
        labels: (N,) array, 0 = normal ("good"), 1 = defective (any other subfolder)
    """
    test_dir = os.path.join(data_root, category, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Expected folder not found: {test_dir}")

    vectors_list, paths, labels = [], [], []
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

    if not vectors_list:
        raise RuntimeError(f"No test images found under {test_dir}")

    return np.vstack(vectors_list), paths, np.array(labels)
