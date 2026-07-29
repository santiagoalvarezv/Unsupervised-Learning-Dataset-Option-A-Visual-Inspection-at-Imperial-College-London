# Image Anomaly Detection: PCA, Basic ConvAE, and CNN+PCA

This version is a direct revision of the original group project rather than a
separate replacement. It keeps the original three-method pipeline:

1. **PCA** on raw grayscale pixels
2. **Autoencoder**, now changed from a flattened MLP to a basic convolutional autoencoder
3. **CNN**, unchanged in principle: ImageNet-pretrained ResNet18 features + PCA

The same commands and method names remain available: `pca`, `autoencoder`, and `cnn`.

## What was changed

- Kept `cnn_model.py` and the CNN scoring route.
- Replaced `MLPRegressor` with a real convolutional encoder-decoder.
- PCA and ConvAE both use the same `128x128` grayscale images and a 64-number representation.
- All three methods use the same deterministic split of normal training images:
  - 80% normal fit set
  - 20% normal validation set
- Thresholds are selected from the normal validation scores, not from images used to fit the model.
- ConvAE uses plain MSE, Adam, sigmoid output, and early stopping on normal validation MSE.
- Evaluation now also reports AP, F1, specificity, and per-defect recall.

## Dataset structure

```text
data/
└── metal_nut/
    ├── train/
    │   └── good/
    └── test/
        ├── good/
        ├── bent/
        ├── color/
        ├── flip/
        └── scratch/
```

Any test subfolder other than `good` is treated as defective.

## Install

```bash
pip install -r requirements.txt
```

PyTorch is now required by both the ConvAE and CNN methods.

## Recommended main comparison

```bash
python run_all.py --category metal_nut --methods pca,autoencoder
```

This is the clean, fair baseline comparison.

## Keep and run the CNN extension

```bash
python run_all.py --category metal_nut --methods pca,autoencoder,cnn
```

The first CNN run may download ResNet18 ImageNet weights. The CNN method still
uses RGB `224x224` input internally, so it should be discussed as a pretrained
transfer-learning extension rather than as a perfectly matched raw-image baseline.

You can also run only CNN:

```bash
python run_all.py --category metal_nut --methods cnn
```

## Quick code test

```bash
python run_all.py \
  --category metal_nut \
  --methods pca,autoencoder \
  --img-size 64x64 \
  --pca-components 16 \
  --latent-dim 16 \
  --epochs 3 \
  --patience 2
```

## Formal default settings

```text
PCA / ConvAE input: 128x128 grayscale
PCA components: 64
ConvAE latent dimension: 64
Normal validation fraction: 20%
Threshold: 95th percentile of normal validation scores
ConvAE loss: pixel MSE
Optimizer: Adam
Learning rate: 0.001
Epochs: 50
Early-stopping patience: 8
```

## Important interpretation

- The main fair comparison is **raw-pixel PCA vs basic ConvAE**.
- The retained CNN method has an external ImageNet-pretraining advantage and
  uses a different RGB resolution. It is still valid, but should be labelled
  **ResNet18 features + PCA** and presented as an enhanced method.
- ConvAE is not guaranteed to beat PCA. The purpose of the revision is to make
  the comparison meaningful, not to force a preferred result.

## File-name note

The uploaded group files contained names such as `config(2).py` and
`cnn_model(4).py`, while the imports expected `config.py` and `cnn_model.py`.
This folder uses the canonical names, so the scripts can import each other.
