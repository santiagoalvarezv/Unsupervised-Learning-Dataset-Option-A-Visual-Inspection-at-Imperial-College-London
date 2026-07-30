# Changes from the uploaded group version

## Preserved

- `train.py -> evaluate.py -> visualize.py -> compare.py` workflow
- method names `pca`, `autoencoder`, and `cnn`
- ResNet18 global feature extraction and PCA detector in `cnn_model.py`
- result file naming based on category and method
- original reconstruction/error-map visualisation for PCA and Autoencoder
- original image-and-score visualisation for CNN

## Replaced

- The old Autoencoder was `MLPRegressor` on flattened pixels.
- It is now a PyTorch convolutional autoencoder with four downsampling
  convolutions and four transposed-convolution upsampling layers.

## Added

- shared deterministic normal fit/validation split
- normal-validation threshold selection
- ConvAE early stopping based on normal-validation MSE
- fixed PCA component count and matching ConvAE latent dimension
- AP, F1, specificity, confusion counts, and per-defect rates
- checks in `compare.py` that methods used the same normal split
- canonical Python filenames without `(2)` or `(4)` suffixes
