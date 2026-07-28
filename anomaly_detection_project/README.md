# Anomaly Detection Project (Option A: Visual Inspection)

Three anomaly detectors, all trained only on normal images:

1. **PCA** — linear, works on raw pixels.
2. **Autoencoder** — nonlinear neural network, also works on raw pixels.
3. **CNN features** — PCA applied to deep features from a pretrained
   ResNet18 instead of raw pixels (closer to what current state-of-the-art
   methods do on this benchmark).

Each method's reconstruction/feature error is used as its anomaly score.
Includes a comparison script to decide which method performs better per
category.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

`torch` and `torchvision` are listed but only needed for `--method cnn` —
skip installing those two if you're only using `pca`/`autoencoder`.

## 2. Where to put your dataset (this is the important part)

Don't upload the images to Claude, and don't push them to GitHub either
(see section 8) — the project reads them directly from disk. Place your
dataset inside the `data/` folder, following this structure (the same one
MVTec-AD style datasets already use):

```
anomaly_detection_project/
└── data/
    ├── bottle/
    │   ├── train/
    │   │   ├── good/               <- optional subfolder, only normal images
    │   │   │   ├── 000.png
    │   │   │   └── ...
    │   │   └── (or just drop images directly in train/, no subfolder needed)
    │   └── test/
    │       ├── good/               <- normal test images
    │       ├── broken_large/       <- one subfolder per defect type
    │       ├── broken_small/
    │       └── contamination/
    ├── hazelnut/
    │   ├── train/
    │   └── test/(good, crack, cut, hole, print, ...)
    ├── metal_nut/
    ├── transistor/
    └── wood/
```

Key rules:

- **`train/`** only ever contains normal images. You can put them directly
  in `train/`, or inside a `train/good/` subfolder if that's how your
  download is already organized — both work, no code changes needed.
- The `test/good/` subfolder is treated as **normal** (label 0).
- **Any other** subfolder inside `test/` (the name doesn't matter — it can
  be `broken_large`, `crack`, `scratch`, etc.) is treated as **defective**
  (label 1).
- No code edits needed for new defect types: the script automatically walks
  every subfolder under `test/`.
- If your downloaded dataset already has this exact structure, just
  copy/move each category folder into `data/`. No renaming required.

## 3. Choosing which category to run

Edit `config.py`:

```python
CATEGORY = "bottle"   # or "hazelnut", "metal_nut", "transistor", "wood"
```

Or pass it on the command line without touching the file (see below).

## 4. Running everything for one category: `run_all.py`

This is the easiest way to run the whole pipeline. One command trains and
evaluates every method you choose and produces the comparison:

```bash
python run_all.py --category bottle                      # pca + autoencoder (default)
python run_all.py --category bottle --methods pca,cnn     # pca + cnn only
python run_all.py --category bottle --methods pca,autoencoder,cnn   # all three
```

Run it once per category (`bottle`, `hazelnut`, `metal_nut`, `transistor`,
`wood`). Each run does, in order, for every method in `--methods`:

1. `train.py` + `evaluate.py` + `visualize.py` for that method
2. `compare.py` at the end (side-by-side metrics table + combined ROC curve)

`run_all.py` also accepts `--img-size`, `--pca-variance`, and
`--threshold-percentile` if you want to override the defaults for one
category (passed through to every method's training step):

```bash
python run_all.py --category metal_nut --img-size 128x128
```

### Running steps individually (optional)

If you'd rather run one step at a time (e.g. while debugging), call each
script directly, always specifying `--method` (`pca`, `autoencoder`, or
`cnn` — default is `pca` if omitted):

```bash
python train.py --category bottle --method pca
python evaluate.py --category bottle --method pca
python visualize.py --category bottle --method pca

python train.py --category bottle --method autoencoder
python evaluate.py --category bottle --method autoencoder
python visualize.py --category bottle --method autoencoder

python compare.py --category bottle --methods pca,autoencoder
```

Extra flags for `train.py`:
- `--img-size` (e.g. `128x128`), `--pca-variance` (e.g. `0.95`),
  `--threshold-percentile` (e.g. `95`) — see section 6. `--pca-variance`
  applies to both `pca` and `cnn` methods.
- `--ae-bottleneck` (default 32) / `--ae-hidden` (default 128) — autoencoder
  architecture size.
- `--noise-std` (default 0.0) — turns the autoencoder into a *denoising*
  autoencoder: Gaussian noise is added to the input during training only,
  while the target stays the clean image. Try `--noise-std 0.1`. Optional —
  leaving it at 0 (the default) trains a plain autoencoder, nothing changes
  if you never pass this flag.

The exact settings used for a given run are saved alongside its model, so
`evaluate.py`/`visualize.py` always pick up the right values automatically
— no need to repeat the flags on those steps.

## 5. The CNN features method in more detail

PCA and the autoencoder both work directly on raw pixel values, which is
why they miss subtle defects in some categories (see notes on `metal_nut`
and `wood` below). A stronger approach — closer to what actually wins on
the MVTec-AD benchmark (methods like PaDiM/PatchCore) — is to first extract
**deep features** from a CNN pretrained on ImageNet (ResNet18), then apply
PCA on those features instead of on raw pixels. Deep features already
encode textures, edges and shapes learned from millions of photos, so they
carry far more information than pixel values alone.

The **first** time you run `--method cnn`, it needs internet access to
download the pretrained ResNet18 weights (~45MB) — cached locally
afterwards, no internet needed on later runs.

Notes specific to this method:
- Reuses the `--pca-variance` flag from the `pca` method (PCA is just
  applied to CNN features here instead of pixels).
- There's no pixel-level "reconstruction" in feature space, so
  `visualize.py` shows only the original image + its anomaly score for this
  method (no reconstruction/heatmap panels like PCA/autoencoder get).
- `--no-pretrained` uses a randomly-initialised ResNet18 instead of
  ImageNet weights — only useful for offline testing, gives much weaker
  results, don't use it for your actual report.

## 6. What to check in `results/`

Every file is named `<category>_<method>_...`, so different methods'
outputs never overwrite each other:

- `<category>_<method>_score_histogram.png` — normal vs. defective score
  distribution, with the threshold line.
- `<category>_<method>_true_positive.png` — correctly detected defects
  (original / reconstruction / error map for pca/autoencoder; original +
  score only for cnn).
- `<category>_<method>_false_alarm.png` — normal images flagged as
  defective.
- `<category>_<method>_missed_defect.png` — defects the model failed to
  catch.
- `<category>_<method>_correct_normal.png` — normal images correctly
  passed.
- `<category>_<methods>_roc.png` — combined ROC curve from `compare.py`,
  the clearest single picture for "which method is better" (e.g.
  `bottle_pca_vs_autoencoder_roc.png`).

The four example categories above (true positive / false alarm / missed
defect / correct normal) are exactly what the project guide asks for in the
deliverables section.

## 7. Simple tweaks if results aren't convincing

Passed to `train.py` (or `run_all.py`, which forwards them):

- Bigger `--img-size` (e.g. `128x128`) if fine detail is being lost —
  slower to run (doesn't apply to `cnn`, which always resizes to 224x224
  internally).
- Higher `--pca-variance` (e.g. `0.95`) for finer reconstructions
  (pca/cnn).
- Lower `--threshold-percentile` (e.g. `95`) if too many defects go
  undetected (lowers the alarm bar); higher if there are too many false
  alarms. Note: this changes the operating point, not the underlying
  AUC-ROC — see notes below.
- For the autoencoder specifically: `--ae-bottleneck`, `--ae-hidden`, or
  `--noise-std` (denoising variant).
- Try `--method cnn` if `pca`/`autoencoder` are both struggling on a
  category — see section 5.

## 8. Sharing the project on GitHub (without the dataset)

Do **not** upload the `data/` folder to GitHub — it's thousands of files
and GitHub isn't meant for datasets this size. A `.gitignore` file should
already be excluding it (along with `models/`, `venv/`, and Python cache
files) so it won't get pushed by accident. If you don't have a
`.gitignore` yet, create one at the project root with:

```
data/
models/*.joblib
venv/
__pycache__/
*.pyc
```

**What your teammates need to do:**

1. Clone/pull the repo (they'll get all the code, but no `data/` folder).
2. Download the dataset themselves from the official MVTec-AD website:
   **https://www.mvtec.com/company/research/datasets/mvtec-ad**
   (free download, just fill in the short form). They only need these 5
   categories: `bottle`, `hazelnut`, `metal_nut`, `transistor`, `wood`.
3. Place the downloaded category folders into their own local `data/`
   folder, following the structure in section 2.

**Pushing your code to GitHub** (from the project root, in the VS Code
terminal):

```bash
git init
git add .
git commit -m "Anomaly detection project"
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

If the remote repo already has commits (e.g. you created it with a README
on GitHub's website first), you may need:

```bash
git pull origin main --allow-unrelated-histories
```

before pushing.

## 9. Notes for the report

- Training never touches `test/` images — the script makes this impossible
  by design (`train.py` only ever reads from the `train/` folder).
- The AUC-ROC printed by `evaluate.py`/`compare.py` is the most honest
  metric when classes are imbalanced (few defects vs. many normal images) —
  it doesn't depend on where the threshold is set, unlike precision/recall.
- Each example's error map is your defect localisation: the brighter/hotter
  the area, the more it contributed to the alarm (pca/autoencoder only).
- If one category performs much worse than the others (e.g. low AUC-ROC
  even after trying different image sizes/thresholds), that's a legitimate
  finding for the report, not necessarily a bug — explain the likely cause
  (e.g. object rotation/misalignment between photos breaking a
  pixel-by-pixel method like PCA) rather than only chasing a higher number.
- When comparing methods, one with perfect precision but low recall isn't
  automatically "better" — think about which type of mistake matters more
  for a real inspection line (missing a real defect is usually far costlier
  than a false alarm). AUC-ROC is the fairer way to compare methods'
  underlying separation ability, independent of the specific threshold each
  one happens to use.
- A denoising autoencoder (`--noise-std > 0`) isn't guaranteed to help —
  if it doesn't improve AUC over the plain autoencoder for a category,
  that's still worth reporting: it suggests the limitation is the
  bottleneck's information capacity, not pixel memorisation.
- If `cnn` clearly beats `pca`/`autoencoder` on a category where those two
  struggled, that's strong evidence the limitation was working in raw pixel
  space rather than a fundamental limit of reconstruction-error-based
  anomaly detection.
