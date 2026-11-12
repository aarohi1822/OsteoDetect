# TB Chest X-ray Classifier: Leakage Audit and Re-evaluation

Binary classifier for chest X-rays: Tuberculosis vs Normal. The data is the
[TB Chest Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset):
4200 PNGs (3500 Normal, 700 TB), 512×512.

This folder re-audits the original Colab notebook (`BONE_TB.ipynb`). It also
replaces that notebook's reported numbers with results on a held-out test set
that has no leakage.

> Despite the notebook's name ("Bone TB"), the dataset is **pulmonary** chest
> radiographs. It contains no bone or skeletal TB images.

## What was wrong with the original evaluation

| Issue | Evidence | Effect |
|---|---|---|
| **Validation set = training set** | `train_dir` and `val_dir` are the same folder; both generators print `Found 4200 images` | The reported **95.69 %** "validation accuracy" is training accuracy. There was no held-out data. |
| **ResNet50 collapsed to the majority class** | `val_accuracy` is exactly **0.8333** in every epoch, and 3500/4200 = 0.8333 | The ResNet50 predicted "Normal" for every image. The cause is preprocessing: Keras ResNet50 expects `preprocess_input` (caffe-style mean subtraction on raw 0–255 pixels), but the notebook fed it `/255` pixels, so inputs were almost constant. The backbone was also fully frozen. |
| **No test set; accuracy as the only metric** | n/a | Accuracy hides poor TB recall when classes are 5:1 imbalanced (see results below). |
| **Previous README claims** | It said "bone tuberculosis vs pulmonary TB" and "test accuracy ~92.69 %" | The data is **Normal vs TB chest X-rays**, and the notebook had no test split. Neither claim is supported. They are replaced by the results below. |

## 1. Duplicate audit (`src/hash_audit.py`)

For every image the script computes:
- the MD5 of the file bytes
- the MD5 of the decoded pixels
- a perceptual hash (pHash) and a difference hash (dHash)
- a 64×64 grayscale thumbnail, z-scored

It then compares every pair of images. Pairs are tiered by the Pearson
correlation of their thumbnails. The cut-offs were set by looking at pairs on
either side of each threshold.

| Tier | Rule | Pairs |
|---|---|---|
| Exact file duplicate | identical file MD5 | 3 |
| Exact pixel duplicate | identical pixel MD5, different file | 3 |
| Near-duplicate (same image, re-encoded or re-cropped) | corr ≥ 0.985 | 7 |
| Likely same patient (repeat study) | 0.955 ≤ corr < 0.985 | 27 |
| **Total** | | **40 pairs → 31 groups, 68 images** |

- **No pair crosses labels**, so no image appears as both TB and Normal.
- Pairs per source: 25 RSNA/RSNA (mostly repeat studies), 13 NLM/NLM (mostly
  exact or near copies), 4 Belarus/Belarus, 3 NLM/RSNA.
- pHash alone is not a reliable detector here. Most chest X-rays sit 10–24 bits
  apart, and many genuine same-patient pairs were 8–16 bits apart. The script
  therefore uses pixel correlation as the main signal.
- **Overlap across splits:** in the original notebook it was **100 %**
  (train = val). In the new split below it is **0 pairs**. `make_split.py`
  asserts this, and the run fails if any pair crosses splits.

Outputs: `reports/image_index.csv`, `reports/duplicate_pairs.csv`,
`reports/dup_groups.csv`.

## 2. Split (`src/make_split.py`)

**There are no patient IDs.** The metadata only gives filename, format, size
and source URL, so a true patient-level split is impossible.

As the closest stand-in:
- every duplicate or same-patient group from step 1 is treated as one group;
- the split uses `StratifiedGroupKFold`, stratified by **label × source**, at
  70 / 15 / 15 with seed 42.

| Split | Normal (NLM) | Normal (RSNA) | TB (Belarus) | TB (NLM) | Total |
|---|---|---|---|---|---|
| train | 291 | 2168 | 216 | 279 | 2954 |
| val | 63 | 458 | 47 | 57 | 625 |
| test | 52 | 468 | 43 | 58 | **621** |

Output: `reports/splits.csv`.

## 3–4. Results on the held-out test set

Training setup:
- All models train on **train**.
- The checkpoint with the lowest **val** loss is kept.
- Each model is scored **once** on **test**, with a 0.5 threshold.
- Brackets show 95 % bootstrap confidence intervals (2000 resamples).
- TB is the positive class.

| Model | Accuracy | Precision | Recall (sensitivity) | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| *Notebook claim: custom CNN (on training data)* | *95.69 %* | n/a | n/a | n/a | n/a | n/a |
| *Notebook claim: ResNet50 (on training data)* | *83.33 %* (all "Normal") | n/a | *0 %* | n/a | n/a | n/a |
| Custom CNN | 94.85 % [93.1–96.5] | 98.59 % [95.2–100] | **69.31 %** [60.4–77.8] | 99.81 % | 0.814 [0.745–0.872] | 0.975 [0.958–0.989] |
| ResNet50, notebook recipe (frozen) | 97.26 % [96.0–98.6] | 98.84 % [96.1–100] | 84.16 % [76.5–91.0] | 99.81 % | 0.909 [0.861–0.950] | 0.986 [0.973–0.995] |
| ResNet50, fine-tuned (12 epochs) | 98.87 % [98.1–99.7] | 100 % [100–100] | 93.07 % [87.9–97.8] | 100 % | 0.964 [0.935–0.989] | 0.995 [0.986–1.000] |
| **ResNet50, fine-tuned (25 epochs)** | **99.03 %** [98.2–99.7] | 97.98 % [94.6–100] | **96.04 %** [91.8–99.1] | 99.62 % | **0.970** [0.944–0.991] | **0.999** [0.997–1.000] |

The 12-epoch run was still improving when it stopped. The 25-epoch run uses the
same recipe with a higher epoch cap; val loss bottomed out at epoch 22, and that
checkpoint is the one kept. Neither choice used the test set.

### Confusion matrices (test, n = 621: 520 Normal, 101 TB)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Custom CNN | 519 | 1 | 31 | 70 |
| ResNet50, frozen | 519 | 1 | 16 | 85 |
| ResNet50, fine-tuned (12 epochs) | 520 | 0 | 7 | 94 |
| ResNet50, fine-tuned (25 epochs) | 518 | 2 | 4 | 97 |

### Source-matched subset (test, NLM images only, n = 110)

NLM is the only source that contains **both** classes, so here the model cannot
get the label right just by recognising where an image came from (see
Limitations). Recall drops for every model on this subset.

| Model | Accuracy | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Custom CNN | 80.00 % | 63.79 % | 98.08 % | 0.771 | 0.962 |
| ResNet50, frozen | 92.73 % | 86.21 % | 100 % | 0.926 | 0.991 |
| ResNet50, fine-tuned (12 epochs) | 94.55 % | 89.66 % | 100 % | 0.946 | 0.992 |
| ResNet50, fine-tuned (25 epochs) | 96.36 % | 93.10 % | 100 % | 0.964 | 0.997 |

Correct predictions per source and class on test:

| Model | RSNA Normal | NLM Normal | Belarus TB | NLM TB |
|---|---|---|---|---|
| Custom CNN | 468/468 | 51/52 | 33/43 | 37/58 |
| ResNet50, frozen | 467/468 | 52/52 | 35/43 | 50/58 |
| ResNet50, fine-tuned (12 epochs) | 468/468 | 52/52 | 42/43 | 52/58 |
| ResNet50, fine-tuned (25 epochs) | 466/468 | 52/52 | 43/43 | 54/58 |

### What changed for the fine-tuned ResNet50

| | Notebook recipe | Fine-tuned |
|---|---|---|
| Input preprocessing | `/255` | ImageNet mean/std normalisation (what the pretrained weights expect) |
| Phase 1 | n/a | 3 epochs training only the head, backbone frozen, Adam lr 1e-3 |
| Phase 2 | head only, 10 epochs, Adam lr 1e-4 | `layer3` + `layer4` (conv4_x, conv5_x) unfrozen; AdamW, backbone lr **1e-5**, head lr 1e-4, weight decay 1e-4; up to 12 or 25 epochs, early stopping with patience 4 on val loss |
| BatchNorm in frozen stages | n/a | kept in inference mode |
| Checkpoint | last epoch | lowest val loss (12-epoch run: epoch 12; 25-epoch run: epoch 22) |

The custom CNN and the frozen ResNet50 keep the notebook's settings:
- architecture
- `/255` input
- Adam lr 1e-4
- 15 and 10 epochs
- augmentation (rotation 15°, shift and zoom 0.1, shear, horizontal flip)

Every model uses the same split, the same augmentation, batch size 32 and seed 42.

### Reading these numbers

- **Accuracy hides missed TB cases.** The custom CNN scores about 95 %
  accuracy but misses **31 % of TB cases**. For a screening tool, the number
  that matters is sensitivity (recall), not accuracy.
- **The fine-tuned ResNet50 is the strongest model.** Its recall and F1
  confidence intervals do not overlap the custom CNN's. Training it longer
  (25 epochs vs 12) gained 3 more TB detections at the cost of 2 false
  positives. The two runs' confidence intervals overlap, so the gap between
  them is small.
- **At 0.5, almost every error is a missed TB case.** The next section moves the
  threshold, choosing it on val only.
- **The PyTorch port does not reproduce the Keras collapse.** In the port, the
  frozen ResNet50 trained on `/255` input still learns (AUC 0.986). The Keras
  collapse comes from Keras's caffe-style preprocessing, where `/255` inputs
  are nearly constant after mean subtraction. torchvision's weights are less
  sensitive to the scale mismatch. So the "frozen" row shows what that recipe
  achieves when the input scaling is not catastrophically wrong. It is **not**
  a re-run of the exact Colab model.

### Decision thresholds chosen on val (`src/threshold.py`)

At 0.5 every model favours specificity. `threshold.py` picks thresholds using
**val** predictions only, then applies them to test:

- **sens95:** the highest threshold that reaches ≥ 95 % TB recall on val. This
  is a screening-style operating point.
- **Youden:** the threshold that maximises sensitivity + specificity − 1 on val.

| Model | Rule | Threshold | Test recall | Test specificity | Test F1 | FN / FP | NLM-only recall |
|---|---|---|---|---|---|---|---|
| Custom CNN | 0.5 | 0.500 | 69.3 % | 99.8 % | 0.814 | 31 / 1 | 63.8 % |
| Custom CNN | Youden | 0.217 | 83.2 % | 99.6 % | 0.898 | 17 / 2 | 79.3 % |
| Custom CNN | sens95 | 0.010 | 94.1 % | 85.6 % | 0.701 | 6 / 75 | 91.4 % |
| ResNet50, frozen | Youden | 0.278 | 91.1 % | 98.1 % | 0.906 | 9 / 10 | 93.1 % |
| ResNet50, frozen | sens95 | 0.097 | 95.0 % | 91.4 % | 0.793 | 5 / 45 | 94.8 % |
| ResNet50, fine-tuned (12 epochs) | Youden | 0.151 | 96.0 % | 99.6 % | 0.970 | 4 / 2 | 94.8 % |
| **ResNet50, fine-tuned (25 epochs)** | **sens95 = Youden** | **0.168** | **97.0 %** | **99.6 %** | **0.975** | **3 / 2** | **94.8 %** |

- The custom CNN can only reach screening-level sensitivity by accepting 75
  false positives.
- The fine-tuned ResNet50 reaches 97 % sensitivity with 2 false positives.
- These thresholds come from 625 val images that share the test set's sources.
  On data from another hospital they would need to be chosen again.

Output: `reports/thresholds.json`.

## Corrected Colab notebook (`notebooks/BONE_TB_fixed.ipynb`)

This is a Keras/TensorFlow version of the pipeline above, so the project can
stay in Colab. It does the following:

- uses the same duplicate grouping and a grouped 70/15/15 split
- runs the same custom CNN
- trains ResNet50 with `preprocess_input` and fine-tunes conv4_x and conv5_x
  at lr 1e-5, with BatchNorm frozen
- evaluates on test once, including the NLM-only subset
- assigns the Kaggle upload result to a variable, so the API key is not printed
  into the notebook

Notes:
- `notebooks/make_notebook.py` regenerates the notebook.
- It was smoke-tested end to end with TensorFlow 2.21 on a 5 % subsample at
  1 epoch. That checks the code runs; its metrics are meaningless. The full
  run has not been done.
- Its group IDs are numbered differently from `make_split.py`, so its split
  differs slightly (630 / 2940 / 630). The numbers in this README come from the
  PyTorch scripts.

## Limitations

1. **Source confounding (the biggest caveat).**
   - 88 % of Normal images come from RSNA (US hospitals). Every TB image comes
     from Belarus or NLM.
   - A model can therefore score well by recognising the *source* (scanner,
     processing, burned-in markers, crop) rather than TB itself.
   - Evidence that this is happening: all three models classify RSNA Normals
     almost perfectly, while recall on NLM TB, the only source with both
     classes, is the weakest result.
   - The headline test numbers are therefore **optimistic**. The NLM-only rows
     are a better guide, but with n = 110 their confidence intervals are wide.
2. **No patient IDs.**
   - Splitting is grouped by image similarity, not by patient. Repeat studies
     of one patient that look sufficiently different (correlation < 0.955)
     can still fall into different splits.
   - The RSNA images come from NIH ChestX-ray14, which contains many follow-up
     studies per patient, so some leakage of this kind is likely.
3. **No external validation.**
   - All splits come from one curated Kaggle dataset.
   - Performance on images from other hospitals, scanners or populations
     (for example Shenzhen-only, TBX11K or local clinical data) is unknown.
     It is likely lower, given limitation 1.
4. **A single split and a single seed.**
   - The confidence intervals cover sampling of the test set, but not
     variance from training or from how the data was split.
   - Repeated grouped cross-validation would give more reliable estimates.
5. **Thresholds are not calibrated.** The thresholds above come from 625 val
   images (about 104 TB) drawn from the same sources as test. The predicted
   probabilities are not calibrated, so a threshold of 0.168 does not mean a
   16.8 % TB risk.
6. **Labels.** Labels are image-level and inherited from the source datasets.
   There is no radiologist re-read and no localisation ground truth, so Grad-CAM
   maps from the notebook cannot be checked.
7. **Questionable augmentation.** Horizontal flipping (kept from the notebook to
   allow comparison) produces anatomically impossible images, with the heart on
   the right.
8. **Framework port.**
   - Models were retrained in PyTorch (on an Apple M2 with MPS), not Keras.
   - The architectures and hyperparameters match the notebook, but weights,
     initialisation and the ImageNet checkpoint differ (torchvision
     `IMAGENET1K_V2`).
   - The original Colab `.h5` models were not evaluated, because they were
     trained on all 4200 images and no unseen data exists for them.
9. **Not a medical device.** This is a research and educational prototype. It
   must not be used for diagnosis.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install torch torchvision imagehash scikit-learn pandas openpyxl pillow
kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset -p data
unzip -q data/tuberculosis-tb-chest-xray-dataset.zip -d data/raw

.venv/bin/python src/hash_audit.py          # step 1: hashes + duplicate pairs/groups
.venv/bin/python src/make_split.py          # step 2: grouped split (asserts no cross-split dups)
.venv/bin/python src/train_eval.py cnn            # step 3
.venv/bin/python src/train_eval.py resnet_frozen  # step 3
.venv/bin/python src/train_eval.py resnet_ft      # step 4 (12 epochs)
.venv/bin/python src/train_eval.py resnet_ft25    # step 4 (25 epochs)
cd src && ../.venv/bin/python threshold.py cnn resnet_frozen resnet_ft resnet_ft25
```

Approximate runtimes on an M2 (8 GB): hashing 30 s, CNN ~5 min,
frozen ResNet50 ~9 min, fine-tuned ResNet50 ~21 min (12 epochs) or ~35 min
(25 epochs).

Each model writes:
- `reports/metrics_<model>.json`: test metrics, 95 % confidence intervals,
  NLM-only metrics and the selected checkpoint
- `reports/preds_<model>.csv`: per-image test probabilities
- `reports/log_<model>.txt`: training log

## Citation

T. Rahman et al., "Reliable Tuberculosis Detection using Chest X-ray with Deep
Learning, Segmentation and Visualization," *IEEE Access*, vol. 8,
pp. 191586–191601, 2020. doi:10.1109/ACCESS.2020.3031384
