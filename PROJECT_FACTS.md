# PROJECT_FACTS — OsteoDetect (TB chest X-ray classifier)

Rules followed:
- Every number cites the file and line or notebook cell it comes from.
- `@521f48c` means the file as of the first commit. The original notebook's
  outputs exist only there, because they were cleared in `398b052`.
- "computed from `<file>`" means the number was counted directly from that saved
  output file, for example by counting rows. No other calculation was done.
- **NOT FOUND** means the repo contains no evidence for the item.

---

## 1. One-line summary

The project classifies frontal chest X-rays as **Tuberculosis vs Normal**, and
re-audits an earlier hackathon notebook for data leakage (`README.md:6`).

- **Who it is for:** NOT FOUND. The README only says it is "a research and
  educational prototype" and must not be used for diagnosis (`README.md:220`).
- **An earlier claim:** the first README described the task as "bone
  tuberculosis from pulmonary TB" (`README.md@c189f07:3`). The code does not
  support this. The two classes are the folders `Normal` and `Tuberculosis`
  (`src/hash_audit.py:32`), and the dataset metadata attributes the images to
  chest-X-ray sources.

## 2. Data

| Item | Fact | Source |
|---|---|---|
| Dataset | Kaggle `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` (TB Chest Radiography Database) | `BONE_TB.ipynb` cell 3; `notebooks/BONE_TB_fixed.ipynb` cell 2 |
| Images | 4200 | computed from `reports/image_index.csv` (4200 data rows) |
| Class counts | Normal 3500 (RSNA 3094, NLM 406); TB 700 (Belarus 306, NLM 394) | computed from `reports/image_index.csv`, columns `label` and `source` |
| Where `source` comes from | The `URL` column of each class's `.metadata.xlsx` file | `src/hash_audit.py:33-38` |
| Native resolution | NOT FOUND. The dataset's own README is in `data/`, which is git-ignored (`.gitignore`). | n/a |
| Channels | Loaded as 1-channel grayscale (`convert("L")`), then repeated to 3 channels | `src/train_eval.py:52`, `src/train_eval.py:60` |
| Model input size | Resized to 224×224 (bilinear) and cached as uint8 | `src/train_eval.py:52` |
| Patient IDs | None. The index columns are path, file, label, source, md5_file, md5_pixels, phash, dhash. | `reports/image_index.csv:1` |

**How the split was made** (`src/make_split.py`):

- Groups come from the duplicate audit: thumbnail correlation ≥ 0.955
  (`src/hash_audit.py:27`), joined into connected groups (`src/hash_audit.py:85-96`).
- `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)`
  (`src/make_split.py:18`, `src/make_split.py:28`), stratified on label × source.
- 3 folds become test, 3 become val and 14 become train (`src/make_split.py:32`).

| Split | Images | Normal (NLM / RSNA) | TB (Belarus / NLM) |
|---|---|---|---|
| train | 2954 | 291 / 2168 | 216 / 279 |
| val | 625 | 63 / 458 | 47 / 57 |
| test | 621 | 52 / 468 | 43 / 58 |

All rows above are computed from `reports/splits.csv`.

**Preprocessing and augmentation:**

| Model | Input preprocessing | Source |
|---|---|---|
| CNN and frozen ResNet50 | `/255` scaling only | `src/train_eval.py:185`, `src/train_eval.py:48` |
| Fine-tuned ResNet50 | ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)` | `src/train_eval.py:34`, `src/train_eval.py:47` |

- Training augmentation: `RandomAffine(degrees=15, translate=(0.1, 0.1),
  scale=(0.9, 1.1), shear=0.1)` plus `RandomHorizontalFlip()`
  (`src/train_eval.py:45-46`).
- The original notebook used `rescale=1./255` with `rotation_range=15`,
  `zoom_range=0.1`, `width_shift_range=0.1`, `height_shift_range=0.1`,
  `shear_range=0.1` and `horizontal_flip=True` (`BONE_TB.ipynb` cell 8).

### Leakage checks

**Do duplicate images exist?** Yes, within the dataset. **Do any cross splits?**
In the current split, none that the audit detected.

- Duplicate audit: 40 pairs (computed from `reports/duplicate_pairs.csv`):

  | Kind | Pairs |
  |---|---|
  | exact_file | 3 |
  | exact_pixels | 3 |
  | near_duplicate (correlation ≥ 0.985, `src/hash_audit.py:26`) | 7 |
  | likely_same_patient (0.955 ≤ correlation < 0.985) | 27 |

- The pairs form 31 groups covering 68 images (computed from
  `reports/dup_groups.csv`). Pairs whose labels differ: 0 (computed from
  `reports/duplicate_pairs.csv`).
- **In the current split** (computed from `reports/splits.csv` and
  `reports/duplicate_pairs.csv`):
  - duplicate pairs that span two splits: **0**
  - file MD5s that span two splits: 0
  - pixel MD5s that span two splits: 0
  - filenames shared by train and test: 0
  - `src/make_split.py:37-43` asserts all of the above, so the split script
    fails if any check is violated.
- **Caveat:** the audit only catches pairs above its thresholds. Same-patient
  images with correlation below 0.955 would not be detected.
- **In the original notebook:** `train_dir` and `val_dir` point to the same
  folder (`BONE_TB.ipynb` cell 6), and both generators print
  `Found 4200 images belonging to 2 classes` (`BONE_TB.ipynb@521f48c` cell 8
  output). Its "validation" set was therefore identical to its training set:
  100 % overlap.

**Does a separate test set exist?**

- **In the current pipeline: yes.** 621 images held out (`reports/splits.csv`).
  All four `reports/preds_*.csv` files contain only `split == test` rows, 621
  each (computed). The checkpoint is chosen by lowest val loss
  (`src/train_eval.py:143-146`) and scored on test afterwards
  (`src/train_eval.py:217-225`).
- **In the original notebook: no.** Only `train_dir` and `val_dir` exist, and
  they are the same folder (`BONE_TB.ipynb` cell 6).

## 3. Method

**Custom CNN** (`src/train_eval.py:65-79`), a port of `BONE_TB.ipynb` cell 10:

- Layers: Conv(32, 3×3) → MaxPool → Conv(64, 3×3) → MaxPool → Conv(128, 3×3)
  → MaxPool → Flatten (128·26·26) → Dense 128 → Dropout 0.5 → 1
  (`src/train_eval.py:71-75`)
- Optimiser: Adam, lr 1e-4 (`src/train_eval.py:193`)
- Epochs: 15 (`src/train_eval.py:194`)

**ResNet50 backbone** (shared by both ResNet50 variants):

- torchvision `IMAGENET1K_V2` weights (`src/train_eval.py:86`)
- Head: Linear 2048→128 → ReLU → Dropout 0.5 → Linear 1 (`src/train_eval.py:88`)

**ResNet50, frozen (the notebook's recipe):**

- Backbone frozen (`src/train_eval.py:197`)
- Head only, Adam lr 1e-4 (`src/train_eval.py:198`), 10 epochs (`src/train_eval.py:199`)

**ResNet50, fine-tuned:**

- Phase 1: head only, Adam lr 1e-3 (`src/train_eval.py:205`), 3 epochs
  (`src/train_eval.py:206`)
- Phase 2: `layer3` and `layer4` unfrozen (`src/train_eval.py:209`); AdamW
  with backbone lr 1e-5 and head lr 1e-4, weight decay 1e-4
  (`src/train_eval.py:211-212`); up to 12 epochs (`resnet_ft`) or 25 epochs
  (`resnet_ft25`) (`src/train_eval.py:201`); early stopping with patience 4
  (`src/train_eval.py:213`)
- BatchNorm in frozen stages is kept in eval mode (`src/train_eval.py:104-108`)

**Shared settings:**

- Loss: `BCEWithLogitsLoss` (`src/train_eval.py:124`)
- Batch size: 32 (`src/train_eval.py:186`)
- Seed: 42 (`src/train_eval.py:33`, `src/train_eval.py:181-182`)
- Device: MPS if available, otherwise CPU (`src/train_eval.py:32`)

**Checkpoint selection:** the lowest val loss across all epochs and phases
(`src/train_eval.py:143-146`). The selected checkpoints were:

| Model | Selected checkpoint | Val loss | Source |
|---|---|---|---|
| CNN | `cnn:15` | 0.1278 | `reports/metrics_cnn.json:3-4` |
| ResNet50, frozen | `frozen:10` | 0.1157 | `reports/metrics_resnet_frozen.json:3-4` |
| ResNet50, fine-tuned 12 | `ft-unfrozen:12` | 0.0255 | `reports/metrics_resnet_ft.json:3-4` |
| ResNet50, fine-tuned 25 | `ft-unfrozen:22` | 0.0089 | `reports/metrics_resnet_ft25.json:3-4` |

**Threshold rules** (`src/threshold.py`). Both use val predictions only:

- `sens95`: the highest threshold whose val recall is ≥ 0.95
  (`src/threshold.py:22`, `src/threshold.py:35-36`)
- `youden`: the threshold that maximises TPR − FPR on val (`src/threshold.py:37`)

## 4. Results

**Evaluation setup** (`src/train_eval.py:154-175`, `src/train_eval.py:224`):

- Split: the held-out **test** split (n = 621)
- Task: binary, TB is the positive class; threshold 0.5 unless stated
- Averaging: none; precision, recall and F1 are for the TB class only
- Confidence intervals: 95 % percentile bootstrap, 2000 resamples of the test set
- Seeds: **1** (seed 42), one training run per model

**Main results, test set, threshold 0.5:**

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | TN / FP / FN / TP | Source |
|---|---|---|---|---|---|---|---|---|
| Custom CNN | 0.9485 | 0.9859 | 0.6931 | 0.9981 | 0.8140 | 0.9751 | 519 / 1 / 31 / 70 | `reports/metrics_cnn.json:6-17` |
| ResNet50, frozen | 0.9726 | 0.9884 | 0.8416 | 0.9981 | 0.9091 | 0.9856 | 519 / 1 / 16 / 85 | `reports/metrics_resnet_frozen.json:6-17` |
| ResNet50, fine-tuned 12 | 0.9887 | 1.0000 | 0.9307 | 1.0000 | 0.9641 | 0.9951 | 520 / 0 / 7 / 94 | `reports/metrics_resnet_ft.json:6-17` |
| ResNet50, fine-tuned 25 | 0.9903 | 0.9798 | 0.9604 | 0.9962 | 0.9700 | 0.9990 | 518 / 2 / 4 / 97 | `reports/metrics_resnet_ft25.json:6-17` |

**95 % bootstrap confidence intervals** (lines 20-40 of each `reports/metrics_*.json`):

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Custom CNN | 0.931–0.965 | 0.604–0.778 | 0.745–0.872 | 0.958–0.989 |
| ResNet50, frozen | 0.960–0.986 | 0.765–0.910 | 0.861–0.951 | 0.973–0.995 |
| ResNet50, fine-tuned 12 | 0.981–0.997 | 0.879–0.978 | 0.936–0.989 | 0.986–1.000 |
| ResNet50, fine-tuned 25 | 0.982–0.997 | 0.918–0.991 | 0.944–0.991 | 0.997–1.000 |

**Source-matched subset: test images from NLM only, n = 110** (lines 42-54 of
each `reports/metrics_*.json`). NLM is the only source with both classes:

| Model | Accuracy | Recall | Specificity | F1 | ROC-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|
| Custom CNN | 0.8000 | 0.6379 | 0.9808 | 0.7708 | 0.9619 | 51 / 1 / 21 / 37 |
| ResNet50, frozen | 0.9273 | 0.8621 | 1.0000 | 0.9259 | 0.9914 | 52 / 0 / 8 / 50 |
| ResNet50, fine-tuned 12 | 0.9455 | 0.8966 | 1.0000 | 0.9455 | 0.9917 | 52 / 0 / 6 / 52 |
| ResNet50, fine-tuned 25 | 0.9636 | 0.9310 | 1.0000 | 0.9643 | 0.9973 | 52 / 0 / 4 / 54 |

**Correct predictions by source and class, test, threshold 0.5** (computed
from `reports/preds_*.csv`):

| Model | RSNA Normal | NLM Normal | Belarus TB | NLM TB |
|---|---|---|---|---|
| Custom CNN | 468/468 | 51/52 | 33/43 | 37/58 |
| ResNet50, frozen | 467/468 | 52/52 | 35/43 | 50/58 |
| ResNet50, fine-tuned 12 | 468/468 | 52/52 | 42/43 | 52/58 |
| ResNet50, fine-tuned 25 | 466/468 | 52/52 | 43/43 | 54/58 |

**Thresholds chosen on val, results on test** (`reports/thresholds.json`):

| Model | Rule | Threshold | Test recall | Test specificity | Test F1 | FN / FP | Line |
|---|---|---|---|---|---|---|---|
| Custom CNN | sens95 | 0.0099 | 0.9406 | 0.8558 | 0.7011 | 6 / 75 | 51-52 |
| Custom CNN | youden | 0.2165 | 0.8317 | 0.9962 | 0.8984 | 17 / 2 | 99-100 |
| ResNet50, frozen | sens95 | 0.0966 | 0.9505 | 0.9135 | 0.7934 | 5 / 45 | 197-198 |
| ResNet50, frozen | youden | 0.2775 | 0.9109 | 0.9808 | 0.9064 | 9 / 10 | 245-246 |
| ResNet50, fine-tuned 12 | sens95 | 0.4508 | 0.9406 | 0.9962 | 0.9596 | 6 / 2 | 343-344 |
| ResNet50, fine-tuned 12 | youden | 0.1514 | 0.9604 | 0.9962 | 0.9700 | 4 / 2 | 391-392 |
| ResNet50, fine-tuned 25 | sens95 | 0.1683 | 0.9703 | 0.9962 | 0.9751 | 3 / 2 | 489-490 |
| ResNet50, fine-tuned 25 | youden | 0.1683 | 0.9703 | 0.9962 | 0.9751 | 3 / 2 | 537-538 |

**Ablations present:**

- *Frozen vs fine-tuned backbone, plus input normalisation:* the frozen and
  fine-tuned ResNet50 rows. These two variants differ in several ways at once
  (preprocessing, which layers train, learning rate, number of epochs), so the
  comparison does not isolate the effect of any single change.
- *Training length:* fine-tuned 12 vs 25 epochs. The settings are otherwise
  identical (`src/train_eval.py:201`).
- *Threshold rule:* the table above.

Ablations not present: any single-factor ablation, multiple seeds, and
cross-validation. NOT FOUND.

**The original notebook's reported numbers.** These are not comparable with
the results above, because they were measured on the training data:

| Model | Reported number | Source |
|---|---|---|
| CNN | "Validation Accuracy: 95.69%" | `BONE_TB.ipynb@521f48c` cell 13 output |
| ResNet50 | "Validation Accuracy (ResNet50): 83.33%", with `val_accuracy` 0.8333 in every logged epoch | `BONE_TB.ipynb@521f48c` cell 21 and cell 20 outputs |

The first README's "Test accuracy: ~92.69%" (`README.md@c189f07:9`) appears in
no notebook output. Its source is NOT FOUND.

**Keras notebook (`notebooks/BONE_TB_fixed.ipynb`):** no saved results. Its
outputs are empty (`notebooks/make_notebook.py:7-8`). A smoke test is claimed
in `README.md:153`, but the repo contains no output or log from it.

## 5. Engineering

**Deployment:**

- No API, dashboard or hosted demo exists. NOT FOUND.
- The only interactive entry point is an "Open in Colab" badge for
  `notebooks/BONE_TB_fixed.ipynb` (`README.md:3`).
- Trained weights are not in the repo: `checkpoints/` is git-ignored
  (`.gitignore`).
- The original notebook saved `.h5` files to the Colab runtime
  (`BONE_TB.ipynb` cells 14 and 22). Those files are not in the repo.

**Tests:**

- Unit or integration test files: **0**. `git ls-files` finds none.
- Pass rate: not applicable.
- The only runtime checks are 3 `assert` statements in `src/make_split.py:37-43`.
  They held for the committed `reports/splits.csv`: zero violations, computed
  above.

**Latency and throughput:**

- **Inference latency, PyTorch models:** NOT FOUND.
- **Inference latency, original Keras model:** one image took "29ms/step"
  (`BONE_TB.ipynb@521f48c` cell 16 output). The hardware is NOT FOUND; the
  notebook metadata only says GPU T4 (`BONE_TB.ipynb` metadata).
- **Training wall-clock time per epoch, from the logs:**

  | Run | Epoch times | Source |
  |---|---|---|
  | CNN | 15–38 s | `reports/log_cnn.txt:1-15` |
  | ResNet50, frozen | 45–63 s | `reports/log_resnet_frozen.txt` |
  | ResNet50, fine-tuned 12 | 45–96 s | `reports/log_resnet_ft.txt:1-15` |
  | ResNet50, fine-tuned 25 | 52 s to 10,925 s | `reports/log_resnet_ft25.txt:16` (the 10,925 s epoch) |

  The 25-epoch run's times vary so much that they do not measure throughput.
  The device is not printed in any log; hardware is stated only in a comment
  (`requirements.txt:1`, "Apple M2, PyTorch MPS backend").

**Notable design decisions:**

- Duplicate groups stand in for patient IDs (`src/make_split.py:3-7`).
- The split is stratified on label × source, so every split keeps the same
  source mix (`src/make_split.py:25`).
- An NLM-only test subset is reported because it is the only source with both
  classes (`src/train_eval.py:221-225`).
- The README results tables are generated from the saved reports
  (`src/results_table.py`), not typed by hand.
- Package versions are pinned (`requirements.txt`).
- The Colab notebook assigns the `files.upload()` result to a variable, so the
  credentials file contents are not printed (`notebooks/make_notebook.py:38-40`).

## 6. Surprising findings

1. **The original ResNet50 predicted only "Normal".** Its `val_accuracy` was
   0.8333 in every logged epoch (`BONE_TB.ipynb@521f48c` cell 20 output), and
   3500 / 4200 = 0.8333 (computed from `reports/image_index.csv`). The
   pretrained model scored *below* the custom CNN's 95.69 % (cell 13 output),
   on the same leaked data.
2. **The original CNN's reported number seems to come from epoch 7, not 15.**
   The training log in cell 12 stops partway through "Epoch 8/15"
   (`BONE_TB.ipynb@521f48c` cell 12 output). Epoch 7's `val_accuracy` of
   0.9569 equals the 95.69 % reported in cell 13. Whether training was
   interrupted is NOT FOUND.
3. **In the PyTorch port, the frozen ResNet50 learned.** Its test ROC-AUC is
   0.9856 (`reports/metrics_resnet_frozen.json:12`), so the Keras collapse did
   not reproduce. The two frameworks use different pretrained weights and input
   handling (`src/train_eval.py:86`).
4. **The frozen ResNet50 did better on the harder NLM subset than overall.**
   Its NLM-only recall is 0.8621 (`reports/metrics_resnet_frozen.json:46`)
   against 0.8416 overall (`reports/metrics_resnet_frozen.json:9`). The other
   three models all lose recall on NLM.
5. **The 12-epoch run's threshold rules backfired on test.** The `sens95`
   threshold (0.4508) gave a test recall of 0.9406, below the 0.95 target and
   worse than the Youden threshold's 0.9604 (`reports/thresholds.json:343-344`,
   `reports/thresholds.json:391-392`).
6. **The two fine-tuning runs are identical where they overlap.** The first 15
   epoch lines (3 warm-up + 12 fine-tuning) of `reports/log_resnet_ft.txt` and
   `reports/log_resnet_ft25.txt` match exactly, apart from the timings
   (computed with `diff`). Seed 42 gave a reproducible run on this hardware.
7. **The fine-tuned 12-epoch model was still improving when it stopped.** It
   was selected at its final epoch, `ft-unfrozen:12`
   (`reports/metrics_resnet_ft.json:3`). The 25-epoch run kept improving val
   loss until epoch 22 (`reports/metrics_resnet_ft25.json:3`).

## 7. Limitations: what the evaluation does NOT prove

- **That the models detect TB rather than the image source.**
  - Normal images are 88 % RSNA (3094 / 3500), and TB images are only Belarus
    or NLM (computed from `reports/image_index.csv`).
  - RSNA Normals are 466–468 / 468 correct for every model (computed from
    `reports/preds_*.csv`).
  - The NLM-only check has just 110 images (`reports/metrics_*.json:43`), of
    which 58 are TB (computed from `reports/splits.csv`).
- **Patient-level separation.** There are no patient IDs, and the duplicate
  audit only catches pairs above its thresholds (`src/hash_audit.py:26-27`).
- **Generalisation to other hospitals or datasets.** No external test set
  exists. NOT FOUND.
- **Stability across seeds or splits.** There is 1 seed and 1 split; the
  confidence intervals cover test-set sampling only (`src/train_eval.py:166`).
- **Calibrated probabilities.** No calibration code exists. NOT FOUND.
- **Clinical performance or reader agreement.** There is no radiologist
  comparison and no localisation ground truth. NOT FOUND.
- **The Keras notebook's performance.** It has no saved full-run outputs.
- **Grad-CAM faithfulness.** The original notebook's Grad-CAM cell (cell 18)
  appears *before* the ResNet50 is trained (cell 20). The notebook records no
  execution counts (every `execution_count` is `null` in
  `BONE_TB.ipynb@521f48c`), so whether the heatmap used a trained model is
  NOT FOUND.

## 8. Tech stack: libraries actually imported

| Where | Libraries | Source |
|---|---|---|
| `src/` (PyTorch pipeline) | torch, torchvision (`transforms.v2`, `models.resnet50`), numpy, pandas, scikit-learn (`StratifiedGroupKFold`, `metrics`, `roc_curve`), Pillow, imagehash; stdlib hashlib, json, sys, time, pathlib | `src/hash_audit.py:8-15`, `src/make_split.py:10-14`, `src/train_eval.py:13-27`, `src/threshold.py:11-20`, `src/results_table.py:6-10` |
| Excel reading | openpyxl, used indirectly as the engine behind `pd.read_excel` | `src/hash_audit.py:33` |
| `notebooks/BONE_TB_fixed.ipynb` | tensorflow / keras (`ImageDataGenerator`, `ResNet50`, `preprocess_input`), numpy, pandas, scikit-learn, Pillow, google.colab; stdlib os, hashlib | `notebooks/make_notebook.py:28-31`, `:39`, `:106-107`, `:127`, `:143` |
| `BONE_TB.ipynb` (original) | tensorflow / keras, matplotlib, numpy, opencv (`cv2`, used in Grad-CAM), google.colab; stdlib zipfile, os | `BONE_TB.ipynb` cells 0, 11, 15, 17, 18 |
| Data download | kaggle CLI (a shell command, not imported) | `BONE_TB.ipynb` cell 3; `notebooks/BONE_TB_fixed.ipynb` cell 2 |

Notes:
- `zipfile` and `matplotlib.cm` are imported in the original notebook
  (cells 0 and 18) but never used. NOT FOUND in use.
- `scipy` and `PyWavelets` are pinned in `requirements.txt` but never imported
  directly. They are dependencies of imagehash.

## 9. Timeline

| | Date | Commit |
|---|---|---|
| First commit | 2026-05-11 23:27:52 +0530 | `521f48c` "Add files via upload" |
| Last commit | 2026-11-12 18:15:29 +0530 | `abbaf59` "Restructure README, generate results tables from reports, pin requirements" |

The repo has 6 commits in total (`git log`). The first README says
"Hackathon project, 2025" (`README.md@c189f07:5`). The date of the hackathon
itself is NOT FOUND in the commit history.

## 10. Visuals

**In the current tree:** no image files. A search for png, jpg, jpeg, svg and
gif finds none outside the git-ignored `data/`.

**Only in git history** (`BONE_TB.ipynb@521f48c`; outputs cleared in `398b052`):

| Path | What it shows | Caveat |
|---|---|---|
| `BONE_TB.ipynb@521f48c`, cell 16 output (`image/png`) | An uploaded file "TB 2.jpeg" with the prediction "TB Negative (Confidence: 0.29)" | The true label is NOT FOUND (only the filename suggests TB). It comes from the leaked-split model. |
| `BONE_TB.ipynb@521f48c`, cell 18 output (`image/png`) | A Grad-CAM heatmap on an uploaded "TB 1.jpg" | Whether the model was trained at that point is NOT FOUND (see Limitations). |

**Suggestion for a LinkedIn post:** neither existing image is suitable. The
first shows a model from the leaked split, and the second may show an untrained
model. The two best visuals would have to be **made** from saved outputs. They
do not exist yet:

1. **Before / after bar chart:** the notebook's 95.69 % "validation" accuracy
   (cell 13, measured on training data) next to the held-out test recall for
   each model (`reports/metrics_*.json:9`). It tells the story "accuracy looked
   great, recall told the truth".
2. **Per-source heatmap:** correct predictions by source and class (computed
   from `reports/preds_*.csv`). It shows the RSNA shortcut and the weaker NLM
   TB results at a glance.
