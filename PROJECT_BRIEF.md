# Project brief: OsteoDetect (TB chest X-ray classifier)

**How to use this brief.** Everything below was checked against the project's
code, saved results and git history. Use only these facts.
- Do not round numbers differently or add metrics that are not listed here.
- **NOT FOUND** means the project has no evidence for that item. Do not fill
  it in, estimate it or imply it.
- Repo: https://github.com/aarohi1822/OsteoDetect

---

## Answers to two direct questions

**1. Are there duplicate images across the train, val and test splits?**

- **In the current pipeline: no detected duplicates cross splits.**
  - An image-hashing audit found 40 duplicate or same-patient pairs inside the
    dataset (31 groups, 68 images).
  - Each group was kept inside a single split.
  - 0 pairs, 0 exact file copies and 0 exact pixel copies span two splits. The
    split script asserts this and fails otherwise.
  - Caveat: same-patient images that look less alike than the audit's
    threshold (thumbnail correlation below 0.955) would not be detected.
- **In the original hackathon notebook: 100 % overlap.** The training and
  validation folders were the same folder of 4200 images.

**2. Does a separate test set exist?**

- **In the current pipeline: yes.** 621 images are held out. Checkpoints were
  chosen on val, and test was scored once.
- **In the original notebook: no.** It had only train and "val", and those
  were identical.

---

## 1. One-line summary

A binary classifier for chest X-rays (**Tuberculosis vs Normal**), plus a
leakage audit and re-evaluation of an earlier hackathon notebook that had
reported inflated accuracy.

- **Intended users:** NOT FOUND. The project calls itself "a research and
  educational prototype", not for diagnosis.
- **Not bone TB:** an early README described the task as "bone TB vs pulmonary
  TB". That is wrong: the classes are Normal vs TB chest X-rays.

## 2. Data

- **Dataset:** TB Chest Radiography Database (Kaggle,
  `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`).
- **Size:** 4200 images.
  - Normal: 3500 (3094 from RSNA, 406 from NLM)
  - TB: 700 (306 from Belarus, 394 from NLM)
- **Channels and size:** loaded as grayscale, repeated to 3 channels, resized
  to 224×224. Native resolution: NOT FOUND in the repo.
- **Patient IDs:** none exist in the dataset.

**How the split was made:**

- Duplicate and same-patient groups stand in for patient IDs.
- `StratifiedGroupKFold` with 20 folds, seed 42, stratified on label × image
  source. 3 folds became test, 3 val and 14 train.

| Split | Images | Normal (NLM / RSNA) | TB (Belarus / NLM) |
|---|---|---|---|
| Train | 2954 | 291 / 2168 | 216 / 279 |
| Val | 625 | 63 / 458 | 47 / 57 |
| Test | 621 | 52 / 468 | 43 / 58 |

**Preprocessing:**

- CNN and frozen ResNet50: pixels scaled to 0–1 (`/255`).
- Fine-tuned ResNet50: ImageNet mean/std normalisation.
- Training augmentation for every model: rotation ±15°, translation 10 %,
  scale 0.9–1.1, shear, horizontal flip.

## 3. Method

The models are PyTorch ports of the original Keras notebook.

| Model | Architecture | Training |
|---|---|---|
| Custom CNN | 3 conv blocks (32/64/128 filters, 3×3, max-pool) → Dense 128 → Dropout 0.5 → 1 | Adam lr 1e-4, 15 epochs |
| ResNet50, frozen (notebook recipe) | ImageNet ResNet50, frozen backbone + head (Dense 128 → Dropout 0.5 → 1) | Head only, Adam lr 1e-4, 10 epochs, `/255` input |
| ResNet50, fine-tuned | Same backbone and head | 3 epochs head only (lr 1e-3), then the top 2 stages unfrozen, AdamW (backbone lr 1e-5, head lr 1e-4, weight decay 1e-4); run twice, capped at 12 and at 25 epochs; early stopping with patience 4; BatchNorm frozen in the frozen stages |

**Shared settings:** binary cross-entropy loss, batch size 32, seed 42.

**Model selection:** the checkpoint with the lowest val loss. The selected
epochs were CNN 15, frozen 10, fine-tuned-12 epoch 12, fine-tuned-25 epoch 22.

**Threshold tuning, val only:**
- `sens95`: the highest threshold with ≥ 95 % val recall
- `Youden`: the threshold maximising sensitivity + specificity − 1

## 4. Results

**Setup:**
- Held-out test set: 621 images (520 Normal, 101 TB)
- Binary task, TB is the positive class
- Precision, recall and F1 are for the TB class (no averaging)
- Threshold 0.5 unless stated
- 95 % bootstrap confidence intervals, 2000 resamples
- **One seed, one run per model**

| Model | Accuracy | Precision | Recall (sensitivity) | Specificity | F1 | ROC-AUC | Missed TB / false alarms |
|---|---|---|---|---|---|---|---|
| Custom CNN | 94.85 % | 98.59 % | **69.31 %** | 99.81 % | 0.814 | 0.975 | 31 / 1 |
| ResNet50, frozen | 97.26 % | 98.84 % | 84.16 % | 99.81 % | 0.909 | 0.986 | 16 / 1 |
| ResNet50, fine-tuned (12 epochs) | 98.87 % | 100 % | 93.07 % | 100 % | 0.964 | 0.995 | 7 / 0 |
| **ResNet50, fine-tuned (25 epochs)** | **99.03 %** | 97.98 % | **96.04 %** | 99.62 % | **0.970** | **0.999** | 4 / 2 |

**95 % confidence intervals for recall:**

| Model | Recall CI |
|---|---|
| CNN | 60.4–77.8 % |
| Frozen | 76.5–91.0 % |
| Fine-tuned 12 | 87.9–97.8 % |
| Fine-tuned 25 | 91.8–99.1 % |

**Source-matched check: NLM images only** (n = 110: 52 Normal, 58 TB). NLM is
the only source that contributes both classes:

| Model | Accuracy | Recall | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|---|
| CNN | 80.0 % | 63.8 % | 98.1 % | 0.771 | 0.962 |
| Frozen | 92.7 % | 86.2 % | 100 % | 0.926 | 0.991 |
| Fine-tuned 12 | 94.5 % | 89.7 % | 100 % | 0.945 | 0.992 |
| Fine-tuned 25 | 96.4 % | 93.1 % | 100 % | 0.964 | 0.997 |

**Correct predictions by source and class** (test, threshold 0.5):

| Model | RSNA Normal | NLM Normal | Belarus TB | NLM TB |
|---|---|---|---|---|
| CNN | 468/468 | 51/52 | 33/43 | 37/58 |
| Frozen | 467/468 | 52/52 | 35/43 | 50/58 |
| Fine-tuned 12 | 468/468 | 52/52 | 42/43 | 52/58 |
| Fine-tuned 25 | 466/468 | 52/52 | 43/43 | 54/58 |

**Thresholds tuned on val, applied to test:**

| Model | Rule | Threshold | Test recall | Test specificity | Missed TB / false alarms |
|---|---|---|---|---|---|
| CNN | sens95 | 0.010 | 94.1 % | 85.6 % | 6 / 75 |
| Frozen | sens95 | 0.097 | 95.0 % | 91.3 % | 5 / 45 |
| Fine-tuned 25 | sens95 = Youden | 0.168 | **97.0 %** | 99.6 % | 3 / 2 |

**What the original notebook reported (invalid):** these were measured on the
training data, so they cannot be compared with the results above.

| Model | Reported |
|---|---|
| CNN | "Validation accuracy" 95.69 % |
| ResNet50 | 83.33 %, which equals 3500/4200: it predicted "Normal" for everything |

An early README's "92.69 % test accuracy" appears in no output. Its source is
NOT FOUND.

**Comparisons in the results above:**
- Frozen vs fine-tuned backbone. These differ in several settings at once, so
  this is not a clean single-factor test.
- 12 vs 25 epochs.
- Threshold rules.

There is no multi-seed run, no cross-validation and no single-factor ablation.

## 5. Engineering

- **Deployment:** no API, dashboard or hosted demo. The only interactive entry
  point is an "Open in Colab" badge for a corrected Keras notebook. That
  notebook has only been smoke-tested on a 5 % sample, and its full-run results
  are NOT FOUND. Trained weights are not published.
- **Tests:** 0 test files. The split script has 3 runtime `assert` checks for
  leakage, and all of them pass on the committed split.
- **Latency:** NOT FOUND for the final models. The original Keras notebook
  printed 29 ms for a single-image prediction (hardware NOT FOUND).
- **Training time per epoch** (local Apple M2, wall clock): CNN 15–38 s,
  frozen ResNet50 45–63 s, fine-tuned ResNet50 45–96 s. The 25-epoch run's
  times are not reliable, because the machine was not dedicated to it.
- **Design decisions:**
  - Duplicate groups as a stand-in for patient IDs
  - A split stratified by source
  - An NLM-only test subset to probe the source shortcut
  - Thresholds tuned on val only
  - README results tables generated from the saved metric files
  - Pinned `requirements.txt`
  - Reproducible runs: the 12-epoch and 25-epoch runs are identical over their
    first 15 epochs

## 6. Surprising findings

1. **The pretrained ResNet50 in the original notebook did worse than the small
   CNN.** It got 83.33 % against 95.69 %, because it collapsed to predicting
   "Normal" for every image. The cause was wrong input preprocessing plus a
   fully frozen backbone.
2. **The original CNN's 95.69 % appears to come from epoch 7 of 15.** Its
   training log stops partway through epoch 8. Whether training was
   interrupted is NOT FOUND.
3. **High accuracy hid poor sensitivity.** The CNN scores 94.85 % accuracy but
   misses 31 % of TB cases.
4. **In the PyTorch port, the frozen ResNet50 learned fine** (ROC-AUC 0.986).
   The Keras collapse did not reproduce.
5. **The frozen ResNet50's recall was higher on the harder NLM subset** than
   overall (86.2 % vs 84.2 %). The other models all dropped.
6. **One threshold rule missed its own target on test.** For the 12-epoch
   model, the "≥ 95 % recall on val" threshold gave only 94.1 % recall on test.

## 7. Limitations: what the evaluation does NOT prove

- **That the model detects TB rather than the hospital the image came from.**
  - 88 % of Normal images come from one source (RSNA), and TB images come only
    from Belarus and NLM.
  - Every model gets RSNA Normals almost perfectly right.
  - The fair NLM-only check has just 110 images.
- **Patient-level separation.** There are no patient IDs, and the duplicate
  audit only catches images that look very similar.
- **Generalisation.** No external dataset or hospital was tested.
- **Stability.** One seed and one split. The confidence intervals cover
  test-set sampling only.
- **Calibrated probabilities, clinical performance and radiologist
  comparison.** None of these were done.
- **Grad-CAM explanations.** The original notebook's Grad-CAM image may have
  been made with an untrained model; this is NOT FOUND either way.

## 8. Tech stack

Only libraries that are actually used:

- **Pipeline:** Python 3.12, PyTorch 2.7.1, torchvision 0.22.1, scikit-learn,
  NumPy, pandas, Pillow, imagehash, openpyxl (Excel metadata), Apple MPS
  backend.
- **Corrected notebook:** TensorFlow 2.21 / Keras 3, scikit-learn, pandas,
  NumPy, Pillow, imagehash, Google Colab.
- **Original notebook:** TensorFlow / Keras, OpenCV, Matplotlib, NumPy, Google
  Colab.
- **Data download:** Kaggle CLI.

## 9. Timeline

- First commit: **2026-05-11**. The first README calls it a "Hackathon project,
  2025"; the hackathon date itself is NOT FOUND.
- Last commit: **2026-11-12**.
- Total commits: 6.

## 10. Visuals

- **The repo contains no image files.**
- The only two plots exist in old git history (outputs of the original
  notebook):
  - a single prediction on an uploaded image, labelled "TB Negative (0.29)",
    from the leaked-split model
  - a Grad-CAM heatmap that may come from an untrained model
- **Neither is recommended for posting.**
- The two best visuals for a LinkedIn post would have to be **created** from
  the results above:
  1. **Before / after bar chart:** the notebook's "95.69 % validation accuracy"
     (measured on training data) next to each model's real held-out test
     recall: 69.3 %, 84.2 %, 93.1 %, 96.0 %.
  2. **Per-source heatmap:** the "Correct predictions by source and class"
     table. It shows that every model gets RSNA Normals almost perfectly right,
     the source shortcut, while results on the NLM subset are lower.
