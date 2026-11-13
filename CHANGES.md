# CHANGES (vs PROJECT_FACTS.md, commit 7f05d6b → HEAD 5e4910f)

**1. Requested fixes**
- Hash images, report duplicates: **done**. 40 pairs; 0 cross splits (`reports/duplicate_pairs.csv`, `reports/splits.csv`).
- Patient-level split: **partly done**. Patient IDs NOT FOUND; duplicate groups used as a proxy (`src/make_split.py`).
- Evaluate CNN + ResNet50 on held-out test: **done** for the PyTorch ports (`reports/metrics_*.json`); the original Keras `.h5` models were not evaluated.
- Retrain ResNet50 with top blocks unfrozen and lower LR: **done** (`src/train_eval.py:209-213`, `reports/metrics_resnet_ft*.json`).
- README results table + Limitations: **done** (`README.md`).
- Remove Kaggle key: **partly done**. Outputs cleared at HEAD (`398b052`), but the key is still in `521f48c`; revocation NOT FOUND.
- Val-chosen thresholds; 25-epoch run; corrected Colab notebook: **done / done / partly done** (notebook only smoke-tested; full-run results NOT FOUND).
- Commit and push to GitHub: **done** (PR #1 merged as `50c9465`).
- README cleanup, pinned `requirements.txt`, check demo link: **done**. A hosted demo is NOT FOUND; the Colab badge target returns HTTP 200.
- Per-source table, PROJECT_FACTS.md, PROJECT_BRIEF.md, 2 figures embedded in README: **done**.

**2. Changed values in PROJECT_FACTS.md**
- No metric changed: `reports/` and the pipeline scripts are identical to `7f05d6b` (`git diff 7f05d6b HEAD`).
- Commit count 6 → 8; last commit `abbaf59` 2026-11-12 18:15:29 → `5e4910f` 2026-11-13 11:46:18 +0530 (`git log`).
- Image files in tree 0 → 2 (`assets/`). README citations shifted: smoke-test claim `README.md:153` → `:161`; disclaimer `:220` → `:229`.

**3. Identity**
- Project name in README: "OsteoDetect: TB Chest X-ray Classifier" (`README.md:1`). Repo: https://github.com/aarohi1822/OsteoDetect, not renamed (GitHub API returns this name with no redirect).
- Demo URL: https://colab.research.google.com/github/aarohi1822/OsteoDetect/blob/main/notebooks/BONE_TB_fixed.ipynb (`README.md:3`). Hosted app: NOT FOUND.

- **4. New images:** `assets/before_after.png`, `assets/source_heatmap.png`. Both exist locally and on GitHub (HTTP 200, `image/png`).
- **5. Tests:** 0 test files (NOT FOUND). The 3 leakage `assert`s in `src/make_split.py:37-43` (4 conditions) all pass against `reports/splits.csv`.
