"""Step 2: leakage-safe train/val/test split.

The dataset ships no patient IDs, so a true patient-level split is impossible.
Best available proxy: every duplicate / likely-same-patient cluster from
hash_audit.py is a group, and groups never straddle splits. Stratified by
label x source so each split has the same class and source mix.

Output: reports/splits.csv (file, label, source, dup_group, split)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
SEED = 42


def main():
    idx = pd.read_csv(OUT / "image_index.csv")
    g = pd.read_csv(OUT / "dup_groups.csv")[["file", "dup_group"]]
    df = idx.merge(g, on="file")
    strata = df.label.astype(str) + "_" + df.source

    # 20 folds -> 3 test, 3 val, 14 train  (=15/15/70)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = np.empty(len(df), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(df, strata, df.dup_group)):
        fold[te] = k
    df["split"] = np.where(fold < 3, "test", np.where(fold < 6, "val", "train"))
    df.to_csv(OUT / "splits.csv", index=False)

    # --- verification -------------------------------------------------------
    leaks = df.groupby("dup_group").split.nunique()
    assert (leaks == 1).all(), "duplicate group spans splits"
    for col in ["md5_file", "md5_pixels"]:
        assert (df.groupby(col).split.nunique() == 1).all(), f"{col} spans splits"
    pairs = pd.read_csv(OUT / "duplicate_pairs.csv")
    s = dict(zip(df.file, df.split))
    cross = pairs[[s[a] != s[b] for a, b in zip(pairs.a, pairs.b)]]
    assert cross.empty
    print(pd.crosstab([df.split], [df.label, df.source], margins=True))
    print("dup pairs crossing splits:", len(cross))


if __name__ == "__main__":
    main()
