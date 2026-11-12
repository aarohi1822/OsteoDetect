"""Step 1: hash every image and find exact / near-duplicates.

Outputs (reports/):
  image_index.csv      one row per image: path, label, source, md5 hashes, phash, dhash
  duplicate_pairs.csv  every exact or near-duplicate pair with distances
  dup_groups.csv       connected components of duplicates (used as split groups)
"""
import hashlib
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/TB_Chest_Radiography_Database"
OUT = ROOT / "reports"

# Thresholds on Pearson correlation of 64x64 z-scored grayscale thumbnails.
# Chosen by visual inspection of pairs around each cut (see README, "Duplicate audit"):
#   >= 0.985 : same image re-encoded / re-cropped           -> "near_duplicate"
#   >= 0.955 : different exposure, visibly the same patient -> "likely_same_patient"
#   ~0.95    : mostly different patients with similar habitus
CORR_NEAR = 0.985
CORR_PATIENT = 0.955


def load_index() -> pd.DataFrame:
    rows = []
    for label in ["Normal", "Tuberculosis"]:
        meta = pd.read_excel(DATA / f"{label}.metadata.xlsx")
        src = dict(zip(meta["FILE NAME"].str.strip(), meta["URL"]))
        for p in sorted((DATA / label).glob("*.png")):
            url = src.get(p.stem, "")
            source = ("RSNA" if "rsna" in url else "NLM" if "PMC4256233" in url
                      else "Belarus" if "tuberculosis.by" in url else "unknown")
            rows.append({"path": str(p.relative_to(ROOT)), "file": p.name,
                         "label": int(label == "Tuberculosis"), "source": source})
    return pd.DataFrame(rows)


def main():
    df = load_index()
    md5_file, md5_px, ph, dh, thumbs = [], [], [], [], []
    for i, rel in enumerate(df["path"]):
        p = ROOT / rel
        md5_file.append(hashlib.md5(p.read_bytes()).hexdigest())
        im = Image.open(p).convert("L")
        md5_px.append(hashlib.md5(im.tobytes()).hexdigest())
        ph.append(imagehash.phash(im))
        dh.append(imagehash.dhash(im))
        t = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float32).ravel()
        thumbs.append((t - t.mean()) / (t.std() + 1e-6))
        if i % 500 == 0:
            print(f"hashed {i}/{len(df)}", file=sys.stderr)
    df["md5_file"], df["md5_pixels"] = md5_file, md5_px
    df["phash"], df["dhash"] = [str(h) for h in ph], [str(h) for h in dh]
    df.to_csv(OUT / "image_index.csv", index=False)

    # pairwise pHash Hamming distance, vectorised
    bits = np.array([h.hash.ravel() for h in ph], dtype=np.uint8)          # N x 64
    dist = (bits[:, None, :] != bits[None, :, :]).sum(-1)                  # N x N
    T = np.stack(thumbs)
    n = len(df)
    iu, ju = np.triu_indices(n, k=1)
    C = (T @ T.T) / T.shape[1]
    cand = C[iu, ju] >= CORR_PATIENT
    pairs = []
    for i, j in zip(iu[cand], ju[cand]):
        corr = float(C[i, j])
        kind = ("exact_file" if md5_file[i] == md5_file[j]
                else "exact_pixels" if md5_px[i] == md5_px[j]
                else "near_duplicate" if corr >= CORR_NEAR else "likely_same_patient")
        if kind:
            pairs.append({"a": df.file[i], "b": df.file[j], "kind": kind,
                          "phash_dist": int(dist[i, j]), "pixel_corr": round(corr, 4),
                          "label_a": df.label[i], "label_b": df.label[j],
                          "source_a": df.source[i], "source_b": df.source[j]})
    pairs = pd.DataFrame(pairs, columns=["a", "b", "kind", "phash_dist", "pixel_corr",
                                         "label_a", "label_b", "source_a", "source_b"])
    pairs.to_csv(OUT / "duplicate_pairs.csv", index=False)

    # union-find -> duplicate groups
    parent = list(range(n))
    idx = {f: k for k, f in enumerate(df.file)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(pairs.a, pairs.b):
        parent[find(idx[a])] = find(idx[b])
    df["dup_group"] = [find(k) for k in range(n)]
    df[["file", "label", "source", "dup_group"]].to_csv(OUT / "dup_groups.csv", index=False)

    # pHash distance distribution (for choosing / justifying the threshold)
    d = dist[iu, ju]
    print("pHash distance quantiles over all pairs:",
          {q: int(np.quantile(d, q)) for q in [0.0001, 0.001, 0.01, 0.5]})
    print("confirmed pairs (corr>=%.3f): %d" % (CORR_PATIENT, len(pairs)))
    print(pairs.kind.value_counts().to_string())
    print("cross-label dup pairs:", int((pairs.label_a != pairs.label_b).sum()))
    sizes = df.dup_group.value_counts()
    print("images in a dup group:", int(sizes[sizes > 1].sum()), "groups:", int((sizes > 1).sum()))


if __name__ == "__main__":
    main()
