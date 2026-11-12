"""Choose decision thresholds on VAL, then report TEST at those thresholds.

The test set is never used to pick a threshold. Two rules:
  * sens95 : highest threshold whose VAL recall (sensitivity) >= 0.95, a screening-style
             operating point
  * youden : threshold maximising VAL (sensitivity + specificity - 1)

  python src/threshold.py cnn resnet_frozen resnet_ft resnet_ft25
Writes reports/thresholds.json.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from train_eval import CKPT, CXR, DEV, OUT, CustomCNN, ResNetTB, metrics, predict

TARGET_RECALL = 0.95


def val_preds(name, df):
    model = (CustomCNN() if name == "cnn" else ResNetTB()).to(DEV)
    model.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location=DEV))
    ds = CXR(df[df.split == "val"], augment=False, imagenet_norm=name.startswith("resnet_ft"))
    return predict(model, DataLoader(ds, batch_size=32))


def pick(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0, 1)
    ok = tpr >= TARGET_RECALL
    sens95 = float(thr[ok].max())
    youden = float(thr[np.argmax(tpr - fpr)])
    return {"sens95": sens95, "youden": youden}


def main(names):
    df = pd.read_csv(OUT / "splits.csv")
    out = {}
    for name in names:
        pv, yv = val_preds(name, df)
        test = pd.read_csv(OUT / f"preds_{name}.csv")
        yt, pt = test.label.values, test.p_tb.values
        nlm = (test.source == "NLM").values
        res = {}
        for rule, t in {"default_0.5": 0.5, **pick(yv, pv)}.items():
            res[rule] = {"threshold": t, "val": metrics(yv, pv, t),
                         "test": metrics(yt, pt, t), "test_nlm_only": metrics(yt[nlm], pt[nlm], t)}
            m = res[rule]["test"]
            print(f"{name:14s} {rule:12s} t={t:.4f} test acc={m['accuracy']:.4f} "
                  f"rec={m['recall']:.4f} spec={m['specificity']:.4f} f1={m['f1']:.4f} "
                  f"cm={m['confusion_matrix']}  nlm_rec={res[rule]['test_nlm_only']['recall']:.4f}")
        out[name] = res
    json.dump(out, open(OUT / "thresholds.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main(sys.argv[1:])
