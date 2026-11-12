"""Steps 3-4: train on TRAIN, select checkpoint on VAL, report on held-out TEST.

PyTorch port of the two Keras models in BONE_TB.ipynb (same layers, same
augmentation, same optimiser/LR/epochs), plus a properly fine-tuned ResNet50.

  python src/train_eval.py cnn              # notebook custom CNN
  python src/train_eval.py resnet_frozen    # notebook ResNet50 (frozen, /255 input)
  python src/train_eval.py resnet_ft        # fixed ResNet50 (ImageNet norm, top blocks unfrozen)
  python src/train_eval.py resnet_ft25      # same, phase 2 up to 25 epochs

Writes reports/metrics_<model>.json and reports/preds_<model>.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
CKPT = ROOT / "checkpoints"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 42
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- data -----
class CXR(Dataset):
    """Images cached as 224x224 uint8 (Keras flow_from_directory also resizes first)."""

    cache: dict = {}

    def __init__(self, df, augment, imagenet_norm):
        self.df = df.reset_index(drop=True)
        aug = [T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
               T.RandomHorizontalFlip()] if augment else []   # mirrors notebook ImageDataGenerator
        norm = [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)] if imagenet_norm else []
        self.tf = T.Compose(aug + [T.ToDtype(torch.float32, scale=True)] + norm)

    def _load(self, rel):
        if rel not in CXR.cache:
            im = Image.open(ROOT / rel).convert("L").resize((224, 224), Image.BILINEAR)
            CXR.cache[rel] = torch.from_numpy(np.asarray(im).copy())
        return CXR.cache[rel]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        x = self._load(self.df.path[i]).unsqueeze(0).expand(3, -1, -1)
        return self.tf(x), torch.tensor(float(self.df.label[i]))


# -------------------------------------------------------------- models -----
class CustomCNN(nn.Module):
    """Exact layer-for-layer port of the notebook's Sequential model (11.17M params)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 26 * 26, 128), nn.ReLU(),
                                  nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class ResNetTB(nn.Module):
    """ResNet50 (ImageNet) + GAP -> Dense128 -> Dropout0.5 -> 1, as in the notebook."""

    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

    def set_trainable(self, blocks):
        """Freeze everything, then unfreeze named top blocks (e.g. ['layer3','layer4'])."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for b in blocks:
            for p in getattr(self.backbone, b).parameters():
                p.requires_grad = True
        self._trainable = set(blocks)

    def train(self, mode=True):
        super().train(mode)
        # keep BatchNorm running stats fixed in frozen parts (Keras does this for trainable=False)
        if mode:
            for name, m in self.backbone.named_children():
                if name not in getattr(self, "_trainable", set()):
                    m.eval()
        return self


# ------------------------------------------------------------ training -----
@torch.no_grad()
def predict(model, loader):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(DEV))).cpu())
        ys.append(y)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def run_phase(model, opt, tr, va, epochs, tag, best, patience=None):
    lossf = nn.BCEWithLogitsLoss()
    stale = 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        p, y = predict(model, va)
        vloss = float(nn.functional.binary_cross_entropy(torch.tensor(p).clamp(1e-7, 1 - 1e-7),
                                                         torch.tensor(y)))
        vauc = roc_auc_score(y, p)
        vacc = accuracy_score(y, p > 0.5)
        print(f"[{tag}] ep{ep:02d} train_loss={tot/n:.4f} val_loss={vloss:.4f} "
              f"val_acc={vacc:.4f} val_auc={vauc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vloss < best["vloss"]:
            best.update(vloss=vloss, epoch=f"{tag}:{ep}")
            torch.save(model.state_dict(), best["path"])
            stale = 0
        else:
            stale += 1
            if patience and stale >= patience:
                print(f"[{tag}] early stop", flush=True)
                break


def metrics(y, p, thr=0.5):
    yhat = (p > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "accuracy": accuracy_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "f1": f1_score(y, yhat, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if len(set(y)) == 2 else float("nan"),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def bootstrap_ci(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    acc = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        m = metrics(y[i], p[i])
        for k in keys:
            acc[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


def main(name):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    CKPT.mkdir(exist_ok=True)
    df = pd.read_csv(OUT / "splits.csv")
    imnet = name.startswith("resnet_ft")          # notebook fed /255 pixels to every model
    mk = lambda s, aug, sh: DataLoader(CXR(df[df.split == s], aug, imnet), batch_size=32,
                                       shuffle=sh, num_workers=0)
    tr, va, te = mk("train", True, True), mk("val", False, False), mk("test", False, False)
    best = {"vloss": float("inf"), "path": CKPT / f"{name}.pt", "epoch": None}

    if name == "cnn":
        model = CustomCNN().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        run_phase(model, opt, tr, va, 15, "cnn", best)
    elif name == "resnet_frozen":
        model = ResNetTB().to(DEV)
        model.set_trainable([])
        opt = torch.optim.Adam(model.head.parameters(), lr=1e-4)
        run_phase(model, opt, tr, va, 10, "frozen", best)
    elif name in ("resnet_ft", "resnet_ft25"):
        ft_epochs = 25 if name == "resnet_ft25" else 12
        model = ResNetTB().to(DEV)
        # phase 1: warm up the new head on a frozen backbone
        model.set_trainable([])
        opt = torch.optim.Adam(model.head.parameters(), lr=1e-3)
        run_phase(model, opt, tr, va, 3, "ft-head", best)
        # phase 2: unfreeze top two stages (conv4_x, conv5_x), much lower LR on the backbone
        model.load_state_dict(torch.load(best["path"]))
        model.set_trainable(["layer3", "layer4"])
        opt = torch.optim.AdamW([
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 1e-5},
            {"params": model.head.parameters(), "lr": 1e-4}], weight_decay=1e-4)
        run_phase(model, opt, tr, va, ft_epochs, "ft-unfrozen", best, patience=4)
    else:
        raise SystemExit(f"unknown model {name}")

    model.load_state_dict(torch.load(best["path"]))
    p, y = predict(model, te)
    test = df[df.split == "test"].reset_index(drop=True)
    test["p_tb"] = p
    test.to_csv(OUT / f"preds_{name}.csv", index=False)
    nlm = (test.source == "NLM").values
    res = {"model": name, "selected_checkpoint": best["epoch"], "val_loss": best["vloss"],
           "test": metrics(y, p), "test_ci95": bootstrap_ci(y, p),
           "test_nlm_only": metrics(y[nlm], p[nlm])}
    json.dump(res, open(OUT / f"metrics_{name}.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
