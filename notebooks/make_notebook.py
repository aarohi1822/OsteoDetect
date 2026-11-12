"""Generates BONE_TB_fixed.ipynb (Keras / Colab), the leakage-free version of BONE_TB.ipynb."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                               "outputs": [], "source": s.strip()})

md("""
# TB Chest X-ray Classifier (leakage-free)

Corrected version of `BONE_TB.ipynb`. Changes:
1. **Grouped train / val / test split.** The original used the same folder for train and val. Exact
   duplicates and likely same-patient images are grouped so a group never spans two splits.
2. **Test set is used once, at the end.** Checkpoints are chosen on val.
3. **ResNet50 uses `preprocess_input`**, not `/255`. This was why the original model predicted "Normal" for every image.
4. **ResNet50 is fine-tuned**: head warm-up, then conv4_x and conv5_x unfrozen at lr 1e-5.
5. **Full metrics**: precision, recall, specificity, F1, ROC-AUC, confusion matrix, plus a source-matched
   (NLM-only) subset.

⚠️ Never print `files.upload()` output. It echoes your `kaggle.json` (API key) into the notebook.
""")

code("""
# Versions this notebook was smoke-tested with; Colab's preinstalled pandas/numpy/pillow are left as-is.
!pip -q install tensorflow==2.21.0 keras==3.15.1 scikit-learn==1.9.1 imagehash==4.3.2 openpyxl==3.1.5
import os, hashlib, numpy as np, pandas as pd, tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
SEED = 42
tf.keras.utils.set_random_seed(SEED)
""")

code("""
# Kaggle credentials: assign the upload result so it is NOT echoed into the notebook output.
from google.colab import files
_uploaded = files.upload()
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json && rm kaggle.json
!kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset -p /content/data -q
!unzip -q -o /content/data/tuberculosis-tb-chest-xray-dataset.zip -d /content/data/tb
DATA = "/content/data/tb/TB_Chest_Radiography_Database"
""")

md("## 1. Index, hash and group duplicates")
code("""
def source_of(url):
    return ("RSNA" if "rsna" in url else "NLM" if "PMC4256233" in url
            else "Belarus" if "tuberculosis.by" in url else "unknown")

rows = []
for label in ["Normal", "Tuberculosis"]:
    meta = pd.read_excel(f"{DATA}/{label}.metadata.xlsx")
    src = dict(zip(meta["FILE NAME"].str.strip(), meta["URL"]))
    for f in sorted(os.listdir(f"{DATA}/{label}")):
        if f.endswith(".png"):
            rows.append({"path": f"{DATA}/{label}/{f}", "file": f,
                         "label": label, "y": int(label == "Tuberculosis"),
                         "source": source_of(src.get(f[:-4], ""))})
df = pd.DataFrame(rows)

thumbs, md5s = [], []
for p in df.path:
    im = Image.open(p).convert("L")
    md5s.append(hashlib.md5(im.tobytes()).hexdigest())
    t = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float32).ravel()
    thumbs.append((t - t.mean()) / (t.std() + 1e-6))
T = np.stack(thumbs); C = T @ T.T / T.shape[1]
df["md5"] = md5s

# corr >= 0.985: same image re-encoded or re-cropped; >= 0.955: visibly the same patient (checked by eye)
CORR_PATIENT = 0.955
iu, ju = np.triu_indices(len(df), 1)
hit = (C[iu, ju] >= CORR_PATIENT) | (df.md5.values[iu] == df.md5.values[ju])
parent = list(range(len(df)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i, j in zip(iu[hit], ju[hit]):
    parent[find(i)] = find(j)
df["group"] = [find(k) for k in range(len(df))]
pairs = pd.DataFrame({"a": df.file.values[iu[hit]], "b": df.file.values[ju[hit]], "corr": C[iu[hit], ju[hit]]})
print(f"duplicate / same-patient pairs: {len(pairs)}  "
      f"(exact md5: {(df.md5.values[iu[hit]] == df.md5.values[ju[hit]]).sum()}, "
      f"corr>=0.985: {(pairs['corr'] >= 0.985).sum()})")
print("cross-label pairs:", int((df.y.values[iu[hit]] != df.y.values[ju[hit]]).sum()))
""")

md("## 2. Grouped, stratified 70 / 15 / 15 split (no patient IDs exist; groups are the proxy)")
code("""
strata = df.y.astype(str) + "_" + df.source
fold = np.empty(len(df), int)
for k, (_, te) in enumerate(StratifiedGroupKFold(20, shuffle=True, random_state=SEED)
                            .split(df, strata, df.group)):
    fold[te] = k
df["split"] = np.where(fold < 3, "test", np.where(fold < 6, "val", "train"))
assert (df.groupby("group").split.nunique() == 1).all()
assert (df.groupby("md5").split.nunique() == 1).all()
print(pd.crosstab(df.split, [df.label, df.source], margins=True))
""")

code("""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre

AUG = dict(rotation_range=15, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
           shear_range=0.1, horizontal_flip=True, fill_mode="nearest")

def flows(resnet):
    pre = dict(preprocessing_function=resnet_pre) if resnet else dict(rescale=1/255.)
    kw = dict(x_col="path", y_col="label", target_size=(224, 224), batch_size=32, class_mode="binary",
              classes=["Normal", "Tuberculosis"])
    tr = ImageDataGenerator(**AUG, **pre).flow_from_dataframe(df[df.split == "train"], shuffle=True, seed=SEED, **kw)
    va = ImageDataGenerator(**pre).flow_from_dataframe(df[df.split == "val"], shuffle=False, **kw)
    te = ImageDataGenerator(**pre).flow_from_dataframe(df[df.split == "test"], shuffle=False, **kw)
    return tr, va, te

def ckpt(name):
    return [tf.keras.callbacks.ModelCheckpoint(f"{name}.keras", monitor="val_loss", save_best_only=True)]
""")

md("## 3a. Custom CNN (architecture unchanged)")
code("""
from tensorflow.keras import layers, models, optimizers
tr, va, te = flows(resnet=False)
cnn = models.Sequential([
    layers.Input((224, 224, 3)),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dense(128, activation="relu"), layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")])
cnn.compile(optimizers.Adam(1e-4), "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
cnn.fit(tr, validation_data=va, epochs=15, callbacks=ckpt("cnn"))
cnn = tf.keras.models.load_model("cnn.keras")
""")

md("## 3b / 4. ResNet50: `preprocess_input`, head warm-up, then fine-tune conv4_x + conv5_x")
code("""
from tensorflow.keras.applications import ResNet50
tr_r, va_r, te_r = flows(resnet=True)
base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
inp = layers.Input((224, 224, 3))
x = base(inp, training=False)            # keeps BatchNorm in inference mode even when unfrozen
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
resnet = models.Model(inp, layers.Dense(1, activation="sigmoid")(x))

# phase 1: head only
base.trainable = False
resnet.compile(optimizers.Adam(1e-3), "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
resnet.fit(tr_r, validation_data=va_r, epochs=3, callbacks=ckpt("resnet"))

# phase 2: unfreeze conv4_x and conv5_x (BatchNorm layers stay frozen), low LR
base.trainable = True
for l in base.layers:
    l.trainable = l.name.startswith(("conv4_", "conv5_")) and not isinstance(l, layers.BatchNormalization)
resnet.compile(optimizers.Adam(1e-5), "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
resnet.fit(tr_r, validation_data=va_r, epochs=25,
           callbacks=ckpt("resnet") + [tf.keras.callbacks.EarlyStopping("val_loss", patience=4)])
resnet = tf.keras.models.load_model("resnet.keras")
""")

md("## 5. Held-out TEST evaluation (run once)")
code("""
def report(model, gen, name, thr=0.5):
    p = model.predict(gen, verbose=0).ravel()
    y = np.asarray(gen.classes)
    sub = df[df.split == "test"].set_index("path").loc[gen.filepaths]
    out = {}
    for tag, m in {"test": np.ones(len(y), bool), "test_nlm_only": (sub.source == "NLM").values}.items():
        yh = (p[m] > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[m], yh, labels=[0, 1]).ravel()
        out[tag] = dict(n=int(m.sum()), accuracy=accuracy_score(y[m], yh), precision=precision_score(y[m], yh, zero_division=0),
                        recall=recall_score(y[m], yh), specificity=tn / (tn + fp), f1=f1_score(y[m], yh),
                        roc_auc=roc_auc_score(y[m], p[m]), tn=tn, fp=fp, fn=fn, tp=tp)
    return pd.DataFrame(out).T.assign(model=name)

results = pd.concat([report(cnn, te, "custom_cnn"), report(resnet, te_r, "resnet50_finetuned")])
results.round(4)
""")

md("""
## Limitations
* **Source confounding:** 88 % of Normal images come from RSNA, and every TB image comes from Belarus or NLM. Headline
  test metrics are optimistic. The `test_nlm_only` rows (both classes from one source) are the fairer check.
* **No patient IDs.** Groups come from image similarity, so a dissimilar repeat study of the same patient can still leak.
* **No external validation**, a single seed, and an uncalibrated 0.5 threshold.
* Research prototype, **not for clinical use.**
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
out = Path(__file__).with_name("BONE_TB_fixed.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
