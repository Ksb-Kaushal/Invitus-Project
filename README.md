https://youtube.com/@invictussolution2122?si=I0fkOKKKpLzzrsPV

https://youtu.be/mI3tOjI4oZ8?si=rJZHFunskaS7nfY6

A “stabilizer bar” (a.k.a. sway/anti-roll bar) usually fails or is rejected because of: bent geometry (out-of-spec profile), weld/eyelet defects, surface defects (scratches, dents, cracks, rust/paint peel), or missing/damaged bushings/end-links. ([donsautorepairinc.com](https://donsautorepairinc.com/blog/5-signs-your-stabilizer-links-in-your-vehicle-need-attention/?utm_source=chatgpt.com), [Moog Parts](https://www.moogparts.com/parts-matter/Symptoms-of-Bad-Sway-Bar-Links.html?utm_source=chatgpt.com), [Delphiautoparts](https://www.delphiautoparts.com/resource-center/article/driving-a-car-with-worn-steering-and-suspension-parts?utm_source=chatgpt.com))

Below is a complete, production-style plan you can implement today. It combines:

- a **geometry check** (shape out-of-tolerance) using OpenCV,
- a **surface/weld defect detector** using state-of-the-art anomaly detection (PatchCore/DRAEM via Anomalib), and
- an optional **supervised detector** (YOLO) if you want labeled defect classes and pixel masks.

With good fixturing + lighting and enough good samples, this hybrid setup can realistically achieve ≥99% **recall** at high precision on a controlled line. PatchCore routinely reports ~99% AUROC on MVTec AD; many factories use it as a strong baseline. ([arXiv](https://arxiv.org/abs/2106.08265?utm_source=chatgpt.com), [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf?utm_source=chatgpt.com))

---

# 1) Line setup (critical for 99%)

1. **Fixturing**: place the bar on a jig so it always sits the same way.
2. **Cameras**:
    - **Top view** (global): 8–12 MP industrial camera, ~35–50 mm lens, to see the whole bar.
    - **Close-ups** (optional): 2 cameras aimed at end-eyes/welds.
3. **Lighting**:
    - **Backlight** (panel) under white acrylic for the geometry camera → crisp silhouette for contour matching.
    - **Diffuse dome** or ring light for close-ups → reveals scratches/porosity without harsh speculars.
4. **Background**: matte black or white only (opposite to the bar color).
5. **Trigger**: photoeye + hardware trigger for consistent exposure.

---

# 2) Data you need

- **Normal (good) images**: 500–2,000 (the more the better) for anomaly detection memory banks. PatchCore/DRAEM are trained mainly from *good* images. ([anomalib.readthedocs.io](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/patchcore.html?utm_source=chatgpt.com), [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf?utm_source=chatgpt.com))
- **Known defect images** (optional, for YOLO): aim for **50–200 per defect** (weld crack, deep scratch, paint peel, rust, missing bushing).
- Shoot under the *exact* lighting/angle you will use in production.

Labeling tools (if you do YOLO): Label Studio or Roboflow. YOLO supports detection and segmentation masks. ([Ultralytics Docs](https://docs.ultralytics.com/tasks/segment/?utm_source=chatgpt.com))

---

# 3) Folder structure

```
project/
 ├─ geometry/
 │   ├─ template/               # a few "golden" backlit images
 │   └─ samples/                # test images (backlit)
 ├─ anomaly/                    # for Anomalib (unsupervised AD)
 │   ├─ dataset/
 │   │   ├─ good/               # ONLY good images (train)
 │   │   └─ defect/             # hold-out validation/test (mix)
 └─ yolo/                       # optional supervised detector/segmenter
     ├─ images/train, val, test
     └─ labels/train, val, test   # YOLO txt (and masks if -seg)

```

---

# 4) Environment (Windows friendly)

```bash
# Python 3.10–3.11 recommended
python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip

# Core CV + inference
pip install opencv-python numpy scipy matplotlib

# Unsupervised anomaly detection (Anomalib)
pip install "anomalib[cpu]"         # or choose your GPU extra, e.g. anomalib[cu121]
# Quickstart & CLI/API are documented here:
# Anomalib in 15 Minutes  →  https://anomalib.readthedocs.io  (install/train/predict) :contentReference[oaicite:4]{index=4}

# Optional supervised model (YOLO)
pip install ultralytics
# Docs: training & segmentation usage :contentReference[oaicite:5]{index=5}

```

---

# 5) Geometry out-of-spec check (OpenCV)

**Idea:** build a binary silhouette “template” from golden parts, register each incoming part to the template (ECC alignment), then compute pixel-wise difference and a **shape deviation score**. This is robust, fast, and explainable.

### 5.1 Create the template (`geometry/make_template.py`)

```python
import cv2, glob, numpy as np, os
src_dir = "geometry/template"
imgs = []
for p in glob.glob(os.path.join(src_dir, "*.jpg")) + glob.glob(os.path.join(src_dir, "*.png")):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5,5), 0)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw = 255 - bw  # white bar on black
    imgs.append(bw.astype(np.float32)/255.0)

h, w = imgs[0].shape
acc = np.zeros((h,w), np.float32)
for im in imgs:
    acc += im
template = (acc / len(imgs))
# Keep pixels that are bar >80% of time → crisp silhouette
template = (template > 0.8).astype(np.uint8) * 255
cv2.imwrite("geometry/template_silhouette.png", template)
print("Saved template to geometry/template_silhouette.png")

```

### 5.2 Inspect a new part (`geometry/check_geometry.py`)

```python
import cv2, numpy as np

T = cv2.imread("geometry/template_silhouette.png", cv2.IMREAD_GRAYSCALE)
T = (T>0).astype(np.uint8)*255
Th, Tw = T.shape

def preprocess(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im = cv2.GaussianBlur(im,(5,5),0)
    _, bw = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw = 255 - bw                                # white bar, black bg
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, (5,5))
    bw = cv2.resize(bw, (Tw, Th), interpolation=cv2.INTER_AREA)
    return bw

def ecc_register(src, ref):
    # intensity-normalize for ECC
    src_f = src.astype(np.float32)/255.0
    ref_f = ref.astype(np.float32)/255.0
    warp = np.eye(2,3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    cc, warp = cv2.findTransformECC(ref_f, src_f, warp, cv2.MOTION_EUCLIDEAN, criteria)
    aligned = cv2.warpAffine(src, warp, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR)
    return aligned, warp

def shape_score(aligned, ref):
    # XOR difference ratio
    diff = cv2.bitwise_xor(aligned, ref)
    return np.sum(diff>0)/np.sum(ref>0), diff

if __name__ == "__main__":
    import sys
    path = sys.argv[1]  # e.g., geometry/samples/part001.jpg
    bw = preprocess(path)
    aligned, _ = ecc_register(bw, T)
    score, diff = shape_score(aligned, T)
    print(f"GEOMETRY_DEVIATION={score:.4f}")
    # Typical pass threshold ~0.02–0.05 (2–5% of silhouette area)
    PASS = score < 0.03
    print("PASS" if PASS else "FAIL")
    vis = cv2.cvtColor(T, cv2.COLOR_GRAY2BGR)
    vis[diff>0] = (0,0,255)  # red where deviates
    cv2.imwrite("geometry/last_check_overlay.png", vis)

```

**Run:**

```bash
python geometry\make_template.py
python geometry\check_geometry.py geometry\samples\part001.jpg
# See geometry/last_check_overlay.png for red deviation zones

```

---

# 6) Surface & weld anomaly detection (unsupervised)

**Why:** You rarely have every defect labeled. Modern **Anomalib** implementations (PatchCore, PaDiM, FastFlow, DRAEM, EfficientAD) learn “normal” appearance and flag anything off-distribution. Installation/training/inference are well documented with simple CLI & API. ([anomalib.readthedocs.io](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html), [ar5iv](https://ar5iv.labs.arxiv.org/html/2111.07677?utm_source=chatgpt.com))

### 6.1 Prepare data

- `anomaly/dataset/good/` — 1000+ images of acceptable bars (various batches, minor harmless variation).
- `anomaly/dataset/defect/` — a *small* set containing known defects for validation **only** (not necessary for training in one-class methods).

### 6.2 Quick training (CLI; EfficientAD or PatchCore)

```bash
# Example with EfficientAD (fast, strong baseline)
anomalib train --model efficient_ad \
               --data.datamodule.class_path anomalib.data.Folder \
               --data.datamodule.init_args.name stabbar \
               --data.datamodule.init_args.root anomaly/dataset \
               --data.datamodule.init_args.normal_dir good \
               --data.datamodule.init_args.abnormal_dir defect \
               --trainer.max_epochs 10

# Or PatchCore (higher recall; larger memory bank)
anomalib train --model patchcore \
               --data.datamodule.class_path anomalib.data.Folder \
               --data.datamodule.init_args.name stabbar \
               --data.datamodule.init_args.root anomaly/dataset \
               --data.datamodule.init_args.normal_dir good \
               --data.datamodule.init_args.abnormal_dir defect

```

(Anomalib install & CLI examples are in their “In 15 Minutes” guide.) ([anomalib.readthedocs.io](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html))

### 6.3 Inference (heatmaps + scores)

```bash
# Run on a single image or a directory
anomalib predict --ckpt_path results/stabbar/weights/model.ckpt \
                 --data_path anomaly/dataset/defect

```

You’ll get **pixel-level heatmaps** and an **image-level anomaly score**. PatchCore’s “total recall” behavior is ideal when you must not miss defects (tune threshold from validation ROC curve). ([docs.voxel51.com](https://docs.voxel51.com/tutorials/anomaly_detection.html?utm_source=chatgpt.com))

> Evidence that this approach can reach ~99% AUROC on standard benchmarks is in PatchCore’s paper; it’s a strong indicator that 99% recall is feasible with stable imaging and a good normal dataset. (arXiv, CVF Open Access)
> 

---

# 7) Optional: labeled defect classes with YOLO (dents, rust, missing bushing, etc.)

If you want **named defect types** and polygons, train a **YOLO segmentation** model (`-seg`). Ultralytics shows exactly how to train and export. ([Ultralytics Docs](https://docs.ultralytics.com/tasks/segment/?utm_source=chatgpt.com))

### 7.1 YAML (`yolo/stabbar.yaml`)

```yaml
path: yolo
train: images/train
val: images/val
names:
  0: weld_crack
  1: deep_scratch
  2: rust_peel
  3: missing_bushing

```

### 7.2 Train & infer

```python
# train_yolo.py
from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")  # start with a small segmenter
model.train(data="yolo/stabbar.yaml", epochs=80, imgsz=1280, batch=8, patience=20)
model.val()  # metrics
model.export(format="onnx")  # for deployment

```

**Run:** `python train_yolo.py`

YOLO docs for training/segmentation are here. ([Ultralytics Docs](https://docs.ultralytics.com/modes/train/?utm_source=chatgpt.com))

---

# 8) Single pass/fail pipeline (combining everything)

```python
# inspect_part.py
import cv2, numpy as np, json, os, torch
from ultralytics import YOLO

# ---- GEOMETRY (reuse from step 5) ----
from geometry.check_geometry import preprocess, ecc_register, shape_score
T = cv2.imread("geometry/template_silhouette.png", 0)

# ---- ANOMALY (Anomalib Lightning inferencer) ----
from anomalib.engine import Engine
from anomalib.data import PredictDataset
from anomalib.models import EfficientAd    # or PatchCore

engine = Engine()
ad_model = EfficientAd()
ckpt = "results/stabbar/weights/model.ckpt"  # your trained AD model

# ---- YOLO (optional)
yolo_model = YOLO("runs/segment/train/weights/best.pt")  # path to your best

def anomaly_score(img_path):
    ds = PredictDataset(path=img_path, image_size=(256,256))
    preds = engine.predict(model=ad_model, dataset=ds, ckpt_path=ckpt)
    p = next(iter(preds))
    return float(p.pred_score), p.anomaly_map

def yolo_defects(img_path, conf=0.35):
    res = yolo_model.predict(img_path, conf=conf, verbose=False)[0]
    dets = []
    for b in res.boxes:
        dets.append({"cls": int(b.cls), "name": res.names[int(b.cls)],
                     "conf": float(b.conf), "xyxy": b.xyxy[0].cpu().tolist()})
    return dets, res.plot()  # overlay image (numpy)

def run(img_path):
    # 1) geometry
    bw = preprocess(img_path); aligned, _ = ecc_register(bw, T)
    geo_score, geo_diff = shape_score(aligned, T)
    geo_ok = geo_score < 0.03

    # 2) anomaly
    ad_s, _ = anomaly_score(img_path)
    ad_ok = ad_s < 0.45   # tune on validation set

    # 3) YOLO (optional)
    yolo_dets, overlay = yolo_defects(img_path)
    yolo_ok = len(yolo_dets) == 0

    decision = geo_ok and ad_ok and yolo_ok
    out = {"geometry_deviation": geo_score, "anomaly_score": ad_s,
           "yolo_detections": yolo_dets, "PASS": bool(decision)}
    cv2.imwrite("last_overlay.jpg", overlay if overlay is not None else cv2.imread(img_path))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    import sys; run(sys.argv[1])

```

**Run:**

`python inspect_part.py path\to\new_part.jpg`

- **PASS** is true only if geometry is OK, anomaly score below threshold, and YOLO found no major defects.
- You’ll get `last_overlay.jpg` for visual evidence.

---

# 9) How to reach (and verify) ~99% accuracy

1. **Metrics**
    - Prefer **Recall at fixed low FPR** (e.g., FPR ≤ 0.5%) and **F1** per defect, not just accuracy.
    - Tune the **anomaly threshold** on a separate validation set using ROC/PR curves. PatchCore targets “total recall.” ([docs.voxel51.com](https://docs.voxel51.com/tutorials/anomaly_detection.html?utm_source=chatgpt.com))
2. **Data balance & hard negatives**
    - Add borderline “acceptable” variations (paint shade, oil film, minor scuffs) into **good** training so they don’t trigger false alarms.
    - Mine false positives from the line (active learning) every few days and add to **good** set.
3. **Imaging stability**
    - Fix exposure and white balance. Use shields to block ambient light.
4. **Multi-view if needed**
    - If the bar has concave regions (self-shadows), add a second diffuse light from the opposite side or a second camera.
5. **Per-zone models**
    - Train a small anomaly model per ROI (e.g., left eyelet, right eyelet, center bends). Memory banks become more compact, thresholds tighter.
6. **Weld/eyelet special care**
    - If weld porosity is critical, collect a small labeled set and add a **YOLO-seg** class for “weld_crack/porosity.”
7. **Benchmark and document**
    - Track AUROC, PRO (per-region overlap) and pixel-AP for localization; PatchCore, PaDiM, DRAEM and FastFlow are all supported in Anomalib, so you can A/B test and keep the best. ([anomalib.readthedocs.io](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/patchcore.html?utm_source=chatgpt.com), [arXiv](https://arxiv.org/abs/2108.07610?utm_source=chatgpt.com), [ar5iv](https://ar5iv.labs.arxiv.org/html/2111.07677?utm_source=chatgpt.com))

---

# 10) Deployment notes

- Export YOLO to ONNX/TensorRT; Anomalib models run with PyTorch or OpenVINO. ([Ultralytics Docs](https://docs.ultralytics.com/?utm_source=chatgpt.com))
- Package the pipeline as a FastAPI service with two endpoints: `/inspect` and `/health`.
- Save overlays + JSON for audit.
- Add a small **calibration UI** to nudge thresholds without re-training.

---

## References (what this plan is based on)

- **Anomalib** quick start, training & inference (multiple AD models, including PatchCore, DRAEM, FastFlow, PaDiM, EfficientAD). ([anomalib.readthedocs.io](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html))
- **PatchCore**: “Towards Total Recall in Industrial Anomaly Detection” (reported up to ~99.6% AUROC on MVTec AD). ([arXiv](https://arxiv.org/abs/2106.08265?utm_source=chatgpt.com), [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf?utm_source=chatgpt.com))
- **DRAEM** (surface anomaly detection) and **PaDiM** (patch distribution modeling). ([CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf?utm_source=chatgpt.com), [arXiv](https://arxiv.org/abs/2011.08785?utm_source=chatgpt.com), [ACM Digital Library](https://dl.acm.org/doi/10.1007/978-3-030-68799-1_35?utm_source=chatgpt.com))
- **FastFlow** (normalizing flows for fast AD). ([ar5iv](https://ar5iv.labs.arxiv.org/html/2111.07677?utm_source=chatgpt.com))
- **YOLO** training/segmentation docs by Ultralytics. ([Ultralytics Docs](https://docs.ultralytics.com/modes/train/?utm_source=chatgpt.com))
- Stabilizer link/common failures for context on likely defects (bushings, cracks, play). ([donsautorepairinc.com](https://donsautorepairinc.com/blog/5-signs-your-stabilizer-links-in-your-vehicle-need-attention/?utm_source=chatgpt.com), [Moog Parts](https://www.moogparts.com/parts-matter/Symptoms-of-Bad-Sway-Bar-Links.html?utm_source=chatgpt.com), [Delphiautoparts](https://www.delphiautoparts.com/resource-center/article/driving-a-car-with-worn-steering-and-suspension-parts?utm_source=chatgpt.com))

---

### What you’ll deliver in your demo

1. **Live camera feed** → geometry overlay (red diff), anomaly heatmap, YOLO masks (if any).
2. **Numbers**: geometry deviation %, anomaly score, defect list, final PASS/FAIL.
3. **Validation report**: ROC curves for anomaly model, confusion matrix for YOLO, and a short table proving ≥99% *recall* at your chosen FPR.

If you want, share 20–30 sample “good” and “defect” photos from your line (backlit + diffuse) and I’ll tailor the configs/thresholds to your exact parts.
