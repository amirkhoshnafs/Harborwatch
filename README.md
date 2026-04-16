# HarborWatch: Maritime Vessel & Buoy Detection from Coastal Video

HarborWatch is an end-to-end computer vision project for **maritime object detection** from **coastal video**. The project starts from raw public videos, builds a custom detection dataset, uses **model-assisted annotation** to scale labeling, fine-tunes **RF-DETR**, and runs offline video inference to generate **annotated video**, **structured detection logs**, and **demo-ready visual outputs**.

This repository is designed as a **production-style portfolio project**, not a notebook demo. The focus is on the full engineering pipeline:

* raw video intake
* frame extraction and dataset building
* annotation guidelines and CVAT workflow
* auto-labeling + manual correction
* RF-DETR training
* offline video inference
* saved outputs for inspection and deployment-style use

---

## Visual Results

### Demo Snapshots

<table>
  <tr>
    <td align="center">
      <img src="docs/images/MVI_1478_VIS__f000450__t000015.jpg" width="420"><br>
      <sub>MVI_1478_VIS — clear daytime vessel scene</sub>
    </td>
    <td align="center">
      <img src="docs/images/MVI_1483_VIS__f000150__t000005.jpg" width="420"><br>
      <sub>MVI_1483_VIS — multi-vessel coastal scene</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/images/MVI_1583_VIS__f000000__t000000.jpg" width="420"><br>
      <sub>MVI_1583_VIS — lower-light / harder visibility case</sub>
    </td>
    <td align="center">
      <img src="docs/images/MVI_1592_VIS__f000450__t000015.jpg" width="420"><br>
      <sub>MVI_1592_VIS — mixed-range vessel detection</sub>
    </td>
  </tr>
</table>

### Demo GIFs

<table>
  <tr>
    <td align="center">
      <img src="docs/images/demo_01.gif" width="420"><br>
      <sub>Demo 1</sub>
    </td>
    <td align="center">
      <img src="docs/images/demo_02.gif" width="420"><br>
      <sub>Demo 2</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/images/demo_03.gif" width="420"><br>
      <sub>Demo 3</sub>
    </td>
    <td align="center">
      <img src="docs/images/demo_04.gif" width="420"><br>
      <sub>Demo 4</sub>
    </td>
  </tr>
</table>

---

## Project Goal

The goal of HarborWatch is to build a realistic maritime monitoring pipeline that can:

* detect **large_vessel**, **small_craft**, and **buoy** objects from coastal video
* process raw videos into a trainable dataset
* support efficient data scaling through **auto-labeling + human correction**
* run offline inference on new videos and export useful artifacts

This project is intentionally framed as an **applied CV system** rather than a benchmark-only training repo.

---

## Final Detection Classes

HarborWatch uses a compact 3-class taxonomy for consistency and deployment relevance:

* `buoy`
* `large_vessel`
* `small_craft`

This smaller taxonomy keeps annotation cleaner, reduces class ambiguity, and makes the final detector easier to explain and use.

---

## End-to-End Pipeline

### 1. Raw Video Intake

Public maritime videos are collected from the coastal subset of the **Singapore Maritime Dataset** and organized into a registry with metadata and manual scene triage.

### 2. Frame Sampling

Frames are extracted from raw videos to create an annotation pool.

* initial sparse sampling for dataset bootstrapping
* denser sampling for priority videos during dataset expansion

### 3. Annotation Guidelines

A compact annotation policy is defined for:

* class taxonomy
* bounding box rules
* ambiguity handling
* truncation / occlusion policy
* ignore rules

### 4. Manual Annotation in CVAT

An initial wave of sampled frames is annotated manually in CVAT to build the first training set.

### 5. Model-Assisted Auto-Labeling

After the first detector is trained, it is used to pre-annotate a larger second wave of frames. These predictions are imported into CVAT and **manually reviewed / corrected**.

This is a key part of the HarborWatch data engine:

* first-wave manual labels bootstrap the detector
* the detector accelerates second-wave annotation
* a human reviewer corrects errors before retraining

### 6. RF-DETR Fine-Tuning

RF-DETR Nano is fine-tuned on the HarborWatch dataset using COCO-format exports and a script-first training workflow.

### 7. Offline Video Inference

A final inference script runs on raw videos and saves:

* annotated MP4 output
* detections CSV
* detections JSONL
* snapshots
* run metadata and summaries

---

## Data Creation Workflow

### Dataset v1

* sparse frame extraction from all selected on-shore videos
* manual annotation in CVAT
* COCO export and RF-DETR-ready packaging

### Dataset v2

* denser frame extraction from all videos
* even denser extraction for buoy-priority videos
* auto-annotation using the existing detector
* manual correction in CVAT
* cleaned COCO export
* final RF-DETR-ready dataset for continued fine-tuning

This project intentionally includes **model-assisted annotation** as part of the real engineering workflow.

---

## Repository Structure

```text
harborwatch/
├── configs/
│   ├── annotation.yaml
│   ├── infer_video.yaml
│   ├── project.yaml
│   ├── train_rfdetr*.yaml
│   └── v2_sampling.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── annotations/
│   ├── processed/
│   └── registry/
├── docs/
│   └── images/
├── outputs/
│   ├── analysis/
│   ├── auto_annotate/
│   ├── data_audit/
│   ├── runs/
│   └── train/
├── scripts/
│   ├── register_videos.py
│   ├── generate_contact_sheets.py
│   ├── sample_frames.py
│   ├── sample_frames_v2.py
│   ├── validate_coco_export.py
│   ├── prepare_coco_splits.py
│   ├── export_for_rfdetr.py
│   ├── train_rfdetr_baseline.py
│   └── infer_video.py
├── docs/
├── README.md
└── requirements.txt
```

---

## Key Features

* **Script-first** project design
* **Raw-video-to-model** workflow
* **Custom dataset creation** from public videos
* **CVAT annotation workflow**
* **Auto-labeling + correction loop**
* **COCO packaging and RF-DETR-ready export**
* **Offline video inference pipeline**
* **Saved run artifacts for inspection**
* **Demo snapshots and GIF outputs**

---

## Model

HarborWatch uses **RF-DETR Nano** as the final detector for the current version of the project.

The model choice is practical:

* good tradeoff between quality and compute
* straightforward fine-tuning workflow
* suitable for iterative dataset-building cycles
* strong enough for end-to-end portfolio use

The emphasis of this repository is the **system and data workflow**, not model-family comparison.

---

## Training Data Formats

Internally, the project uses:

* COCO exports from CVAT
* cleaned split datasets organized by video
* RF-DETR-ready split folders with `_annotations.coco.json`

This makes the training pipeline reproducible and tool-friendly.

---

## Inference Outputs

Each offline inference run saves a structured output folder containing:

* `annotated_video.mp4`
* `detections.csv`
* `detections.jsonl`
* `run_metadata.json`
* `summary.md`
* `snapshots/`

This makes HarborWatch feel like a usable CV product rather than just a training script.

---

## Example Usage

### 1. Register raw videos

```bash
python scripts/register_videos.py \
  --video-dir data/raw/smd_visible_onshore/videos \
  --source-name smd_visible_onshore \
  --output-csv data/registry/video_registry.csv \
  --output-json data/registry/video_registry.json \
  --recursive
```

### 2. Sample frames

```bash
python scripts/sample_frames.py \
  --video-root data/raw/smd_visible_onshore/videos \
  --registry-csv data/registry/video_registry.csv \
  --output-dir data/interim/sampled_frames \
  --manifest-csv data/registry/frame_manifest.csv \
  --manifest-jsonl data/registry/frame_manifest.jsonl \
  --sample-every-seconds 5.0 \
  --only-usable-yes \
  --only-keep-for-mvp-yes
```

### 3. Prepare COCO splits

```bash
python scripts/prepare_coco_splits.py \
  --input-json data/annotations/v2/annotations_clean.json \
  --image-root data/interim/v2_sampled_frames \
  --output-dir data/processed/harborwatch_coco_v2_clean \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --seed 42 \
  --copy-mode copy
```

### 4. Export RF-DETR-ready dataset

```bash
python scripts/export_for_rfdetr.py \
  --input-dir data/processed/harborwatch_coco_v2_clean \
  --output-dir data/processed/harborwatch_rfdetr_coco_v2_clean \
  --copy-mode copy
```

### 5. Run offline inference

```bash
python scripts/infer_video.py --config configs/infer_video.yaml
```

---

## Setup

Create and activate your environment, then install dependencies.

```bash
conda create -n harborwatch python=3.10 -y
conda activate harborwatch
pip install -r requirements.txt
```

If you use a different environment manager, keep Python 3.10+ and install the same dependencies.

---

## Requirements

Typical core dependencies include:

* Python 3.10+
* OpenCV
* NumPy
* PyYAML
* RF-DETR
* CVAT for annotation workflow

---

## Annotation Workflow

HarborWatch uses a two-stage annotation workflow:

### Stage 1 — Manual labels

A first wave of frames is labeled manually in CVAT to establish a high-quality initial dataset.

### Stage 2 — Model-assisted expansion

The initial detector is used to pre-annotate a larger frame pool. Those predictions are imported back into CVAT and **manually corrected**.

This is an intentional part of the repository design because it mirrors how real-world CV data pipelines often scale.

---

## What Makes This Repository Strong

This project is not just a trained detector.

It includes:

* raw-data intake
* dataset audits
* annotation policy
* manual labeling
* auto-labeling assistance
* data validation and sanitation
* RF-DETR training
* inference outputs for real videos
* visual demos for GitHub presentation

That is the main value of HarborWatch.

---

## Limitations

Current limitations include:

* no tracking layer yet
* no zone/event engine yet
* no live-stream pipeline yet
* buoy detection remains the hardest class
* current detector is optimized for offline video processing, not real-time deployment

---

## Future Work

Potential next steps:

* multi-object tracking
* zone-based maritime event logic
* traffic summaries and count analytics
* restricted-zone entry alerts
* stronger buoy-focused data expansion
* lightweight deployment packaging

---

## Acknowledgements

* Singapore Maritime Dataset for public coastal video data
* CVAT for annotation and correction workflow
* RF-DETR for the detector training framework

---

## License

MIT
