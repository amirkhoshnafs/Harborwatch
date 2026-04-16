# HarborWatch

HarborWatch is a script-first computer vision pipeline for maritime object detection from coastal videos.

It covers the full workflow:
- Video registry creation
- Frame sampling
- CVAT annotation and validation
- COCO split preparation
- RF-DETR dataset export and training
- Offline inference with video and structured logs

## Detection Classes
- `large_vessel`
- `small_craft`
- `buoy`

## Project Structure
```text
harborwatch/
├── configs/                  # YAML configs for annotation, training, inference
├── data/                     # local datasets, registries, exports (ignored by git)
├── docs/                     # project docs and demo media
├── outputs/                  # training/inference/audit artifacts (ignored by git)
├── scripts/                  # end-to-end pipeline scripts
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Register raw videos
```bash
python scripts/register_videos.py \
  --video-dir data/raw/smd_visible_onshore/videos \
  --source-name smd_visible_onshore \
  --output-csv data/registry/video_registry.csv \
  --output-json data/registry/video_registry.json \
  --recursive
```

### 2. Sample frames for annotation
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

### 3. Validate CVAT COCO export
```bash
python scripts/validate_coco_export.py \
  --input-json data/annotations/raw/annotations.json \
  --image-root data/interim/sampled_frames \
  --output-json outputs/data_audit/phase3/annotation_validation_report.json
```

### 4. Build train/val/test COCO splits
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

### 5. Export RF-DETR dataset layout
```bash
python scripts/export_for_rfdetr.py \
  --input-dir data/processed/harborwatch_coco_v2_clean \
  --output-dir data/processed/harborwatch_rfdetr_coco_v2_clean \
  --copy-mode copy
```

### 6. Train RF-DETR
```bash
python scripts/train_rfdetr_baseline.py --config configs/train_rfdetr.yaml
```

### 7. Run offline inference
```bash
python scripts/infer_video.py --config configs/infer_video.yaml
```

Inference outputs are written under `outputs/runs/<run_name>/`:
- `annotated_video.mp4`
- `detections.csv`
- `detections.jsonl`
- `run_metadata.json`
- `summary.md`
- `snapshots/`

## Key Config Files
- `configs/project.yaml`: project metadata and path defaults
- `configs/annotation.yaml`: class taxonomy and annotation policy
- `configs/train_rfdetr*.yaml`: training runs and resume profiles
- `configs/infer_video.yaml`: inference model/video/output settings

## Annotation Workflow
1. Manually label initial frames in CVAT.
2. Train a baseline detector.
3. Auto-label more frames with `scripts/export_cvat_predictions.py`.
4. Correct labels in CVAT.
5. Retrain on cleaned exports.

## Demo Media

### Snapshots
<table>
  <tr>
    <td align="center">
      <img src="docs/images/snapshot_01.jpg" width="420"><br>
      <sub>Snapshot 1</sub>
    </td>
    <td align="center">
      <img src="docs/images/snapshot_02.jpg" width="420"><br>
      <sub>Snapshot 2</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/images/snapshot_03.jpg" width="420"><br>
      <sub>Snapshot 3</sub>
    </td>
    <td align="center">
      <img src="docs/images/snapshot_04.jpg" width="420"><br>
      <sub>Snapshot 4</sub>
    </td>
  </tr>
</table>

### GIFs
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

## Notes on Version Control
This repository is code/docs first. Large local artifacts should stay out of git:
- Raw/interim/processed data
- CVAT upload frame dumps
- Training checkpoints
- Inference outputs and audits

Use `.gitignore` to keep these untracked while retaining reproducible scripts/configs.

## License
MIT
