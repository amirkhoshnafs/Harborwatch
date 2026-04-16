# scripts/export_cvat_predictions.py

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from rfdetr import RFDETRNano


CLASS_NAMES = ["large_vessel", "buoy", "small_craft"]
CATEGORY_ID_MAP = {
    "large_vessel": 1,
    "buoy": 2,
    "small_craft": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RF-DETR predictions as CVAT-importable COCO JSON."
    )
    parser.add_argument(
        "--images-root",
        type=str,
        required=True,
        help="Root directory containing v2 sampled frames.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint_best_total.pth",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Output COCO JSON path for CVAT import.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        required=True,
        help="Output markdown summary path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Prediction confidence threshold.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="Allowed image extensions.",
    )
    return parser.parse_args()


def find_images(images_root: Path, extensions: list[str]) -> list[Path]:
    allowed = {ext.lower() for ext in extensions}
    images = [
        path for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed
    ]
    return sorted(images)


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [float(x1), float(y1), float(w), float(h)]


def main() -> None:
    args = parse_args()

    images_root = Path(args.images_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()

    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    image_paths = find_images(images_root, args.extensions)
    if not image_paths:
        raise RuntimeError(f"No images found under: {images_root}")

    model = RFDETRNano(pretrain_weights=str(checkpoint))

    coco = {
        "licenses": [],
        "info": {
            "description": "HarborWatch v2 auto-annotations exported from RF-DETR baseline",
            "version": "v2-auto-annotate-0.1",
        },
        "categories": [
            {"id": 1, "name": "large_vessel", "supercategory": ""},
            {"id": 2, "name": "buoy", "supercategory": ""},
            {"id": 3, "name": "small_craft", "supercategory": ""},
        ],
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    per_class_counter = Counter()

    for image_id, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        coco["images"].append(
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_path.name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        detections = model.predict(image_rgb, threshold=args.threshold)

        if len(detections) == 0:
            continue

        for xyxy, conf, cls_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.class_id,
        ):
            cls_id = int(cls_id)
            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                print(f"[WARN] Unexpected class id {cls_id} in {image_path.name}")
                continue

            class_name = CLASS_NAMES[cls_id]
            category_id = CATEGORY_ID_MAP[class_name]
            bbox = xyxy_to_xywh([float(v) for v in xyxy])
            area = bbox[2] * bbox[3]

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [],
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {
                        "score": round(float(conf), 4),
                        "auto_annotated": True,
                    },
                }
            )
            ann_id += 1
            per_class_counter[class_name] += 1

        if image_id % 50 == 0:
            print(f"[INFO] Processed {image_id}/{len(image_paths)} images")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    total_images = len(coco["images"])
    total_annotations = len(coco["annotations"])

    summary_lines = [
        "# HarborWatch V2 Auto-Annotation Summary",
        "",
        f"- Images scanned: **{total_images}**",
        f"- Predicted annotations exported: **{total_annotations}**",
        f"- Confidence threshold: **{args.threshold:.2f}**",
        "",
        "## Predicted boxes by class",
    ]

    if per_class_counter:
        for class_name in CLASS_NAMES:
            summary_lines.append(
                f"- {class_name}: **{per_class_counter.get(class_name, 0)}**"
            )
    else:
        summary_lines.append("- none")

    summary_lines.extend(
        [
            "",
            "## Notes",
            "- This file is intended for CVAT annotation upload and manual correction.",
            "- Treat buoy predictions as suggestions, not ground truth.",
            "- Review missed buoys aggressively during correction.",
        ]
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[INFO] Images scanned: {total_images}")
    print(f"[INFO] Exported annotations: {total_annotations}")
    print(f"[INFO] Output JSON: {output_json}")
    print(f"[INFO] Output summary: {output_md}")


if __name__ == "__main__":
    main()