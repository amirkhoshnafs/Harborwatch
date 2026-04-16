# scripts/sanitize_coco_boxes.py

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip COCO bounding boxes to image bounds and drop invalid boxes."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to input COCO JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to cleaned COCO JSON.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        required=True,
        help="Path to sanitation report JSON.",
    )
    parser.add_argument(
        "--min-width",
        type=float,
        default=1.0,
        help="Minimum width after clipping.",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=1.0,
        help="Minimum height after clipping.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def clip_box(x: float, y: float, w: float, h: float, img_w: float, img_h: float):
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    new_w = max(0.0, x2 - x1)
    new_h = max(0.0, y2 - y1)
    return x1, y1, new_w, new_h


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json).resolve()
    output_json = Path(args.output_json).resolve()
    report_json = Path(args.report_json).resolve()

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    coco = load_json(input_json)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    image_size_map = {
        int(img["id"]): (float(img["width"]), float(img["height"]))
        for img in images
    }

    cleaned_annotations = []
    report = {
        "input_annotations": len(annotations),
        "output_annotations": 0,
        "boxes_changed": 0,
        "boxes_dropped": 0,
        "dropped_examples": [],
    }

    for ann in annotations:
        ann_copy = dict(ann)
        image_id = int(ann_copy["image_id"])
        bbox = ann_copy.get("bbox", None)

        if image_id not in image_size_map or not isinstance(bbox, list) or len(bbox) != 4:
            report["boxes_dropped"] += 1
            if len(report["dropped_examples"]) < 10:
                report["dropped_examples"].append(
                    {"annotation_id": ann_copy.get("id"), "reason": "missing_image_or_invalid_bbox"}
                )
            continue

        img_w, img_h = image_size_map[image_id]
        x, y, w, h = [float(v) for v in bbox]

        new_x, new_y, new_w, new_h = clip_box(x, y, w, h, img_w, img_h)

        changed = (new_x != x) or (new_y != y) or (new_w != w) or (new_h != h)
        if changed:
            report["boxes_changed"] += 1

        if new_w < args.min_width or new_h < args.min_height:
            report["boxes_dropped"] += 1
            if len(report["dropped_examples"]) < 10:
                report["dropped_examples"].append(
                    {
                        "annotation_id": ann_copy.get("id"),
                        "reason": "too_small_after_clipping",
                        "old_bbox": bbox,
                        "new_bbox": [new_x, new_y, new_w, new_h],
                    }
                )
            continue

        ann_copy["bbox"] = [new_x, new_y, new_w, new_h]
        ann_copy["area"] = float(new_w * new_h)
        cleaned_annotations.append(ann_copy)

    cleaned_coco = dict(coco)
    cleaned_coco["annotations"] = cleaned_annotations

    report["output_annotations"] = len(cleaned_annotations)

    write_json(cleaned_coco, output_json)
    write_json(report, report_json)

    print(f"[INFO] Input annotations: {report['input_annotations']}")
    print(f"[INFO] Output annotations: {report['output_annotations']}")
    print(f"[INFO] Boxes changed: {report['boxes_changed']}")
    print(f"[INFO] Boxes dropped: {report['boxes_dropped']}")
    print(f"[INFO] Clean JSON: {output_json}")
    print(f"[INFO] Report JSON: {report_json}")


if __name__ == "__main__":
    main()