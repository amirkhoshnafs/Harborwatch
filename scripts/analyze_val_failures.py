# scripts/analyze_val_failures.py

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRNano


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HarborWatch failure analysis on a COCO split."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory of split images, e.g. data/processed/harborwatch_rfdetr_coco_v1/valid",
    )
    parser.add_argument(
        "--annotations-json",
        type=str,
        required=True,
        help="COCO annotations for the split, e.g. valid/_annotations.coco.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint_best_total.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for overlays and reports",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Prediction confidence threshold",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold for simple TP/FP/FN matching",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many worst cases to copy into summary folders",
    )
    return parser.parse_args()


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    text_thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, text_thickness)

    y_text_top = max(0, y1 - th - 8)
    cv2.rectangle(
        image,
        (x1, y_text_top),
        (x1 + tw + 6, y_text_top + th + 6),
        color,
        thickness=-1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 3, y_text_top + th + 1),
        font,
        scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )
    return image


def resolve_class_name(
    class_id: int,
    id_to_name: dict[int, str],
    zero_based_names: list[str],
) -> str:
    if class_id in id_to_name:
        return id_to_name[class_id]
    if 0 <= class_id < len(zero_based_names):
        return zero_based_names[class_id]
    if (class_id + 1) in id_to_name:
        return id_to_name[class_id + 1]
    return f"class_{class_id}"


def greedy_match(
    gt_boxes: list[np.ndarray],
    pred_boxes: list[np.ndarray],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    matches: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    candidates = []
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            iou = compute_iou(g, p)
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for _, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi))

    unmatched_gt = set(range(len(gt_boxes))) - used_gt
    unmatched_pred = set(range(len(pred_boxes))) - used_pred
    return matches, unmatched_gt, unmatched_pred


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir).resolve()
    annotations_json = Path(args.annotations_json).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()

    overlays_dir = output_dir / "all_overlays"
    worst_dir = output_dir / "worst_images"
    buoy_dir = output_dir / "buoy_failures"

    overlays_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)
    buoy_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(annotations_json)
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    id_to_name = {int(cat["id"]): str(cat["name"]) for cat in categories}
    zero_based_names = [id_to_name[k] for k in sorted(id_to_name.keys())]

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[int(ann["image_id"])].append(ann)

    model = RFDETRNano(pretrain_weights=str(checkpoint))

    rows: list[dict] = []

    for image_info in images:
        image_id = int(image_info["id"])
        file_name = str(image_info["file_name"])
        image_path = images_dir / file_name

        if not image_path.exists():
            print(f"[WARN] Missing image on disk: {image_path}")
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        detections = model.predict(image_rgb, threshold=args.threshold)

        pred_boxes = []
        pred_names = []
        pred_scores = []

        if len(detections) > 0:
            for xyxy, conf, cls_id in zip(
                detections.xyxy,
                detections.confidence,
                detections.class_id,
            ):
                pred_boxes.append(np.array(xyxy, dtype=np.float32))
                pred_scores.append(float(conf))
                pred_names.append(
                    resolve_class_name(int(cls_id), id_to_name=id_to_name, zero_based_names=zero_based_names)
                )

        gt_boxes = []
        gt_names = []

        for ann in anns_by_image.get(image_id, []):
            gt_boxes.append(xywh_to_xyxy(ann["bbox"]))
            gt_names.append(id_to_name[int(ann["category_id"])])

        overlay = image_bgr.copy()

        # Draw GT in green
        for box, name in zip(gt_boxes, gt_names):
            draw_box(overlay, box, (0, 180, 0), f"GT:{name}")

        # Draw predictions in red
        for box, name, score in zip(pred_boxes, pred_names, pred_scores):
            draw_box(overlay, box, (0, 0, 220), f"P:{name} {score:.2f}")

        # Simple class-aware greedy matching
        total_tp = 0
        total_fp = 0
        total_fn = 0
        buoy_fn = 0
        buoy_fp = 0

        class_names = sorted(set(gt_names) | set(pred_names))
        for class_name in class_names:
            gt_idx = [i for i, n in enumerate(gt_names) if n == class_name]
            pred_idx = [i for i, n in enumerate(pred_names) if n == class_name]

            class_gt_boxes = [gt_boxes[i] for i in gt_idx]
            class_pred_boxes = [pred_boxes[i] for i in pred_idx]

            matches, unmatched_gt, unmatched_pred = greedy_match(
                class_gt_boxes,
                class_pred_boxes,
                iou_threshold=args.iou_threshold,
            )

            total_tp += len(matches)
            total_fn += len(unmatched_gt)
            total_fp += len(unmatched_pred)

            if class_name == "buoy":
                buoy_fn += len(unmatched_gt)
                buoy_fp += len(unmatched_pred)

        error_score = total_fp + total_fn

        overlay_path = overlays_dir / file_name
        cv2.imwrite(str(overlay_path), overlay)

        rows.append(
            {
                "file_name": file_name,
                "image_id": image_id,
                "gt_count": len(gt_boxes),
                "pred_count": len(pred_boxes),
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "error_score": error_score,
                "buoy_fp": buoy_fp,
                "buoy_fn": buoy_fn,
                "overlay_path": str(overlay_path),
            }
        )

    if not rows:
        raise RuntimeError("No images were analyzed.")

    rows_sorted = sorted(rows, key=lambda r: (r["error_score"], r["buoy_fn"], r["fp"]), reverse=True)

    csv_path = output_dir / "per_image_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "image_id",
                "gt_count",
                "pred_count",
                "tp",
                "fp",
                "fn",
                "error_score",
                "buoy_fp",
                "buoy_fn",
                "overlay_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

    for row in rows_sorted[: args.top_k]:
        src = Path(row["overlay_path"])
        dst = worst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    buoy_sorted = sorted(rows, key=lambda r: (r["buoy_fn"], r["buoy_fp"], r["error_score"]), reverse=True)
    for row in buoy_sorted[: args.top_k]:
        if row["buoy_fn"] == 0 and row["buoy_fp"] == 0:
            continue
        src = Path(row["overlay_path"])
        dst = buoy_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    total_images = len(rows)
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)
    total_buoy_fp = sum(r["buoy_fp"] for r in rows)
    total_buoy_fn = sum(r["buoy_fn"] for r in rows)

    summary_md = output_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# HarborWatch Validation Failure Analysis",
                "",
                f"- Images analyzed: **{total_images}**",
                f"- Total TP: **{total_tp}**",
                f"- Total FP: **{total_fp}**",
                f"- Total FN: **{total_fn}**",
                f"- Total buoy FP: **{total_buoy_fp}**",
                f"- Total buoy FN: **{total_buoy_fn}**",
                "",
                "## Interpretation",
                "- `worst_images/` contains the highest-error validation examples.",
                "- `buoy_failures/` isolates images where buoy errors are concentrated.",
                "- Use these folders to identify whether the next improvement should be more buoy labels, denser sampling, or annotation refinement.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Analyzed images: {total_images}")
    print(f"[INFO] Total TP: {total_tp}")
    print(f"[INFO] Total FP: {total_fp}")
    print(f"[INFO] Total FN: {total_fn}")
    print(f"[INFO] Total buoy FP: {total_buoy_fp}")
    print(f"[INFO] Total buoy FN: {total_buoy_fn}")
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Summary: {summary_md}")
    print(f"[INFO] Worst overlays: {worst_dir}")
    print(f"[INFO] Buoy failure overlays: {buoy_dir}")


if __name__ == "__main__":
    main()