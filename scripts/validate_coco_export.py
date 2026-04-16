# scripts/validate_coco_export.py

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate HarborWatch COCO annotation export and align it with sampled frames."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to COCO-style annotation JSON.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root directory containing sampled frame images.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to validation report JSON.",
    )
    return parser.parse_args()


def is_number(x) -> bool:
    return isinstance(x, (int, float))


def build_basename_index(image_root: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for path in image_root.rglob("*"):
        if path.is_file():
            index[path.name].append(str(path))
    return index


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json).resolve()
    image_root = Path(args.image_root).resolve()
    output_json = Path(args.output_json).resolve()

    if not input_json.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {input_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    with input_json.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    report = {
        "format_guess": "unknown",
        "is_valid_enough_for_next_step": False,
        "counts": {},
        "category_map": {},
        "errors": [],
        "warnings": [],
        "image_file_checks": {
            "found_exactly_once": 0,
            "missing": 0,
            "duplicate_basename_matches": 0,
            "missing_examples": [],
            "duplicate_examples": [],
        },
        "category_distribution": {},
        "per_image_annotation_count_summary": {},
    }

    if not isinstance(payload, dict):
        report["errors"].append("Top-level JSON is not an object.")
    else:
        has_images = "images" in payload
        has_annotations = "annotations" in payload
        has_categories = "categories" in payload
        if has_images and has_annotations and has_categories:
            report["format_guess"] = "coco_detection"
        else:
            report["errors"].append(
                "JSON is not standard COCO detection format: missing one or more of images, annotations, categories."
            )

    if report["errors"]:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Wrote validation report to: {output_json}")
        return

    images = payload["images"]
    annotations = payload["annotations"]
    categories = payload["categories"]

    report["counts"] = {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
    }

    category_ids = set()
    category_id_to_name = {}
    duplicate_category_ids = set()

    for cat in categories:
        cat_id = cat.get("id")
        cat_name = cat.get("name")

        if cat_id in category_ids:
            duplicate_category_ids.add(cat_id)
        category_ids.add(cat_id)
        category_id_to_name[cat_id] = cat_name

    report["category_map"] = {str(k): v for k, v in category_id_to_name.items()}

    if duplicate_category_ids:
        report["errors"].append(
            f"Duplicate category ids found: {sorted(list(duplicate_category_ids))[:10]}"
        )

    image_ids = set()
    duplicate_image_ids = set()
    image_id_to_size = {}
    image_id_to_name = {}

    for img in images:
        image_id = img.get("id")
        width = img.get("width")
        height = img.get("height")
        file_name = img.get("file_name")

        if image_id in image_ids:
            duplicate_image_ids.add(image_id)
        image_ids.add(image_id)

        if not isinstance(file_name, str) or not file_name.strip():
            report["errors"].append(f"Image id={image_id} has invalid file_name.")

        if not is_number(width) or not is_number(height):
            report["errors"].append(f"Image id={image_id} missing numeric width/height.")
        elif width <= 0 or height <= 0:
            report["errors"].append(f"Image id={image_id} has non-positive width/height.")

        image_id_to_size[image_id] = (float(width), float(height))
        image_id_to_name[image_id] = file_name

    if duplicate_image_ids:
        report["errors"].append(
            f"Duplicate image ids found: {sorted(list(duplicate_image_ids))[:10]}"
        )

    basename_index = build_basename_index(image_root)

    for img in images:
        file_name = img.get("file_name", "")
        matches = basename_index.get(file_name, [])

        if len(matches) == 1:
            report["image_file_checks"]["found_exactly_once"] += 1
        elif len(matches) == 0:
            report["image_file_checks"]["missing"] += 1
            if len(report["image_file_checks"]["missing_examples"]) < 10:
                report["image_file_checks"]["missing_examples"].append(file_name)
        else:
            report["image_file_checks"]["duplicate_basename_matches"] += 1
            if len(report["image_file_checks"]["duplicate_examples"]) < 10:
                report["image_file_checks"]["duplicate_examples"].append(
                    {"file_name": file_name, "matches": matches[:5]}
                )

    ann_ids = set()
    duplicate_ann_ids = set()
    category_counter = Counter()
    per_image_counter = defaultdict(int)

    for ann in annotations:
        ann_id = ann.get("id")
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")
        bbox = ann.get("bbox")

        if ann_id in ann_ids:
            duplicate_ann_ids.add(ann_id)
        ann_ids.add(ann_id)

        if image_id not in image_ids:
            report["errors"].append(
                f"Annotation id={ann_id} references missing image_id={image_id}."
            )

        if category_id not in category_ids:
            report["errors"].append(
                f"Annotation id={ann_id} references missing category_id={category_id}."
            )

        if not isinstance(bbox, list) or len(bbox) != 4:
            report["errors"].append(
                f"Annotation id={ann_id} has invalid bbox; expected list of length 4."
            )
            continue

        x, y, w, h = bbox
        if not all(is_number(v) for v in bbox):
            report["errors"].append(
                f"Annotation id={ann_id} bbox contains non-numeric values."
            )
            continue

        if w <= 0 or h <= 0:
            report["errors"].append(
                f"Annotation id={ann_id} has non-positive bbox size: {bbox}."
            )

        if image_id in image_id_to_size:
            img_w, img_h = image_id_to_size[image_id]
            if x < 0 or y < 0:
                report["warnings"].append(
                    f"Annotation id={ann_id} has negative bbox origin: {bbox}."
                )
            if x + w > img_w or y + h > img_h:
                report["warnings"].append(
                    f"Annotation id={ann_id} bbox exceeds image bounds: {bbox} for image_id={image_id}."
                )

        category_counter[category_id_to_name.get(category_id, str(category_id))] += 1
        per_image_counter[image_id] += 1

    if duplicate_ann_ids:
        report["errors"].append(
            f"Duplicate annotation ids found: {sorted(list(duplicate_ann_ids))[:10]}"
        )

    empty_images = len(images) - len(per_image_counter)
    report["counts"]["images_with_annotations"] = len(per_image_counter)
    report["counts"]["empty_images"] = empty_images
    report["category_distribution"] = dict(category_counter)

    if per_image_counter:
        values = list(per_image_counter.values())
        report["per_image_annotation_count_summary"] = {
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / len(values), 3),
        }

    if len(images) == 0:
        report["errors"].append("No images found.")
    if len(annotations) == 0:
        report["errors"].append("No annotations found.")
    if len(categories) == 0:
        report["errors"].append("No categories found.")

    if report["image_file_checks"]["missing"] > 0:
        report["errors"].append(
            f"{report['image_file_checks']['missing']} annotated image files were not found under image_root."
        )

    if report["image_file_checks"]["duplicate_basename_matches"] > 0:
        report["errors"].append(
            f"{report['image_file_checks']['duplicate_basename_matches']} annotated image file names matched multiple files on disk."
        )

    report["is_valid_enough_for_next_step"] = len(report["errors"]) == 0

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Wrote validation report to: {output_json}")
    print(f"[INFO] Format guess: {report['format_guess']}")
    print(f"[INFO] Images: {report['counts']['images']}")
    print(f"[INFO] Annotations: {report['counts']['annotations']}")
    print(f"[INFO] Categories: {report['counts']['categories']}")
    print(f"[INFO] Image files found exactly once: {report['image_file_checks']['found_exactly_once']}")
    print(f"[INFO] Missing image files: {report['image_file_checks']['missing']}")
    print(f"[INFO] Duplicate basename matches: {report['image_file_checks']['duplicate_basename_matches']}")
    print(f"[INFO] Valid enough for next step: {report['is_valid_enough_for_next_step']}")
    print(f"[INFO] Errors: {len(report['errors'])} | Warnings: {len(report['warnings'])}")


if __name__ == "__main__":
    main()