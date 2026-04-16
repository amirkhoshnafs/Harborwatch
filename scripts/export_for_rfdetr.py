# scripts/export_for_rfdetr.py

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HarborWatch COCO splits to RF-DETR COCO directory structure."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to data/processed/harborwatch_coco_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to RF-DETR-ready dataset output directory.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="Whether to copy or symlink images.",
    )
    return parser.parse_args()


def place_file(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported copy_mode: {copy_mode}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def export_split(
    split_name_in: str,
    split_name_out: str,
    input_dir: Path,
    output_dir: Path,
    copy_mode: str,
) -> tuple[int, int]:
    input_json = input_dir / "annotations" / f"{split_name_in}.json"
    input_images_dir = input_dir / "images" / split_name_in
    output_split_dir = output_dir / split_name_out
    output_json = output_split_dir / "_annotations.coco.json"

    payload = load_json(input_json)

    image_count = 0
    for image in payload["images"]:
        file_name = image["file_name"]
        src = input_images_dir / file_name
        dst = output_split_dir / file_name

        if not src.exists():
            raise FileNotFoundError(f"Missing source image for split {split_name_in}: {src}")

        place_file(src, dst, copy_mode)
        image_count += 1

    write_json(payload, output_json)
    ann_count = len(payload["annotations"])

    return image_count, ann_count


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    split_map = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }

    total_images = 0
    total_annotations = 0

    for split_in, split_out in split_map.items():
        image_count, ann_count = export_split(
            split_name_in=split_in,
            split_name_out=split_out,
            input_dir=input_dir,
            output_dir=output_dir,
            copy_mode=args.copy_mode,
        )
        total_images += image_count
        total_annotations += ann_count
        print(
            f"[INFO] {split_in} -> {split_out}: "
            f"{image_count} images | {ann_count} annotations"
        )

    print(f"[INFO] Exported RF-DETR-ready dataset to: {output_dir}")
    print(f"[INFO] Total images: {total_images}")
    print(f"[INFO] Total annotations: {total_annotations}")


if __name__ == "__main__":
    main()