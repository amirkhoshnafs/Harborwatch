# scripts/prepare_coco_splits.py

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare HarborWatch COCO train/val/test splits by video."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to raw COCO annotation JSON.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root directory containing sampled frame images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed HarborWatch COCO dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio by video.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio by video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="Whether to copy or symlink images into split folders.",
    )
    return parser.parse_args()


def load_coco(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    required = {"images", "annotations", "categories"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"COCO JSON missing keys: {sorted(missing)}")

    return payload


def build_basename_index(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = defaultdict(list)

    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in index:
            duplicates[path.name].append(str(path))
        else:
            index[path.name] = path

    if duplicates:
        preview = {k: v[:3] for k, v in list(duplicates.items())[:5]}
        raise RuntimeError(
            "Duplicate image basenames found under image_root. "
            f"Examples: {preview}"
        )

    return index


def infer_video_stem(file_name: str) -> str:
    marker = "__f"
    if marker not in file_name:
        raise ValueError(
            f"Could not infer video stem from file_name: {file_name}. "
            "Expected naming like VIDEO__f000123__t000005p00.jpg"
        )
    return file_name.split(marker)[0]


def split_video_stems(
    video_stems: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    stems = sorted(video_stems)
    rng = random.Random(seed)
    rng.shuffle(stems)

    n = len(stems)
    if n < 3:
        raise ValueError("Need at least 3 videos to create train/val/test splits.")

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_stems = set(stems[:n_train])
    val_stems = set(stems[n_train:n_train + n_val])
    test_stems = set(stems[n_train + n_val:])

    split_map: dict[str, str] = {}
    for stem in train_stems:
        split_map[stem] = "train"
    for stem in val_stems:
        split_map[stem] = "val"
    for stem in test_stems:
        split_map[stem] = "test"

    return split_map


def ensure_split_dirs(output_dir: Path) -> None:
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)


def place_image(src: Path, dst: Path, copy_mode: str) -> None:
    if dst.exists():
        return

    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported copy_mode: {copy_mode}")


def filter_coco_for_split(
    payload: dict,
    split_image_ids: set[int],
    file_name_overrides: dict[int, str],
) -> dict:
    images = []
    annotations = []

    for img in payload["images"]:
        if img["id"] in split_image_ids:
            img_copy = dict(img)
            img_copy["file_name"] = file_name_overrides[img["id"]]
            images.append(img_copy)

    for ann in payload["annotations"]:
        if ann["image_id"] in split_image_ids:
            annotations.append(dict(ann))

    return {
        "licenses": payload.get("licenses", []),
        "info": payload.get("info", {}),
        "categories": payload["categories"],
        "images": images,
        "annotations": annotations,
    }


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_summary_markdown(summary: dict) -> str:
    def lines(items: list[str]) -> str:
        if not items:
            return "- none"
        return "\n".join(f"- `{x}`" for x in items)

    md = f"""# HarborWatch COCO Split Summary

## Split policy
- Split type: **by video**
- Train ratio: **{summary['train_ratio']:.2f}**
- Val ratio: **{summary['val_ratio']:.2f}**
- Seed: **{summary['seed']}**
- Copy mode: **{summary['copy_mode']}**

## Overall counts
- Total videos: **{summary['total_videos']}**
- Total images: **{summary['total_images']}**
- Total annotations: **{summary['total_annotations']}**

## Train
- Videos: **{summary['splits']['train']['videos']}**
- Images: **{summary['splits']['train']['images']}**
- Annotations: **{summary['splits']['train']['annotations']}**

### Train videos
{lines(summary['splits']['train']['video_list'])}

## Val
- Videos: **{summary['splits']['val']['videos']}**
- Images: **{summary['splits']['val']['images']}**
- Annotations: **{summary['splits']['val']['annotations']}**

### Val videos
{lines(summary['splits']['val']['video_list'])}

## Test
- Videos: **{summary['splits']['test']['videos']}**
- Images: **{summary['splits']['test']['images']}**
- Annotations: **{summary['splits']['test']['annotations']}**

### Test videos
{lines(summary['splits']['test']['video_list'])}
"""
    return md


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    payload = load_coco(input_json)
    basename_index = build_basename_index(image_root)

    for img in payload["images"]:
        img["video_stem"] = infer_video_stem(img["file_name"])

    video_stems = sorted({img["video_stem"] for img in payload["images"]})
    split_map = split_video_stems(
        video_stems=video_stems,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    ensure_split_dirs(output_dir)

    annotations_by_image_id: dict[int, int] = defaultdict(int)
    for ann in payload["annotations"]:
        annotations_by_image_id[ann["image_id"]] += 1

    split_image_ids: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    file_name_overrides_by_split: dict[str, dict[int, str]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    for img in payload["images"]:
        image_id = img["id"]
        file_name = img["file_name"]
        video_stem = img["video_stem"]
        split = split_map[video_stem]

        src_path = basename_index.get(file_name)
        if src_path is None:
            raise FileNotFoundError(
                f"Could not locate image file referenced by COCO JSON: {file_name}"
            )

        dst_path = output_dir / "images" / split / file_name
        place_image(src_path, dst_path, args.copy_mode)

        split_image_ids[split].add(image_id)
        file_name_overrides_by_split[split][image_id] = file_name

    split_payloads = {}
    for split in ["train", "val", "test"]:
        split_payloads[split] = filter_coco_for_split(
            payload=payload,
            split_image_ids=split_image_ids[split],
            file_name_overrides=file_name_overrides_by_split[split],
        )
        write_json(
            split_payloads[split],
            output_dir / "annotations" / f"{split}.json",
        )

    summary = {
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "copy_mode": args.copy_mode,
        "total_videos": len(video_stems),
        "total_images": len(payload["images"]),
        "total_annotations": len(payload["annotations"]),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        split_videos = sorted([stem for stem, s in split_map.items() if s == split])
        split_images = split_payloads[split]["images"]
        split_annotations = split_payloads[split]["annotations"]

        summary["splits"][split] = {
            "videos": len(split_videos),
            "images": len(split_images),
            "annotations": len(split_annotations),
            "video_list": split_videos,
        }

    summary_json_path = output_dir / "split_summary.json"
    summary_md_path = output_dir / "split_summary.md"

    write_json(summary, summary_json_path)
    summary_md_path.write_text(build_summary_markdown(summary), encoding="utf-8")

    print(f"[INFO] Prepared HarborWatch COCO dataset at: {output_dir}")
    print(f"[INFO] Total videos: {summary['total_videos']}")
    print(f"[INFO] Total images: {summary['total_images']}")
    print(f"[INFO] Total annotations: {summary['total_annotations']}")

    for split in ["train", "val", "test"]:
        stats = summary["splits"][split]
        print(
            f"[INFO] {split}: "
            f"{stats['videos']} videos | "
            f"{stats['images']} images | "
            f"{stats['annotations']} annotations"
        )


if __name__ == "__main__":
    main()