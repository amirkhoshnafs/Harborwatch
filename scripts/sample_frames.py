# scripts/sample_frames.py

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

REQUIRED_REGISTRY_COLUMNS = {
    "source_name",
    "file_name",
    "relative_path",
    "fps",
    "frame_count",
    "usable",
    "keep_for_mvp",
    "scene_type",
    "lighting",
    "weather",
    "camera_motion",
}

def validate_registry_columns(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Registry CSV is empty.")

    available = set(rows[0].keys())
    missing = sorted(REQUIRED_REGISTRY_COLUMNS - available)

    if missing:
        raise ValueError(
            "Registry CSV is missing required columns: "
            + ", ".join(missing)
        )

@dataclass
class FrameRecord:
    source_name: str
    video_file_name: str
    video_relative_path: str
    video_stem: str
    frame_index: int
    timestamp_seconds: float
    fps: float
    frame_width: int
    frame_height: int
    image_file_name: str
    image_relative_path: str
    sampling_strategy: str
    usable: str
    keep_for_mvp: str
    scene_type: str
    lighting: str
    weather: str
    camera_motion: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample frames from HarborWatch raw videos."
    )
    parser.add_argument(
        "--video-root",
        type=str,
        required=True,
        help="Root directory matching the registry relative_path entries.",
    )
    parser.add_argument(
        "--registry-csv",
        type=str,
        required=True,
        help="Path to video registry CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save sampled frames.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=str,
        required=True,
        help="Path to output frame manifest CSV.",
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=str,
        required=True,
        help="Path to output frame manifest JSONL.",
    )
    parser.add_argument(
        "--sample-every-seconds",
        type=float,
        default=5.0,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved frames.",
    )
    parser.add_argument(
        "--image-extension",
        type=str,
        default=".jpg",
        help="Image extension for saved frames.",
    )
    parser.add_argument(
        "--only-usable-yes",
        action="store_true",
        help="Keep only rows where usable == yes.",
    )
    parser.add_argument(
        "--only-keep-for-mvp-yes",
        action="store_true",
        help="Keep only rows where keep_for_mvp == yes.",
    )
    return parser.parse_args()

def load_registry(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    normalized_rows = []
    for row in rows:
        normalized_row = {}
        for key, value in row.items():
            clean_key = key.strip() if key is not None else key
            normalized_row[clean_key] = value
        normalized_rows.append(normalized_row)

    return normalized_rows


def normalize_flag(value: str) -> str:
    return (value or "").strip().lower()


def filter_rows(
    rows: list[dict],
    only_usable_yes: bool,
    only_keep_for_mvp_yes: bool,
) -> list[dict]:
    filtered: list[dict] = []

    for row in rows:
        usable = normalize_flag(row.get("usable", ""))
        keep_for_mvp = normalize_flag(row.get("keep_for_mvp", ""))

        if only_usable_yes and usable != "yes":
            continue
        if only_keep_for_mvp_yes and keep_for_mvp != "yes":
            continue

        filtered.append(row)

    return filtered


def get_frame_indices(frame_count: int, fps: float, sample_every_seconds: float) -> list[int]:
    if frame_count <= 0 or fps <= 0 or sample_every_seconds <= 0:
        return []

    duration_seconds = frame_count / fps
    sample_times = np.arange(0.0, duration_seconds, sample_every_seconds, dtype=float)

    indices = sorted(
        {
            min(frame_count - 1, int(round(t * fps)))
            for t in sample_times
        }
    )

    if not indices:
        indices = [0]

    return indices


def format_timestamp_for_name(timestamp_seconds: float) -> str:
    whole = int(timestamp_seconds)
    frac = int(round((timestamp_seconds - whole) * 100))
    return f"{whole:06d}p{frac:02d}"


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def write_manifest_csv(records: list[FrameRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("No frame records to write.")

    fieldnames = list(asdict(records[0]).keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_manifest_jsonl(records: list[FrameRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    video_root = Path(args.video_root).resolve()
    registry_csv = Path(args.registry_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_csv = Path(args.manifest_csv).resolve()
    manifest_jsonl = Path(args.manifest_jsonl).resolve()

    rows = load_registry(registry_csv)
    rows = filter_rows(
        rows=rows,
        only_usable_yes=args.only_usable_yes,
        only_keep_for_mvp_yes=args.only_keep_for_mvp_yes,
    )

    if not rows:
        raise RuntimeError("No videos remained after filtering.")

    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[FrameRecord] = []
    total_saved = 0

    for row in rows:
        rel_path = row["relative_path"]
        file_name = row["file_name"]
        video_path = video_root / rel_path

        if not video_path.exists():
            print(f"[WARN] Missing video: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {video_path}")
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frame_indices = get_frame_indices(
            frame_count=frame_count,
            fps=fps,
            sample_every_seconds=args.sample_every_seconds,
        )

        video_stem = Path(file_name).stem
        video_output_dir = output_dir / video_stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        saved_for_video = 0

        for frame_index in frame_indices:
            frame = read_frame(cap, frame_index)
            if frame is None:
                print(f"[WARN] Failed frame {frame_index} in {file_name}")
                continue

            timestamp_seconds = frame_index / fps if fps > 0 else 0.0
            timestamp_tag = format_timestamp_for_name(timestamp_seconds)

            image_file_name = (
                f"{video_stem}__f{frame_index:06d}__t{timestamp_tag}{args.image_extension}"
            )
            image_path = video_output_dir / image_file_name

            ok = cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not ok:
                print(f"[WARN] Failed to save frame image: {image_path}")
                continue

            record = FrameRecord(
                source_name=row["source_name"],
                video_file_name=file_name,
                video_relative_path=rel_path,
                video_stem=video_stem,
                frame_index=frame_index,
                timestamp_seconds=round(timestamp_seconds, 3),
                fps=round(fps, 3),
                frame_width=frame_width,
                frame_height=frame_height,
                image_file_name=image_file_name,
                image_relative_path=str(image_path.relative_to(output_dir.parent.parent)),
                sampling_strategy="sparse_uniform_5s",
                usable=row.get("usable", ""),
                keep_for_mvp=row.get("keep_for_mvp", ""),
                scene_type=row.get("scene_type", ""),
                lighting=row.get("lighting", ""),
                weather=row.get("weather", ""),
                camera_motion=row.get("camera_motion", ""),
            )
            records.append(record)
            saved_for_video += 1
            total_saved += 1

        cap.release()
        print(f"[OK] {file_name}: saved {saved_for_video} frames")

    if not records:
        raise RuntimeError("No frames were sampled.")

    write_manifest_csv(records, manifest_csv)
    write_manifest_jsonl(records, manifest_jsonl)

    print(f"\n[INFO] Total sampled frames: {total_saved}")
    print(f"[INFO] Manifest CSV: {manifest_csv}")
    print(f"[INFO] Manifest JSONL: {manifest_jsonl}")


if __name__ == "__main__":
    main()