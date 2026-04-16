# scripts/register_videos.py

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2


@dataclass
class VideoRecord:
    source_name: str
    file_name: str
    relative_path: str
    extension: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    duration_minutes: float
    file_size_bytes: int
    usable: str
    keep_for_mvp: str
    scene_type: str
    lighting: str
    weather: str
    camera_motion: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan raw video files and build a HarborWatch video registry."
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing raw videos.",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="smd_visible_onshore",
        help="Short source name stored in the registry.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to output CSV registry.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        required=True,
        help="Path to output JSON registry.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for videos.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp4", ".avi", ".mov", ".mkv"],
        help="Allowed video extensions.",
    )
    return parser.parse_args()


def find_video_files(
    video_dir: Path,
    extensions: list[str],
    recursive: bool,
) -> list[Path]:
    normalized_exts = {ext.lower() for ext in extensions}

    if recursive:
        candidates = [p for p in video_dir.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in video_dir.iterdir() if p.is_file()]

    video_files = [p for p in candidates if p.suffix.lower() in normalized_exts]
    return sorted(video_files)


def safe_round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def extract_video_metadata(video_path: Path, source_name: str, root_dir: Path) -> VideoRecord:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    cap.release()

    duration_seconds = (frame_count / fps) if fps > 0 else 0.0
    file_size_bytes = video_path.stat().st_size

    return VideoRecord(
        source_name=source_name,
        file_name=video_path.name,
        relative_path=str(video_path.relative_to(root_dir)),
        extension=video_path.suffix.lower(),
        width=width,
        height=height,
        fps=safe_round(fps, 3),
        frame_count=frame_count,
        duration_seconds=safe_round(duration_seconds, 3),
        duration_minutes=safe_round(duration_seconds / 60.0, 3),
        file_size_bytes=file_size_bytes,
        usable="unknown",
        keep_for_mvp="unknown",
        scene_type="",
        lighting="",
        weather="",
        camera_motion="",
        notes="",
    )


def records_to_csv(records: Iterable[VideoRecord], output_csv: Path) -> None:
    records = list(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        raise ValueError("No records to write.")

    fieldnames = list(asdict(records[0]).keys())

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def records_to_json(records: Iterable[VideoRecord], output_json: Path) -> None:
    payload = [asdict(record) for record in records]
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    video_dir = Path(args.video_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json = Path(args.output_json).resolve()

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    video_files = find_video_files(
        video_dir=video_dir,
        extensions=args.extensions,
        recursive=args.recursive,
    )

    if not video_files:
        raise RuntimeError(f"No video files found in: {video_dir}")

    records: list[VideoRecord] = []
    for video_path in video_files:
        try:
            record = extract_video_metadata(
                video_path=video_path,
                source_name=args.source_name,
                root_dir=video_dir,
            )
            records.append(record)
            print(
                f"[OK] {video_path.name} | "
                f"{record.width}x{record.height} | "
                f"{record.fps} FPS | "
                f"{record.frame_count} frames"
            )
        except Exception as exc:
            print(f"[WARN] Failed to process {video_path}: {exc}")

    if not records:
        raise RuntimeError("No valid videos were processed.")

    records_to_csv(records, output_csv)
    records_to_json(records, output_json)

    print(f"\n[INFO] Wrote CSV registry to: {output_csv}")
    print(f"[INFO] Wrote JSON registry to: {output_json}")
    print(f"[INFO] Total videos registered: {len(records)}")


if __name__ == "__main__":
    main()