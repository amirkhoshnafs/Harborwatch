# scripts/sample_frames_v2.py

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class FrameRecord:
    source_name: str
    video_file_name: str
    video_relative_path: str
    video_stem: str
    frame_index: int
    timestamp_seconds: float
    source_fps: float
    sampling_fps: float
    frame_width: int
    frame_height: int
    image_file_name: str
    image_relative_path: str
    sampling_tier: str
    usable: str
    keep_for_mvp: str
    scene_type: str
    lighting: str
    weather: str
    camera_motion: str


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiered HarborWatch v2 frame sampler."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v2_sampling.yaml",
        help="Path to v2 sampling config.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def validate_registry_columns(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Registry CSV is empty.")

    available = set(rows[0].keys())
    missing = sorted(REQUIRED_REGISTRY_COLUMNS - available)
    if missing:
        raise ValueError(
            "Registry CSV is missing required columns: " + ", ".join(missing)
        )


def load_priority_videos(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Priority video list not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def normalize_flag(value: str) -> str:
    return (value or "").strip().lower()


def filter_rows(
    rows: list[dict],
    only_usable_yes: bool,
    only_keep_for_mvp_yes: bool,
) -> list[dict]:
    filtered = []

    for row in rows:
        usable = normalize_flag(row.get("usable", ""))
        keep_for_mvp = normalize_flag(row.get("keep_for_mvp", ""))

        if only_usable_yes and usable != "yes":
            continue
        if only_keep_for_mvp_yes and keep_for_mvp != "yes":
            continue

        filtered.append(row)

    return filtered


def get_frame_indices(frame_count: int, source_fps: float, sampling_fps: float) -> list[int]:
    if frame_count <= 0 or source_fps <= 0 or sampling_fps <= 0:
        return []

    duration_seconds = frame_count / source_fps
    sample_times = np.arange(0.0, duration_seconds, 1.0 / sampling_fps, dtype=float)

    indices = sorted(
        {
            min(frame_count - 1, int(round(t * source_fps)))
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


def read_frame(cap: cv2.VideoCapture, frame_index: int):
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


def build_report(
    rows_processed: list[dict],
    records: list[FrameRecord],
    priority_videos: set[str],
    output_report_md: Path,
    default_fps: float,
    priority_fps: float,
) -> None:
    per_video_counts: dict[str, int] = {}
    for record in records:
        per_video_counts[record.video_file_name] = per_video_counts.get(record.video_file_name, 0) + 1

    total_videos = len(rows_processed)
    total_frames = len(records)
    priority_video_counts = {
        video: per_video_counts.get(video, 0)
        for video in sorted(priority_videos)
    }

    top_lines = []
    for video_name, count in sorted(per_video_counts.items(), key=lambda x: x[1], reverse=True):
        tag = " [PRIORITY]" if video_name in priority_videos else ""
        top_lines.append(f"- `{video_name}`: **{count}** frames{tag}")

    priority_lines = []
    for video_name, count in priority_video_counts.items():
        priority_lines.append(f"- `{video_name}`: **{count}** frames")

    report = f"""# HarborWatch V2 Sampling Report

## Strategy
- Default sampling rate: **{default_fps:.1f} fps**
- Priority sampling rate: **{priority_fps:.1f} fps**
- Policy: all usable MVP videos are sampled, while buoy-priority videos are sampled more densely

## Totals
- Total videos sampled: **{total_videos}**
- Total extracted frames: **{total_frames}**

## Priority videos
{chr(10).join(priority_lines) if priority_lines else "- none"}

## Per-video extracted frame counts
{chr(10).join(top_lines) if top_lines else "- none"}

## Notes
This is the input pool for HarborWatch v2 annotation expansion.
The next step is auto-annotation with the current best checkpoint, then manual review and correction.
"""
    output_report_md.parent.mkdir(parents=True, exist_ok=True)
    output_report_md.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_yaml(config_path)

    paths = cfg["paths"]
    sampling = cfg["sampling"]

    video_root = Path(paths["video_root"]).resolve()
    registry_csv = Path(paths["video_registry_csv"]).resolve()
    priority_videos_txt = Path(paths["priority_videos_txt"]).resolve()
    output_frames_dir = Path(paths["output_frames_dir"]).resolve()
    output_manifest_csv = Path(paths["output_manifest_csv"]).resolve()
    output_manifest_jsonl = Path(paths["output_manifest_jsonl"]).resolve()
    output_report_md = Path(paths["output_report_md"]).resolve()

    default_fps = float(sampling["default_fps"])
    priority_fps = float(sampling["priority_fps"])
    only_usable_yes = bool(sampling["only_usable_yes"])
    only_keep_for_mvp_yes = bool(sampling["only_keep_for_mvp_yes"])
    image_extension = str(sampling["image_extension"])
    jpeg_quality = int(sampling["jpeg_quality"])

    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    rows = load_registry(registry_csv)
    validate_registry_columns(rows)
    rows = filter_rows(
        rows=rows,
        only_usable_yes=only_usable_yes,
        only_keep_for_mvp_yes=only_keep_for_mvp_yes,
    )

    priority_videos = load_priority_videos(priority_videos_txt)

    output_frames_dir.mkdir(parents=True, exist_ok=True)

    records: list[FrameRecord] = []

    for row in rows:
        file_name = row["file_name"]
        rel_path = row["relative_path"]
        video_path = video_root / rel_path

        if not video_path.exists():
            print(f"[WARN] Missing video: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {video_path}")
            continue

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        is_priority = file_name in priority_videos
        sampling_fps = priority_fps if is_priority else default_fps
        sampling_tier = "priority" if is_priority else "default"

        frame_indices = get_frame_indices(
            frame_count=frame_count,
            source_fps=source_fps,
            sampling_fps=sampling_fps,
        )

        video_stem = Path(file_name).stem
        video_output_dir = output_frames_dir / video_stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0

        for frame_index in frame_indices:
            frame = read_frame(cap, frame_index)
            if frame is None:
                print(f"[WARN] Failed frame {frame_index} in {file_name}")
                continue

            timestamp_seconds = frame_index / source_fps if source_fps > 0 else 0.0
            timestamp_tag = format_timestamp_for_name(timestamp_seconds)

            image_file_name = (
                f"{video_stem}__f{frame_index:06d}__t{timestamp_tag}{image_extension}"
            )
            image_path = video_output_dir / image_file_name

            ok = cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not ok:
                print(f"[WARN] Failed saving {image_path}")
                continue

            record = FrameRecord(
                source_name=row["source_name"],
                video_file_name=file_name,
                video_relative_path=rel_path,
                video_stem=video_stem,
                frame_index=frame_index,
                timestamp_seconds=round(timestamp_seconds, 3),
                source_fps=round(source_fps, 3),
                sampling_fps=sampling_fps,
                frame_width=frame_width,
                frame_height=frame_height,
                image_file_name=image_file_name,
                image_relative_path=str(image_path.relative_to(output_frames_dir.parent.parent)),
                sampling_tier=sampling_tier,
                usable=row.get("usable", ""),
                keep_for_mvp=row.get("keep_for_mvp", ""),
                scene_type=row.get("scene_type", ""),
                lighting=row.get("lighting", ""),
                weather=row.get("weather", ""),
                camera_motion=row.get("camera_motion", ""),
            )
            records.append(record)
            saved_count += 1

        cap.release()
        priority_tag = " [PRIORITY]" if is_priority else ""
        print(f"[OK] {file_name}: saved {saved_count} frames{priority_tag}")

    if not records:
        raise RuntimeError("No frames were extracted for v2.")

    write_manifest_csv(records, output_manifest_csv)
    write_manifest_jsonl(records, output_manifest_jsonl)
    build_report(
        rows_processed=rows,
        records=records,
        priority_videos=priority_videos,
        output_report_md=output_report_md,
        default_fps=default_fps,
        priority_fps=priority_fps,
    )

    print(f"\n[INFO] Total extracted frames: {len(records)}")
    print(f"[INFO] Manifest CSV: {output_manifest_csv}")
    print(f"[INFO] Manifest JSONL: {output_manifest_jsonl}")
    print(f"[INFO] Report: {output_report_md}")


if __name__ == "__main__":
    main()