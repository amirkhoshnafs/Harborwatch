# scripts/generate_contact_sheets.py

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HarborWatch contact sheets for manual video triage."
    )
    parser.add_argument(
        "--video-root",
        type=str,
        required=True,
        help="Root directory that matches relative_path entries in the registry.",
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
        help="Directory to save contact sheet images.",
    )
    parser.add_argument(
        "--samples-per-video",
        type=int,
        default=9,
        help="Number of frames to sample per video.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of columns in contact sheet.",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=420,
        help="Thumbnail width in pixels.",
    )
    parser.add_argument(
        "--thumb-height",
        type=int,
        default=236,
        help="Thumbnail height in pixels.",
    )
    parser.add_argument(
        "--save-individual-frames",
        action="store_true",
        help="Also save individual sampled frames.",
    )
    parser.add_argument(
        "--frames-output-dir",
        type=str,
        default="outputs/data_audit/phase1/frame_previews",
        help="Directory for optional individual frame outputs.",
    )
    return parser.parse_args()


def load_registry(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_sample_indices(frame_count: int, samples_per_video: int) -> List[int]:
    if frame_count <= 0:
        return []
    if samples_per_video <= 1:
        return [max(0, frame_count // 2)]

    indices = np.linspace(
        0,
        max(0, frame_count - 1),
        num=samples_per_video,
        dtype=int,
    )
    return indices.tolist()


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def draw_overlay(
    frame: np.ndarray,
    file_name: str,
    frame_index: int,
    timestamp_seconds: float,
) -> np.ndarray:
    canvas = frame.copy()
    text_1 = file_name
    text_2 = f"frame={frame_index} | t={timestamp_seconds:.2f}s"

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 54), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        text_1,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text_2,
        (10, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def make_contact_sheet(
    frames: list[np.ndarray],
    columns: int,
    thumb_width: int,
    thumb_height: int,
    header_text: str,
) -> np.ndarray:
    if not frames:
        raise ValueError("No frames provided for contact sheet.")

    rows = math.ceil(len(frames) / columns)
    margin = 12
    header_h = 54

    resized_frames = [
        cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]

    sheet_width = columns * thumb_width + (columns + 1) * margin
    sheet_height = rows * thumb_height + (rows + 1) * margin + header_h

    sheet = np.full((sheet_height, sheet_width, 3), 245, dtype=np.uint8)

    cv2.rectangle(sheet, (0, 0), (sheet_width, header_h), (20, 20, 20), thickness=-1)
    cv2.putText(
        sheet,
        header_text,
        (12, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for idx, frame in enumerate(resized_frames):
        row = idx // columns
        col = idx % columns
        x = margin + col * (thumb_width + margin)
        y = header_h + margin + row * (thumb_height + margin)
        sheet[y:y + thumb_height, x:x + thumb_width] = frame

    return sheet


def sanitize_stem(name: str) -> str:
    return Path(name).stem.replace(" ", "_")


def main() -> None:
    args = parse_args()

    video_root = Path(args.video_root).resolve()
    registry_csv = Path(args.registry_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    frames_output_dir = Path(args.frames_output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_individual_frames:
        frames_output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_registry(registry_csv)
    if not rows:
        raise RuntimeError("Registry is empty.")

    for row in rows:
        rel_path = row["relative_path"]
        file_name = row["file_name"]
        frame_count = int(float(row["frame_count"])) if row["frame_count"] else 0
        fps = float(row["fps"]) if row["fps"] else 0.0

        video_path = video_root / rel_path
        if not video_path.exists():
            print(f"[WARN] Missing video: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Could not open video: {video_path}")
            continue

        sample_indices = get_sample_indices(frame_count, args.samples_per_video)
        sampled_frames: list[np.ndarray] = []

        for i, frame_index in enumerate(sample_indices):
            frame = read_frame(cap, frame_index)
            if frame is None:
                print(f"[WARN] Failed frame {frame_index} for {file_name}")
                continue

            timestamp_seconds = (frame_index / fps) if fps > 0 else 0.0
            frame = draw_overlay(frame, file_name, frame_index, timestamp_seconds)
            sampled_frames.append(frame)

            if args.save_individual_frames:
                frame_file = frames_output_dir / f"{sanitize_stem(file_name)}__{i:02d}.jpg"
                cv2.imwrite(str(frame_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        cap.release()

        if not sampled_frames:
            print(f"[WARN] No sampled frames for {file_name}")
            continue

        header_text = f"{file_name} | samples={len(sampled_frames)}"
        sheet = make_contact_sheet(
            frames=sampled_frames,
            columns=args.columns,
            thumb_width=args.thumb_width,
            thumb_height=args.thumb_height,
            header_text=header_text,
        )

        output_path = output_dir / f"{sanitize_stem(file_name)}_contact_sheet.jpg"
        cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[OK] Wrote {output_path.name}")


if __name__ == "__main__":
    main()