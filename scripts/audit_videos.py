# scripts/audit_videos.py

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a simple HarborWatch Phase 1 video audit report."
    )
    parser.add_argument(
        "--registry-csv",
        type=str,
        required=True,
        help="Path to video registry CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        required=True,
        help="Path to markdown report.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def to_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def build_report(rows: list[dict]) -> str:
    total_videos = len(rows)
    total_frames = sum(to_int(row["frame_count"]) for row in rows)
    total_duration_seconds = sum(to_float(row["duration_seconds"]) for row in rows)
    total_duration_minutes = total_duration_seconds / 60.0
    total_duration_hours = total_duration_minutes / 60.0

    unique_resolutions = sorted(
        {
            f'{to_int(row["width"])}x{to_int(row["height"])}'
            for row in rows
            if row["width"] and row["height"]
        }
    )

    fps_values = [to_float(row["fps"]) for row in rows if row["fps"]]
    min_fps = min(fps_values) if fps_values else 0.0
    max_fps = max(fps_values) if fps_values else 0.0

    rows_sorted_by_duration = sorted(
        rows, key=lambda row: to_float(row["duration_seconds"]), reverse=True
    )

    longest_section_lines = []
    for row in rows_sorted_by_duration[:10]:
        longest_section_lines.append(
            f"- `{row['file_name']}` — {to_float(row['duration_minutes']):.2f} min, "
            f"{to_int(row['frame_count'])} frames, "
            f"{to_int(row['width'])}x{to_int(row['height'])}, "
            f"{to_float(row['fps']):.2f} FPS"
        )

    report = f"""# HarborWatch Phase 1 Video Audit

## Summary
- Total videos: **{total_videos}**
- Total frames: **{total_frames:,}**
- Total duration: **{total_duration_minutes:.2f} minutes** ({total_duration_hours:.2f} hours)
- Unique resolutions: **{", ".join(unique_resolutions) if unique_resolutions else "N/A"}**
- FPS range: **{min_fps:.2f} to {max_fps:.2f}**

## Notes
This report is an initial metadata-only audit.
No annotation, class mapping, or frame sampling has been done yet.

## Longest videos
{chr(10).join(longest_section_lines) if longest_section_lines else "- None"}

## Next manual review fields to fill in
For each video in the registry CSV, manually review and fill:
- `usable`
- `keep_for_mvp`
- `scene_type`
- `lighting`
- `weather`
- `camera_motion`
- `notes`

## Exit criteria for Phase 1A
- All raw videos are registered
- Metadata has been extracted successfully
- A first-pass audit report exists
- The team can begin manual scene triage next
"""
    return report


def main() -> None:
    args = parse_args()

    registry_csv = Path(args.registry_csv).resolve()
    output_md = Path(args.output_md).resolve()

    if not registry_csv.exists():
        raise FileNotFoundError(f"Registry CSV not found: {registry_csv}")

    rows = load_rows(registry_csv)
    if not rows:
        raise RuntimeError("Registry CSV is empty.")

    report = build_report(rows)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")

    print(f"[INFO] Wrote audit report to: {output_md}")


if __name__ == "__main__":
    main()