# scripts/audit_sampled_frames.py

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a HarborWatch sampled-frame audit report."
    )
    parser.add_argument(
        "--manifest-csv",
        type=str,
        required=True,
        help="Path to frame manifest CSV.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        required=True,
        help="Path to markdown report.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict]:
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


def build_report(rows: list[dict]) -> str:
    total_frames = len(rows)
    unique_videos = sorted({row["video_file_name"] for row in rows})

    per_video_counts: dict[str, int] = defaultdict(int)
    lighting_counter: Counter[str] = Counter()
    weather_counter: Counter[str] = Counter()
    scene_counter: Counter[str] = Counter()

    for row in rows:
        per_video_counts[row["video_file_name"]] += 1
        lighting_counter[row.get("lighting", "unknown")] += 1
        weather_counter[row.get("weather", "unknown")] += 1
        scene_counter[row.get("scene_type", "unknown")] += 1

    top_videos = sorted(
        per_video_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    def format_counter(counter: Counter[str]) -> str:
        if not counter:
            return "- none"
        lines = []
        for key, value in counter.most_common():
            lines.append(f"- {key}: **{value}**")
        return "\n".join(lines)

    top_video_lines = []
    for video_name, count in top_videos[:20]:
        top_video_lines.append(f"- `{video_name}`: **{count}** sampled frames")

    report = f"""# HarborWatch Frame Sampling Report

## Summary
- Total sampled frames: **{total_frames}**
- Unique source videos: **{len(unique_videos)}**
- Average sampled frames per video: **{(total_frames / len(unique_videos)):.2f}**

## Sampled frames per video
{chr(10).join(top_video_lines) if top_video_lines else "- none"}

## Scene type distribution
{format_counter(scene_counter)}

## Lighting distribution
{format_counter(lighting_counter)}

## Weather distribution
{format_counter(weather_counter)}

## Notes
This report summarizes the sparse-uniform candidate frame pool.
These sampled frames are not annotations yet.
They are the input pool for the next dataset-creation steps.

The current pool is intentionally sparse. Dense sampling and hard-case mining
will be added later if needed.

## Exit criteria for Phase 1C
- Sampled frame images exist on disk
- Frame manifest CSV and JSONL exist
- Coverage across MVP videos looks reasonable
- The project is ready for annotation policy and annotation subset design
"""
    return report


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv).resolve()
    output_md = Path(args.output_md).resolve()

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    rows = load_rows(manifest_csv)
    if not rows:
        raise RuntimeError("Frame manifest is empty.")

    report = build_report(rows)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")

    print(f"[INFO] Wrote report to: {output_md}")


if __name__ == "__main__":
    main()