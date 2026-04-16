# scripts/infer_video.py

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import yaml
from rfdetr import RFDETRNano


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HarborWatch offline video inference."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/infer_video.yaml",
        help="Path to inference config YAML.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV uses BGR

def draw_box(
    image,
    box,
    color,
    label,
    thickness: int,
    font_scale: float,
):
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)

    text_y = max(0, y1 - th - 8)
    cv2.rectangle(
        image,
        (x1, text_y),
        (x1 + tw + 6, text_y + th + 6),
        color,
        thickness=-1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 3, text_y + th + 1),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

def load_class_names_from_coco(coco_json_path: Path) -> list[str]:
    with coco_json_path.open("r", encoding="utf-8-sig") as f:
        coco = json.load(f)

    categories = sorted(coco["categories"], key=lambda c: int(c["id"]))
    return [str(cat["name"]) for cat in categories]

def resolve_class_name(class_id: int, class_names: list[str]) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_yaml(config_path)

    run_name = cfg["run"]["name"]
    checkpoint_path = Path(cfg["model"]["checkpoint"]).resolve()
    categories_json = Path(cfg["model"]["categories_json"]).resolve()
    class_names = load_class_names_from_coco(categories_json)

    video_path = Path(cfg["input"]["video_path"]).resolve()
    output_root = Path(cfg["output"]["root_dir"]).resolve()

    threshold = float(cfg["inference"]["threshold"])
    frame_stride = int(cfg["inference"]["frame_stride"])
    max_frames = cfg["inference"].get("max_frames", None)

    snapshots_enabled = bool(cfg["snapshots"]["enabled"])
    min_seconds_between_snapshots = float(cfg["snapshots"]["min_seconds_between"])

    line_thickness = int(cfg["render"]["line_thickness"])
    font_scale = float(cfg["render"]["font_scale"])
    draw_confidence = bool(cfg["render"]["draw_confidence"])

    class_colors_hex = dict(cfg["render"].get("class_colors_hex", {}))
    class_color_map = {
        class_name: hex_to_bgr(hex_color)
        for class_name, hex_color in class_colors_hex.items()
    }

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    run_dir = output_root / run_name
    snapshots_dir = run_dir / "snapshots"
    ensure_dir(run_dir)
    ensure_dir(snapshots_dir)

    annotated_video_path = run_dir / "annotated_video.mp4"
    detections_csv_path = run_dir / "detections.csv"
    detections_jsonl_path = run_dir / "detections.jsonl"
    metadata_json_path = run_dir / "run_metadata.json"
    summary_md_path = run_dir / "summary.md"

    print("[INFO] HarborWatch offline inference")
    print(f"[INFO] Run name: {run_name}")
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Output dir: {run_dir}")

    model = RFDETRNano(pretrain_weights=str(checkpoint_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if source_fps <= 0:
        source_fps = 25.0

    duration_seconds = total_frames / source_fps if total_frames > 0 else 0.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(annotated_video_path),
        fourcc,
        source_fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {annotated_video_path}")

    csv_fieldnames = [
        "run_name",
        "video_file_name",
        "frame_index",
        "timestamp_seconds",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    detections_per_class = Counter()
    frames_with_detections = 0
    total_detections = 0
    snapshot_count = 0
    last_snapshot_time = -1e9

    processed_frames = 0
    frame_index = 0

    with detections_csv_path.open("w", newline="", encoding="utf-8") as csv_f, \
         detections_jsonl_path.open("w", encoding="utf-8") as jsonl_f:

        csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fieldnames)
        csv_writer.writeheader()

        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            if max_frames is not None and processed_frames >= int(max_frames):
                break

            timestamp_seconds = frame_index / source_fps

            run_inference = (frame_index % frame_stride == 0)

            detections_this_frame = []

            if run_inference:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detections = model.predict(frame_rgb, threshold=threshold)

                if len(detections) > 0:
                    for xyxy, conf, cls_id in zip(
                        detections.xyxy,
                        detections.confidence,
                        detections.class_id,
                    ):
                        cls_id = int(cls_id)
                        class_name = resolve_class_name(cls_id, class_names)

                        x1, y1, x2, y2 = [float(v) for v in xyxy]
                        confidence = float(conf)

                        row = {
                            "run_name": run_name,
                            "video_file_name": video_path.name,
                            "frame_index": frame_index,
                            "timestamp_seconds": round(timestamp_seconds, 3),
                            "class_id": cls_id,
                            "class_name": class_name,
                            "confidence": round(confidence, 4),
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        }

                        csv_writer.writerow(row)
                        jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")

                        detections_this_frame.append(row)
                        detections_per_class[class_name] += 1
                        total_detections += 1

            overlay = frame_bgr.copy()

            if detections_this_frame:
                frames_with_detections += 1

                for det in detections_this_frame:
                    label = det["class_name"]
                    if draw_confidence:
                        label = f"{label} {det['confidence']:.2f}"

                    box_color = class_color_map.get(det["class_name"], (0, 0, 220))

                    draw_box(
                        overlay,
                        [det["x1"], det["y1"], det["x2"], det["y2"]],
                        color=box_color,
                        label=label,
                        thickness=line_thickness,
                        font_scale=font_scale,
                    )

                if snapshots_enabled and (timestamp_seconds - last_snapshot_time >= min_seconds_between_snapshots):
                    snapshot_name = (
                        f"{video_path.stem}__f{frame_index:06d}__t{int(timestamp_seconds):06d}.jpg"
                    )
                    snapshot_path = snapshots_dir / snapshot_name
                    cv2.imwrite(str(snapshot_path), overlay)
                    snapshot_count += 1
                    last_snapshot_time = timestamp_seconds

            writer.write(overlay)

            processed_frames += 1
            frame_index += 1

            if processed_frames % 100 == 0:
                print(
                    f"[INFO] Processed {processed_frames} frames | "
                    f"detections so far: {total_detections}"
                )

    cap.release()
    writer.release()

    metadata = {
        "project_name": cfg["project"]["name"],
        "run_name": run_name,
        "video_path": str(video_path),
        "checkpoint_path": str(checkpoint_path),
        "class_names": class_names,
        "threshold": threshold,
        "frame_stride": frame_stride,
        "source_fps": source_fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "total_frames_in_source": total_frames,
        "processed_frames": processed_frames,
        "duration_seconds": round(duration_seconds, 3),
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "snapshots_saved": snapshot_count,
        "detections_per_class": dict(detections_per_class),
        "output_files": {
            "annotated_video": str(annotated_video_path),
            "detections_csv": str(detections_csv_path),
            "detections_jsonl": str(detections_jsonl_path),
            "run_metadata_json": str(metadata_json_path),
            "summary_md": str(summary_md_path),
            "snapshots_dir": str(snapshots_dir),
        },
    }

    with metadata_json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "# HarborWatch Inference Run Summary",
        "",
        f"- Run name: **{run_name}**",
        f"- Video: **{video_path.name}**",
        f"- Checkpoint: **{checkpoint_path.name}**",
        f"- Processed frames: **{processed_frames}**",
        f"- Source FPS: **{source_fps:.2f}**",
        f"- Duration (seconds): **{duration_seconds:.2f}**",
        f"- Total detections: **{total_detections}**",
        f"- Frames with detections: **{frames_with_detections}**",
        f"- Snapshots saved: **{snapshot_count}**",
        "",
        "## Detections by class",
    ]

    if detections_per_class:
        for class_name in class_names:
            summary_lines.append(
                f"- {class_name}: **{detections_per_class.get(class_name, 0)}**"
            )
    else:
        summary_lines.append("- none")

    summary_lines.extend(
        [
            "",
            "## Output files",
            f"- Annotated video: `{annotated_video_path}`",
            f"- Detections CSV: `{detections_csv_path}`",
            f"- Detections JSONL: `{detections_jsonl_path}`",
            f"- Metadata JSON: `{metadata_json_path}`",
            f"- Snapshots dir: `{snapshots_dir}`",
        ]
    )

    summary_md_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n[INFO] Inference complete")
    print(f"[INFO] Annotated video: {annotated_video_path}")
    print(f"[INFO] Detections CSV: {detections_csv_path}")
    print(f"[INFO] Detections JSONL: {detections_jsonl_path}")
    print(f"[INFO] Metadata JSON: {metadata_json_path}")
    print(f"[INFO] Summary MD: {summary_md_path}")
    print(f"[INFO] Snapshots saved: {snapshot_count}")


if __name__ == "__main__":
    main()