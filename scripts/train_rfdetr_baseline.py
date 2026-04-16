# scripts/train_rfdetr_baseline.py

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from rfdetr import RFDETRNano


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or resume HarborWatch RF-DETR baseline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_rfdetr.yaml",
        help="Path to training config YAML.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)

    dataset_dir = Path(cfg["dataset"]["dataset_dir"]).resolve()
    output_dir = Path(cfg["train"]["output_dir"]).resolve()
    resume_path = cfg["train"].get("resume", None)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    if resume_path is not None:
        resume_path = str(Path(resume_path).resolve())

    print("[INFO] Starting HarborWatch RF-DETR run")
    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Dataset dir: {dataset_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Epochs: {cfg['train']['epochs']}")
    print(f"[INFO] Batch size: {cfg['train']['batch_size']}")
    print(f"[INFO] Grad accum steps: {cfg['train']['grad_accum_steps']}")
    print(f"[INFO] LR: {cfg['train']['lr']}")
    print(f"[INFO] Resume: {resume_path}")

    model = RFDETRNano()

    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        grad_accum_steps=cfg["train"]["grad_accum_steps"],
        lr=cfg["train"]["lr"],
        output_dir=str(output_dir),
    )

    if resume_path:
        train_kwargs["resume"] = resume_path

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()