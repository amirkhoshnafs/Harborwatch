# HarborWatch Project Scope

## Project Name
HarborWatch

## Full Title
HarborWatch: Maritime Video Analytics from Raw Coastal Footage with RF-DETR

## One-Sentence Summary
HarborWatch is a production-style computer vision project that turns raw coastal maritime video into vessel and buoy detections, structured analytics outputs, annotated video, and deployment-ready inference artifacts.

## Primary Use Case
Fixed-camera coastal / harbor monitoring for maritime activity understanding.

## Problem Framing
In a real-world setting, operators may only provide raw video footage without a ready-made training dataset. HarborWatch is designed to simulate that scenario by starting from public maritime videos, building a custom detection dataset, training an RF-DETR detector, and deploying the result as a usable video analytics pipeline.

## Main Data Source
Singapore Maritime Dataset (visible on-shore videos)

## Secondary Data Source
SeaDronesSee (reserved for later cross-domain evaluation or V2 expansion)

## MVP Goal
Build an end-to-end system that:
1. Starts from raw maritime videos
2. Samples and curates frames
3. Uses a custom annotation policy
4. Trains RF-DETR on a self-created dataset
5. Runs inference on full videos
6. Produces annotated video and structured outputs

## MVP Includes
- raw video intake
- frame sampling
- custom annotation design
- dataset creation
- train / val / test split creation
- RF-DETR fine-tuning
- evaluation
- offline video inference
- annotated MP4 outputs
- CSV / JSON detection exports
- run summaries and visual outputs

## MVP Excludes
- live streaming
- dashboard UI
- AIS fusion
- multi-camera fusion
- geolocation
- advanced collision forecasting
- production serving infrastructure

## Core Classes
- buoy
- small_craft
- large_vessel

## Required Outputs Per Inference Run
- annotated video
- frame-level detections CSV
- frame-level detections JSONL
- run metadata JSON
- summary markdown report
- visual snapshots

## Engineering Style
- modular
- script-first
- reproducible
- config-driven where useful
- minimal practical OOP
- no notebook dependency
- strong saved outputs and reports

## Project Success Criteria
The project is successful if it demonstrates:
1. Strong data engineering from raw video to training-ready dataset
2. A credible RF-DETR maritime detection baseline
3. A clean offline video inference pipeline
4. Production-style outputs suitable for a portfolio repository

## Planned Future Extensions
- tracking
- restricted-zone logic
- buoy crossing events
- loitering detection
- traffic summaries
- RTSP support
- cross-domain testing with SeaDronesSee