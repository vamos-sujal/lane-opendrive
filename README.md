# ADAS Lane2OpenDRIVE

This repository defines a fail-closed monocular lane perception pipeline for Colab/T4. It separates detector output, temporal tracking, geometry, topology, metric validation, and OpenDRIVE validation. It never invents camera calibration, lane width, vehicle speed, scale, or road length.

## Colab

Open `notebooks/ADAS_Lane2OpenDRIVE_Colab.ipynb` in Google Colab with a T4 runtime. The notebook mounts Drive, clones this repository, verifies CUDA, installs pinned dependencies, checks the detector contract, resolves one video from:

`/content/drive/MyDrive/lane_to_opendrive/input`

Use `input.mp4` or place exactly one `.mp4`/`.mov`/`.avi` file there. Results are written to:

`/content/drive/MyDrive/lane_to_opendrive/runs/<run_id>/`

## Local checks

```bash
python -m py_compile $(find src scripts -name '*.py')
PYTHONPATH=. python -m pytest -q
```

The full detector execution is intended for a CUDA Colab runtime, not this CPU development environment.

## Detector and weights

The Colab default uses the official Ultra-Fast-Lane-Detection CULane checkpoint because it accepts arbitrary raw frames and is practical on a T4. It produces real 2D lane pixels, overlays, counts, and temporal tracks. LATR remains available as a research 3D backend, but its official OpenLane runtime is not a raw-MP4 API.

Metric distance, calibrated visual-motion speed, top-down metric geometry, and OpenDRIVE remain disabled unless trusted camera intrinsics, camera height, downward pitch, and a measured lane width are supplied in `configs/camera.yaml`. Telemetry is not required for the visual-motion estimate, but monocular video alone cannot identify absolute scale. The system does not convert 2D pixels to invented meters.

Without a verified detector runtime and supplied metric calibration, the pipeline produces a `NON_METRIC` report and does not emit an OpenDRIVE file. This is intentional. Calibration values must come from a checkerboard/target calibration or manufacturer specification and must match the video camera and resolution.

## Metric configuration

Fill the null values in `configs/camera.yaml` with measured values. The camera model uses x=right, y=forward, z=up, and `pitch_deg` is positive downward. Do not set `calibration_available: true` until every value is known. The run reports `VISUAL_MOTION_CALIBRATED` only when tracked static lane evidence supports a speed estimate; this is a camera-motion estimate, not a claim of independently verified vehicle speed.

## Safety contract

OpenDRIVE generation requires `METRIC_VALID` geometry backed by supplied calibration or trusted telemetry. Missing calibration, invalid coordinates, XML errors, inconsistent lane structures, and uncertain scale are reported as failures rather than converted into plausible-looking meters.

See [docs/ARCHITECTURE_DECISION.md](docs/ARCHITECTURE_DECISION.md) and [configs/camera.yaml](configs/camera.yaml).

## Production model baseline

For a fresh multi-model T4 run, use [notebooks/Production_RoadMetrics_Colab.ipynb](notebooks/Production_RoadMetrics_Colab.ipynb). It preflights real Ultralytics vehicle tracking, Depth Anything V2 Metric Outdoor, and Cityscapes SegFormer checkpoints before reading the video. Results use explicit `MEASURED`, `ESTIMATED`, and `UNAVAILABLE` statuses.

This baseline does not convert Cityscapes road masks into lane markings and does not claim junction topology without a validated lane/topology checkpoint. Those adapters remain unavailable until a checkpoint validated for the target dashcam domain is supplied. That gate is intentional.