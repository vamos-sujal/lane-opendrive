# Production road-perception pipeline

`notebooks/Production_RoadMetrics_Colab.ipynb` is the new T4 runtime path. It validates the environment and loads real pretrained models before processing a video:

- Ultralytics YOLO11m for vehicle detection and tracking.
- Depth Anything V2 Metric Outdoor Base for model-based outdoor depth.
- SegFormer B2 Cityscapes for road-scene semantics.

The model registry is in `src/production/model_registry.py`; the result contract is in `src/production/contracts.py`; execution is in `src/production/runner.py`.

## Status semantics

Every metric has a value, unit, status, source, and optional uncertainty. `MEASURED` requires validated camera geometry and scale. `ESTIMATED` means a pretrained model produced a value but this camera's metric scale has not been independently validated. `UNAVAILABLE` is used when the observation is not identifiable. The pipeline never converts an estimated value into a measured value.

## Current capability gate

Vehicle detection and model-depth estimates are implemented. Lane and junction outputs are intentionally marked unavailable until a lane/topology checkpoint validated for the target dashcam domain is configured. Cityscapes road masks are not treated as lane markings, and model depth is not treated as calibrated distance without camera scale.

The runner also computes a monocular essential-matrix visual-odometry plus depth model-scale estimate for dashcam speed and distance travelled. Vehicle boxes are masked before pose estimation so moving cars do not drive the camera trajectory. These fields are `ESTIMATED_MODEL_SCALE`, not measured metric values, until camera intrinsics and a physical scale source are validated. Road length is limited to the observed/traversed segment; full road extent and OpenDRIVE still require metric lane/topology geometry.

## Colab execution

Run the notebook cells in order. Cell 6 is a preflight that downloads/caches model checkpoints and fails before video processing if a package, CUDA runtime, or checkpoint is incompatible. Cell 8 runs the production baseline and writes `production_report.json` under:

`/content/drive/MyDrive/lane_to_opendrive/production_runs/<run_id>/`

The T4 profile uses moderate checkpoint sizes to leave VRAM for frame tensors. Upgrade a model only after repeating the runtime smoke test on the target GPU.
