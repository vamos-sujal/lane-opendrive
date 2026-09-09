# Architecture decision

## Candidates investigated

### 1. LATR
- Official repository: https://github.com/JMoonr/LATR
- Paper: LATR: 3D Lane Detection from Monocular Images with Transformer
- Status: active, official implementation, benchmarked on OpenLane/Apollo/ONCE
- Model type: single-camera monocular 3D lane detector
- Output: 3D lane polylines / lane representations in a road coordinate frame, with reported 3D errors on benchmark datasets
- Strengths: strong benchmark results, direct single-camera 3D lane estimation, official checkpoints, clear research lineage
- Limitations: original environment is older (PyTorch 1.8, CUDA 10.1/10.2, mmdet/mmdet3d stack), not a turnkey Colab/T4 environment; video inference requires a carefully pinned environment and an explicit checkpoint workflow

### 2. Anchor3DLane
- Official repository: https://github.com/tusen-ai/Anchor3DLane
- Status: official implementation, active maintenance, CVPR 2023
- Model type: monocular 3D lane regression using 3D anchors
- Strengths: strong benchmark performance and clear 3D anchor formulation
- Limitations: also tied to older training/runtime infrastructure; less straightforward to deploy as a no-fuss Colab inference pipeline than a modern package ecosystem; requires careful compatibility work and explicit checkpoint validation

### 3. OpenLane / OpenLane-V2 ecosystem
- Official project: OpenDriveLab / OpenLane
- Status: major dataset and benchmark ecosystem
- Strengths: valuable for dataset semantics, lane topology language, and road-graph reasoning
- Limitations: not a single plug-and-play detector for arbitrary MP4 under a simple Colab setup; more useful as benchmark and semantics source than as direct inference engine for a production-facing system

### 4. Other monocular 3D lane models
- PersFormer, CurveFormer, GenLaneNet, Cond-IPM, and related work were reviewed as part of the design space
- They are relevant as research baselines, but they do not eliminate the need for explicit metric validation and a fail-closed scale gate
- None of them changes the central engineering truth: monocular 3D lane detection alone is not equivalent to calibrated metric 3D without valid camera geometry or an explicit scale source

## Comparison

| Candidate | Single-camera 3D lane | Official checkpoint | Colab/T4 practicality | Metric honesty | Notes |
| --- | --- | --- | --- | --- | --- |
| LATR | Yes | Yes | Moderate to good with pinned deps | Good if calibration is explicit | Best balance of benchmark quality and direct 3D lane output |
| Anchor3DLane | Yes | Yes | Moderate | Good if calibration is explicit | Strong alternative, but more environment friction |
| OpenLane dataset ecosystem | Partial | N/A | Good for data semantics | Not a complete metric detector | Better as benchmark/graph research reference |
| Generic 2D segmentation + IPM | No | Not applicable | Easy but misleading | Poor | Must not be used as metric 3D |

## Selected model

Selected: LATR (official JMoonr/LATR implementation)

## Exact reason for selection

1. It is a genuine monocular 3D lane detector, not a 2D segmentation + callee-based IPM trick.
2. The official repository provides a clear model family and benchmark claims.
3. The outputs are explicitly evaluated in 3D coordinate space, including x/z errors in benchmark datasets.
4. It is a better engineering fit for the target than ad hoc perspective-to-ground geometry.
5. It is a realistic single-camera detector candidate for a T4 GPU environment when paired with a version-pinned, reproducible install and explicit failure handling.

## Known limitations

- The official implementation targets an older MMDetection stack and older PyTorch/CUDA versions.
- Arbitrary MP4 video input is not a turnkey one-click repo feature without custom ingestion and checkpoint loading code.
- Metric scale remains underdetermined from a single monocular video unless valid calibration or telemetry is supplied.
- Without a trusted camera model and scale source, 3D outputs are not metric and must stay flagged as uncertain.
- The detector does not solve topology, persistent lane identity, or OpenDRIVE validity by itself.

## Dependency versions

The official LATR repo documents older versions in its install guide:

- PyTorch 1.8.0
- torchvision 0.9.0
- torchaudio 0.8.0
- CUDA 10.1
- mmcv 1.5.0
- mmdet 2.24.0
- mmdet3d 1.0.0rc3

This project keeps a modern reproducible setup for Colab/T4 by using:

- Python 3.10
- PyTorch 2.2+ / 2.4 compatible with T4 + CUDA 12.1
- OpenCV 4.x
- SciPy, NumPy, Shapely, PyYAML, lxml, matplotlib, pytest, and packaging-managed deps

The roadmap is to run the official model in a pinned environment when compatible; otherwise the pipeline will still run, but it must fail closed and clearly state metric uncertainty.

## Expected GPU requirements

- Minimum practical target: T4 GPU
- Recommended: 16 GB VRAM or better for inference, checkpoint loading, and moderate frame pipelines
- Mixed precision should be used only when supported and safe
- Inference speed is not a substitute for geometric correctness

## Metric limitations

This pipeline explicitly does not assume any of the following without evidence:

- focal length
- principal point
- camera height
- pitch/roll
- lane width
- vehicle speed
- arbitrary scale multiplier
- length of road without measurement

If the project is run without valid calibration or telemetry, the metric state is reported as `METRIC_UNCERTAIN` or `INVALID`, and the system does not fabricate metric OpenDRIVE data.

## What the model does NOT solve

- lane topology inference across long occlusions by itself
- persistent physical lane IDs without a tracker
- metric scale determination from a single generic dashcam video alone
- valid OpenDRIVE generation without validation
- uncertain geometry conversion into a metric road model without calibration

## Verified checkpoint artifact

The official LATR README publishes the OpenLane-1000 checkpoint at Google Drive file ID `1jThvqnJ2cUaAuKdlTuRKjhLCH0Zq62A1` with MD5 `d8ecb900c34fd23a9e7af840aff00843`. These values are recorded in `configs/model.yaml` and are checked by `scripts/download_weights.py`.

## Deployment decision

The checkpoint is verified as an official artifact, but the detector implementation in this repository is not yet a real adapter around the official LATR/MMDetection graph. The official dependency set is incompatible with the modern Colab runtime without an isolated legacy environment. Therefore the pipeline must not label its placeholder observations as LATR detections or emit metric OpenDRIVE.

## Final engineering decision

For arbitrary MP4 deployment, the Colab default is the official Ultra-Fast-Lane-Detection CULane checkpoint. It is a verified pretrained 2D lane detector with direct image inference and is therefore suitable for actual overlays and temporal lane tracking. It is not a metric 3D model; all metric and OpenDRIVE outputs remain gated by calibration.

The selected detector is LATR because it is the most credible official monocular 3D lane detector among the options reviewed, and it matches the project intent better than 2D segmentation plus IPM. However, this repository implements the required safety constraints: it verifies calibration, documents dependency constraints, and refuses to invent metric geometry when the information is not present.
