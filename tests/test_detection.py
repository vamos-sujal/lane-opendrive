from src.common import LaneObservation, MetricState
from src.detection.postprocess import coerce_observations


def test_coerce_observations_and_metric_state():
    raw = [{
        "detection_id": "lane-1",
        "frame_index": 10,
        "timestamp": 0.5,
        "points_3d": [[0.0, 0.0, 0.0], [2.0, 12.0, 0.0]],
        "confidence": 0.8,
        "visibility": 0.9,
        "lane_marking": "SOLID",
        "image_points": [[100, 200], [300, 400]],
        "world_points": [[0.0, 0.0, 0.0], [2.0, 12.0, 0.0]],
        "uncertainty": 0.1,
        "metric_validity": MetricState.METRIC_UNCERTAIN.value,
    }]

    obs = coerce_observations(raw)
    assert len(obs) == 1
    assert isinstance(obs[0], LaneObservation)
    assert obs[0].lane_marking == "SOLID"
    assert obs[0].metric_validity == MetricState.METRIC_UNCERTAIN
