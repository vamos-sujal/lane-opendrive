from src.common import LaneObservation, MetricState
from src.tracking.tracker import LaneTracker


def test_track_reappears_with_same_id():
    tracker = LaneTracker()
    obs1 = LaneObservation(
        detection_id="a",
        frame_index=0,
        timestamp=0.0,
        points_3d=[[0.0, 0.0, 0.0], [2.0, 10.0, 0.0]],
        confidence=0.7,
        visibility=0.8,
        lane_marking="SOLID",
        image_points=[[10, 10], [50, 200]],
        world_points=[[0.0, 0.0, 0.0], [2.0, 10.0, 0.0]],
        uncertainty=0.1,
        metric_validity=MetricState.METRIC_UNCERTAIN,
    )
    tracker.update([obs1], 0)
    assert len(tracker.tracks) == 1
    first_id = next(iter(tracker.tracks))

    tracker.update([], 3)
    assert tracker.tracks[first_id].state.value == "temporarily_lost"

    obs2 = LaneObservation(
        detection_id="a-recovered",
        frame_index=5,
        timestamp=1.0,
        points_3d=[[0.0, 0.0, 0.0], [2.0, 12.0, 0.0]],
        confidence=0.8,
        visibility=0.7,
        lane_marking="DASHED",
        image_points=[[11, 11], [52, 202]],
        world_points=[[0.0, 0.0, 0.0], [2.0, 12.0, 0.0]],
        uncertainty=0.1,
        metric_validity=MetricState.METRIC_UNCERTAIN,
    )
    tracker.update([obs2], 5)
    assert list(tracker.tracks.keys())[0] == first_id
