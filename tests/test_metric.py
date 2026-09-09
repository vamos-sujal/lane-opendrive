from src.common import LaneObservation, MetricState
from src.geometry.metric import GroundPlaneCalibration, image_to_ground, project_observation


def test_missing_calibration_stays_non_metric():
    observation = LaneObservation(
        detection_id="lane", frame_index=0, timestamp=0.0,
        image_points=[[640, 360], [641, 361], [642, 362]],
    )
    assert project_observation(observation, None).metric_validity == MetricState.NON_METRIC


def test_principal_ray_intersects_ground_in_front_of_camera():
    calibration = GroundPlaneCalibration(1000.0, 1000.0, 640.0, 360.0, 1.5, 10.0)
    ground = image_to_ground((640.0, 360.0), calibration)
    assert ground is not None
    assert abs(ground[0]) < 1e-9
    assert ground[1] > 0.0