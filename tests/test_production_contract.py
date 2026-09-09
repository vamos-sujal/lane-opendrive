from src.production.contracts import Measurement, MeasurementStatus, RunContract


def test_metric_contract_preserves_unavailable_status():
    measurement = Measurement(
        None,
        "m/s",
        MeasurementStatus.UNAVAILABLE,
        reason="scale_not_observable",
    )
    payload = measurement.to_dict()
    assert payload["value"] is None
    assert payload["status"] == "UNAVAILABLE"
    assert payload["reason"] == "scale_not_observable"


def test_run_contract_serializes_model_and_metric_state():
    report = RunContract("video.mp4", 10, 10.0, 1.0, "NOT_SUPPLIED", "unknown")
    payload = report.to_dict()
    assert payload["frame_count"] == 10
    assert payload["road_length"]["status"] == "UNAVAILABLE"
