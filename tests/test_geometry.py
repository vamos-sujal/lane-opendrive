from src.common import MetricState


def test_metric_status_enum_is_known():
    assert MetricState.METRIC_VALID.value == "METRIC_VALID"
    assert MetricState.METRIC_UNCERTAIN.value == "METRIC_UNCERTAIN"
    assert MetricState.NON_METRIC.value == "NON_METRIC"
    assert MetricState.INVALID.value == "INVALID"
