"""Tests para observabilidad, actualización de métricas de drift y drift gate."""

from app.observability.drift_metrics import update_drift_gauges
from app.observability.metrics import DATA_DRIFT_SHARE_GAUGE, DATASET_DRIFT_DETECTED_GAUGE


def test_update_drift_gauges() -> None:
    update_drift_gauges(dataset_drift=True, drift_share=0.45, concept_drift=0.03)
    assert DATASET_DRIFT_DETECTED_GAUGE._value.get() == 1.0  # noqa: SLF001
    assert abs(DATA_DRIFT_SHARE_GAUGE._value.get() - 0.45) < 1e-5  # noqa: SLF001

    update_drift_gauges(dataset_drift=False, drift_share=0.10, concept_drift=0.01)
    assert DATASET_DRIFT_DETECTED_GAUGE._value.get() == 0.0  # noqa: SLF001
    assert abs(DATA_DRIFT_SHARE_GAUGE._value.get() - 0.10) < 1e-5  # noqa: SLF001
