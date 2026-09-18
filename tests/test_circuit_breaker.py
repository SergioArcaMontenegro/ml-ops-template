"""Tests unitarios para la máquina de estados del Circuit Breaker (Capítulo 6)."""

import time

from app.services.circuit_breaker import CircuitBreaker, CircuitState


def test_circuit_breaker_initial_state() -> None:
    cb = CircuitBreaker(failure_threshold=3)
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request_to_primary() is True


def test_circuit_breaker_opens_after_threshold() -> None:
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request_to_primary() is False


def test_circuit_breaker_half_open_recovery() -> None:
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    time.sleep(0.1)
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.allow_request_to_primary() is True

    # Éxito en HALF_OPEN -> vuelve a CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request_to_primary() is True


def test_circuit_breaker_half_open_failure() -> None:
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.05)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    time.sleep(0.1)
    assert cb.state == CircuitState.HALF_OPEN
    # Fallo en HALF_OPEN -> vuelve inmediatamente a OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
