# app/services/circuit_breaker.py
"""Implementación de circuit breaker para aislar fallos sostenidos
del backend de inferencia primario.

Estados: CLOSED (normal) -> OPEN (fallback activo) -> HALF_OPEN (prueba
de recuperación) -> CLOSED o de vuelta a OPEN.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_attempts: int = 1

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _half_open_attempts: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._half_open_attempts = 0
        return self._state

    def allow_request_to_primary(self) -> bool:
        current_state = self.state
        if current_state is CircuitState.CLOSED:
            return True
        if current_state is CircuitState.HALF_OPEN:
            if self._half_open_attempts < self.half_open_max_attempts:
                self._half_open_attempts += 1
                return True
            return False
        return False  # OPEN

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1

        if self.state is CircuitState.HALF_OPEN:
            # Un fallo durante la prueba de recuperación reabre el circuito inmediatamente
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            return

        if self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def force_open(self) -> None:
        """Permite forzar la apertura del circuito (p. ej. ante drift crítico)."""
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()

    def force_close(self) -> None:
        """Permite forzar el cierre del circuito tras resolución manual."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
