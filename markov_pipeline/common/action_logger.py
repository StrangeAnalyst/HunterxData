"""ActionLogger — structured, auditable logging for every pipeline action.

Each meaningful action (a guardrail check, a model fit, a table write, a drift
evaluation) is logged as a structured record so the full run is reconstructable
for model governance. When MLflow is active, key/value summaries are also pushed
as tags/params for the audit trail.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@dataclass
class ActionLogger:
    """Structured logger that records ordered, typed pipeline actions.

    Attributes:
        component: Name of the pipeline component (used as logger name).
        records: In-memory list of structured action records.
    """

    component: str
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._logger = _build_logger(f"markov.{self.component}")

    # ------------------------------------------------------------------ #
    # Core logging
    # ------------------------------------------------------------------ #
    def log(self, action: str, level: str = "INFO", **payload: Any) -> None:
        """Record a structured action.

        Args:
            action: Short machine-readable action name (e.g. ``"fit_row_model"``).
            level: Log level name (``INFO``/``WARNING``/``ERROR``).
            **payload: JSON-serialisable context for the action.
        """
        record = {
            "ts": time.time(),
            "component": self.component,
            "action": action,
            "level": level,
            **payload,
        }
        self.records.append(record)
        self._logger.log(
            getattr(logging, level, logging.INFO),
            "%s %s",
            action,
            json.dumps(payload, default=str, sort_keys=True),
        )

    def info(self, action: str, **payload: Any) -> None:
        """Log an INFO-level action."""
        self.log(action, level="INFO", **payload)

    def warn(self, action: str, **payload: Any) -> None:
        """Log a WARNING-level action."""
        self.log(action, level="WARNING", **payload)

    def error(self, action: str, **payload: Any) -> None:
        """Log an ERROR-level action."""
        self.log(action, level="ERROR", **payload)

    # ------------------------------------------------------------------ #
    # Timing context manager
    # ------------------------------------------------------------------ #
    @contextmanager
    def timed(self, action: str, **payload: Any) -> Iterator[None]:
        """Context manager that logs start/end and elapsed seconds for ``action``."""
        start = time.time()
        self.info(f"{action}.start", **payload)
        try:
            yield
        finally:
            elapsed = round(time.time() - start, 4)
            self.info(f"{action}.end", elapsed_s=elapsed, **payload)

    # ------------------------------------------------------------------ #
    # MLflow bridge
    # ------------------------------------------------------------------ #
    def flush_to_mlflow(self) -> None:
        """Best-effort push of a compact action summary to the active MLflow run.

        Silently no-ops when MLflow is unavailable or there is no active run, so
        the logger never becomes a hard dependency in unit tests.
        """
        try:
            import mlflow  # local import — optional dependency

            if mlflow.active_run() is None:
                return
            counts: dict[str, int] = {}
            for rec in self.records:
                key = f"action_count.{rec['action']}"
                counts[key] = counts.get(key, 0) + 1
            mlflow.set_tags({f"{self.component}.{k}": v for k, v in counts.items()})
        except Exception:  # pragma: no cover - defensive, never block the run
            self._logger.debug("MLflow flush skipped (unavailable or no active run)")
