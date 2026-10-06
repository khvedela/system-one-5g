"""State units and physical validation, separate from training support."""

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real


FEATURE_NAMES = ("cpu", "memory", "request_rate", "latency", "error_rate", "replicas")
RANGES = {
    "cpu": (0, 100),
    "memory": (0, 100),
    "request_rate": (0, 10_000),
    "latency": (0, 10_000),
    "error_rate": (0, 1),
    "replicas": (1, 20),
}


@dataclass(frozen=True)
class State:
    cpu: float
    memory: float
    request_rate: float
    latency: float
    error_rate: float
    replicas: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            lower, upper = RANGES[field.name]
            if not lower <= value <= upper:
                raise ValueError(f"{field.name} must be in [{lower}, {upper}]")
        if not isinstance(self.replicas, Integral):
            raise ValueError("replicas must be an integer")

    def values(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_NAMES]

    def as_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in FEATURE_NAMES}
