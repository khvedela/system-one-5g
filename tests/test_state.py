import pytest

from system_one.state import State


def test_state_order_and_boundaries():
    assert State(0, 100, 10_000, 10_000, 1, 20).values() == [0, 100, 10_000, 10_000, 1, 20]


@pytest.mark.parametrize("field,value", [
    ("cpu", -1), ("cpu", 101), ("memory", float("nan")),
    ("latency", float("inf")), ("request_rate", -1),
    ("error_rate", 1.1), ("replicas", 0), ("replicas", 2.0),
    ("replicas", True), ("cpu", "50"),
])
def test_invalid_state(field, value):
    values = dict(cpu=50, memory=50, request_rate=500, latency=100, error_rate=0.01, replicas=2)
    values[field] = value
    with pytest.raises(ValueError):
        State(**values)
