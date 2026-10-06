"""Frozen v1 synthetic oracle; thresholds are experiment assumptions."""

from .actions import Action
from .state import State


POLICY_VERSION = "server-policy-v1"
POLICY_THRESHOLDS = {
    "high_error": 0.10,
    "low_error": 0.01,
    "high_cpu": 75.0,
    "restart_cpu_limit": 85.0,
    "high_memory": 85.0,
    "high_latency": 200.0,
    "high_traffic_per_replica": 300.0,
    "low_cpu": 25.0,
    "low_memory": 40.0,
    "low_latency": 100.0,
    "low_traffic_per_replica": 100.0,
}


def ground_truth(state: State) -> Action:
    """First matching rule wins. Never tuned using model results."""
    t = POLICY_THRESHOLDS
    traffic = state.request_rate / state.replicas
    if state.error_rate >= t["high_error"]:
        return Action.ESCALATE if state.cpu >= t["restart_cpu_limit"] else Action.RESTART
    resource_pressure = state.cpu >= t["high_cpu"] or state.memory >= t["high_memory"]
    if resource_pressure:
        if state.latency >= t["high_latency"] and traffic >= t["high_traffic_per_replica"]:
            return Action.SCALE_OUT
        return Action.ESCALATE
    if (state.replicas > 1 and state.cpu <= t["low_cpu"]
            and state.memory <= t["low_memory"] and state.latency <= t["low_latency"]
            and traffic <= t["low_traffic_per_replica"] and state.error_rate <= t["low_error"]):
        return Action.SCALE_IN
    return Action.NO_ACTION
