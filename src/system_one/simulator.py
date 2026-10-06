"""Versioned toy action-effect model; outcomes are not real service evidence."""

from dataclasses import dataclass
from collections import Counter

import numpy as np

from .actions import Action
from .policy import ground_truth
from .state import State


SIMULATOR_VERSION = "service-effects-v1"


@dataclass
class Service:
    replicas: int = 2
    fault: float = 0.0
    background_cpu: float = 0.0

    def observe(self, requests: float) -> State:
        utilization = requests / (self.replicas * 500)
        cpu = min(100, 10 + 90 * utilization + self.background_cpu)
        memory = min(100, 20 + 45 * utilization)
        latency = min(10_000, 10 / max(.01, 1 - utilization))
        errors = min(1, self.fault + max(0, utilization - 1) * .1)
        return State(cpu, memory, requests, latency, errors, self.replicas)

    def apply(self, action: Action) -> float:
        if action == Action.SCALE_OUT:
            self.replicas = min(10, self.replicas + 1)
            return .2
        if action == Action.SCALE_IN:
            self.replicas = max(1, self.replicas - 1)
            return .2
        if action == Action.RESTART:
            self.fault = 0
            return .5
        return 1.0 if action == Action.ESCALATE else 0.0


def compare_controllers(model, seeds: tuple[int, ...] = (42, 43, 44), steps: int = 60) -> dict:
    results = {}
    scenarios = {"load_burst": (2, 0., 0.), "low_load": (5, 0., 0.),
                 "recoverable_fault": (2, .2, 0.), "background_cpu": (2, 0., 85.)}
    for name, initial in scenarios.items():
        controllers = {"no_action": lambda state: Action.NO_ACTION, "rule": ground_truth,
                       "model": lambda state: Action[model.decide(state).action]}
        rows = {}
        for controller_name, controller in controllers.items():
            costs, latencies, errors, replicas, actions = [], [], [], [], Counter()
            for seed in seeds:
                rng = np.random.default_rng(seed)
                base = 1700 if name == "load_burst" else (100 if name in ("low_load", "background_cpu") else 500)
                traffic = np.clip(rng.normal(base, base * .03, steps), 0, 10_000)
                service = Service(*initial)
                for requests in traffic:
                    state = service.observe(float(requests))
                    action = controller(state)
                    action_cost = service.apply(action)
                    after = service.observe(float(requests))
                    # Explicit dimensionless toy cost: latency/SLO + error penalty + replica and action cost.
                    costs.append(after.latency / 200 + 20 * after.error_rate + .1 * after.replicas + action_cost)
                    latencies.append(after.latency)
                    errors.append(after.error_rate)
                    replicas.append(after.replicas)
                    actions[action.name] += 1
            rows[controller_name] = {"mean_cost": float(np.mean(costs)), "mean_latency_ms": float(np.mean(latencies)),
                                     "mean_error_rate": float(np.mean(errors)), "mean_replicas": float(np.mean(replicas)),
                                     "actions": dict(actions)}
        results[name] = rows
    return {"version": SIMULATOR_VERSION, "seeds": seeds, "steps": steps,
            "cost_definition": "latency_ms/200 + 20*error_rate + 0.1*replicas + action_cost",
            "scope": "toy capacity/fault model; restart immediately clears modeled fault; "
                     "no human remediation is assumed for escalation; not real-network validation",
            "scenarios": results}
