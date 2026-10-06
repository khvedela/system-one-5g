"""Valid, deliberately shifted probes; never used in training or selection."""

import numpy as np

from .state import State


def ood_states(seed: int = 42, samples_per_scenario: int = 100) -> dict[str, list[State]]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    scenarios = {"extreme_latency": [], "high_cpu_low_traffic": [], "near_range_limits": []}
    for _ in range(samples_per_scenario):
        scenarios["extreme_latency"].append(State(
            rng.uniform(30, 60), rng.uniform(30, 60), rng.uniform(100, 1000),
            rng.uniform(5000, 10_000), rng.uniform(0, .02), int(rng.integers(1, 11))))
        scenarios["high_cpu_low_traffic"].append(State(
            rng.uniform(98, 100), rng.uniform(10, 30), rng.uniform(0, 10),
            rng.uniform(0, 20), 0., int(rng.integers(1, 11))))
        scenarios["near_range_limits"].append(State(
            rng.uniform(98, 100), rng.uniform(98, 100), rng.uniform(9000, 10_000),
            rng.uniform(9000, 10_000), rng.uniform(.9, 1), int(rng.integers(11, 21))))
    return scenarios
