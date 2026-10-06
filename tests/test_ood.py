from system_one.ood import ood_states
from system_one.policy import ground_truth
from system_one.actions import Action


def test_ood_scenarios_are_seeded_and_outside_clean_training_support():
    first = ood_states(samples_per_scenario=5)
    assert first == ood_states(samples_per_scenario=5)
    assert len(first) == 3
    assert all(state.latency >= 5000 for state in first["extreme_latency"])
    assert all(state.cpu >= 98 and state.request_rate <= 10 for state in first["high_cpu_low_traffic"])
    assert all(state.replicas > 10 for state in first["near_range_limits"])
    assert all(ground_truth(state) == Action.ESCALATE for state in first["near_range_limits"])
