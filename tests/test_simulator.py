from system_one.actions import Action
from system_one.simulator import Service, compare_controllers


def test_scaling_reduces_pressure_restart_clears_fault_and_capacity_is_bounded():
    service = Service(replicas=2, fault=.2)
    before = service.observe(1200)
    service.apply(Action.SCALE_OUT)
    assert service.observe(1200).latency < before.latency
    service.apply(Action.RESTART)
    assert service.observe(1200).error_rate == 0
    for _ in range(30):
        service.apply(Action.SCALE_IN)
    assert service.replicas == 1
    for _ in range(30):
        service.apply(Action.SCALE_OUT)
    assert service.replicas == 10


def test_simulation_is_reproducible_and_escalation_does_not_invent_remediation():
    class Model:
        def decide(self, state):
            from system_one.decision import Decision
            return Decision("ESCALATE", .5, {})
    first = compare_controllers(Model(), seeds=(1,), steps=3)
    assert first == compare_controllers(Model(), seeds=(1,), steps=3)
    fault = first["scenarios"]["recoverable_fault"]
    assert fault["model"]["mean_error_rate"] == fault["no_action"]["mean_error_rate"]
    assert fault["rule"]["mean_error_rate"] < fault["no_action"]["mean_error_rate"]
