import pytest

from system_one.actions import Action
from system_one.policy import ground_truth
from system_one.state import State


@pytest.mark.parametrize("state,action", [
    (State(50, 50, 500, 100, .01, 2), Action.NO_ACTION),
    (State(25, 40, 200, 100, .01, 2), Action.SCALE_IN),
    (State(25, 40, 100, 100, .01, 1), Action.NO_ACTION),
    (State(75, 50, 600, 200, .01, 2), Action.SCALE_OUT),
    (State(50, 85, 600, 200, .01, 2), Action.SCALE_OUT),
    (State(75, 50, 599, 200, .01, 2), Action.ESCALATE),
    (State(75, 50, 600, 199, .01, 2), Action.ESCALATE),
    (State(84.99, 95, 600, 300, .1, 2), Action.RESTART),
    (State(85, 95, 600, 300, .1, 2), Action.ESCALATE),
])
def test_policy_boundaries_and_precedence(state, action):
    assert ground_truth(state) == action


def test_action_class_order_is_stable():
    assert [(a.name, a.value) for a in Action] == [
        ("NO_ACTION", 0), ("SCALE_IN", 1), ("SCALE_OUT", 2), ("RESTART", 3), ("ESCALATE", 4),
    ]
