from enum import IntEnum


class Action(IntEnum):
    NO_ACTION = 0
    SCALE_IN = 1
    SCALE_OUT = 2
    RESTART = 3
    ESCALATE = 4


ACTION_NAMES = tuple(action.name for action in Action)
