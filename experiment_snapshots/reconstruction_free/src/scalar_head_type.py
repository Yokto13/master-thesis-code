from enum import Enum, auto


class ScalarHeadType(Enum):
    """
    Used to set the type of scalar head (critic, rewards, target critic).
    """

    MSE = auto()
    TWO_HOT = auto()
    HL_GAUSS = auto()
