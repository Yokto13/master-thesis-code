from enum import Enum, auto


class DebugMode(Enum):
    OFF = auto()
    # RL is to verify that the code without WM works.
    # Should run policy value learning + encoder.
    RL = auto()
