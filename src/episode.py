from typing import NamedTuple, Optional

import numpy as np


class Episode(NamedTuple):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    deter_states: Optional[np.ndarray]
    stoch_states: Optional[np.ndarray]
    # Optionaly store environment states for debug modes
    env_states: Optional[np.ndarray] = None


def get_episode_length(episode: Episode) -> int:
    return len(episode.rewards)
