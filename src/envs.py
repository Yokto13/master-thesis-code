import logging
from typing import Callable

import ale_py
import gin
import gymnasium
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, ResizeObservation
from PIL import Image

gymnasium.register_envs(ale_py)

logger = logging.getLogger(__name__)


class DreamerCarRacingWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([-0.9, 0.0, 0.0]),
            2: np.array([0.9, 0.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0]),
            4: np.array([0.0, 0.0, 0.8]),
            5: np.array([0.0, 0.5, 0.0]),
            6: np.array([0.0, 0.8, 0.0]),
        }
        self.action_space = gymnasium.spaces.Discrete(5)

        obs_shape = self.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.uint8
        )

    def step(self, action_idx):
        action = self.actions[action_idx]
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._process_obs(obs)
        return obs, info

    def _process_obs(self, obs):
        obs = np.transpose(obs, (2, 0, 1))
        return obs.astype(np.uint8)


class DreamerCartPolePixelsWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_shape = self.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.uint8
        )

    def step(self, action_idx):
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        obs = self._process_obs(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._process_obs(obs)
        return obs, info

    def _process_obs(self, obs):
        obs = np.transpose(obs, (2, 0, 1))
        return obs.astype(np.uint8)


class PongWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_shape = self.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.uint8
        )

    def step(self, action_idx):
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        obs = self._process_obs(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._process_obs(obs)
        return obs, info

    def _process_obs(self, obs):
        obs = obs.astype(np.uint8)
        obs = np.transpose(obs, (2, 0, 1))
        return obs


class FrameSkip(gymnasium.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class PillowResizeObservation(gymnasium.Wrapper):
    def __init__(self, env, shape=(64, 64)):
        super().__init__(env)
        # shape is (height, width), but PIL resize expects (width, height)
        self.size = (shape[1], shape[0])

        # Update observation space
        old_shape = self.observation_space.shape
        new_shape = (shape[0], shape[1], old_shape[2])
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._resize_obs(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._resize_obs(obs)
        return obs, info

    def _resize_obs(self, image):
        image = Image.fromarray(image)
        image = image.resize(self.size, Image.BILINEAR)
        image = np.array(image)
        return image


def get_wrapper(wrapper_type: str) -> Callable[[gymnasium.Env], gymnasium.Env]:
    match wrapper_type:
        case "car_racing":
            return DreamerCarRacingWrapper
        case "cart_pole":
            return DreamerCartPolePixelsWrapper
        case "pong":
            return PongWrapper
        case _:
            raise ValueError(f"Unknown wrapper type: {wrapper_type}")


@gin.configurable
def make_env(env_id: str, wrapper_type: str, frame_skip=1, noop_max=0, resize_engine="gymnasium", seed=None, **kwargs):
    logger.debug("make_env kwargs: %s", kwargs)
    if "ALE" in env_id:
        env = gymnasium.make(env_id, frameskip=1, **kwargs)
        env = AtariPreprocessing(env, frame_skip=frame_skip, noop_max=noop_max, grayscale_obs=False)
    else:
        env = gymnasium.make(env_id, **kwargs)
        if noop_max > 0:
            raise ValueError("noop_max is currently only supported for Atari environments.")
        env = FrameSkip(env, skip=frame_skip)

    if resize_engine == "gymnasium":  # Internally this still uses cv2
        env = ResizeObservation(env, shape=(64, 64))
    elif resize_engine == "pillow":
        env = PillowResizeObservation(env, shape=(64, 64))

    wrapper = get_wrapper(wrapper_type)
    env = wrapper(env)

    return env
