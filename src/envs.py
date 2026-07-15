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
            1: np.array([-1.0, 0.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),
            3: np.array([0.0, 1.0, 0.0]),
            4: np.array([0.0, 0.0, 0.8]),
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


class CrafterEnv(gymnasium.Env):
    """Adapter for crafter.Env (old gym API) to gymnasium.

    Renders natively at 64x64, so no resize or frame skip is applied.
    Death maps to terminated (info["discount"] == 0), Crafter's internal
    10k-step limit to truncated. info["achievements"] passes through for
    the trainer's Crafter score logging.
    """

    def __init__(self, seed=None):
        import crafter

        self._env = crafter.Env(size=(64, 64), reward=True, seed=seed)
        self.action_space = gymnasium.spaces.Discrete(self._env.action_space.n)
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

    def reset(self, **kwargs):
        obs = self._env.reset()
        return self._process_obs(obs), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        terminated = done and info["discount"] == 0
        truncated = done and not terminated
        return self._process_obs(obs), reward, terminated, truncated, info

    def _process_obs(self, obs):
        return np.transpose(obs, (2, 0, 1)).astype(np.uint8)


class FrameSkip(gymnasium.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        logger.warning(
            "FrameSkipping implementations often max pool last two frames. This implementation does not do that!"
        )

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
    if env_id == "crafter":
        return CrafterEnv(seed=seed)
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
