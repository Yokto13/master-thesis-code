import gin
import numpy as np
import torch


def _to_float32(x: np.ndarray) -> np.ndarray:
    # numpy's float16 <-> float32 casts are scalar loops; torch's are SIMD (~10x faster).
    return torch.from_numpy(x).float().numpy()


def _to_float16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(x).half().numpy()


@gin.configurable
class MultiEnvReplayBuffer:
    """Replay buffer for vectorized environments.

    Storage is one ring buffer per environment, shaped (num_envs, capacity_per_env, ...).
    All environments advance in lockstep: every `push` writes one transition for each
    env at the shared write position, so a single scalar position/size describes all rings.

    Sampled sequences are arrival-aligned: at index t a row holds the observation of
    step t together with the action/reward/done produced by step t-1 (what the RSSM
    consumes when processing observation t). Sequences never cross environments.

    Online rows: the first min(num_envs, batch_size) rows of every batch begin with the
    freshest transitions pushed since the last sample (capped at seq_len), one env per
    row, followed by the beginning of an ordinary random sequence. This guarantees new
    experience is trained on immediately, before it has a chance to be sampled.
    """

    def __init__(self, capacity, num_envs, C, W, H, action_dim, rnn_deter_dim, rnn_stoch_dim):
        self.capacity = capacity // num_envs  # slots per env
        self.num_envs = num_envs
        self.action_dim = action_dim

        E, cap = num_envs, self.capacity
        self.states = np.empty((E, cap, C, W, H), dtype=np.uint8)
        self.actions = np.empty((E, cap), dtype=np.uint8)
        self.rewards = np.empty((E, cap), dtype=np.float32)
        self.dones = np.empty((E, cap), dtype=bool)
        self.is_first = np.empty((E, cap), dtype=bool)
        self.rnn_deter_state = np.empty((E, cap, rnn_deter_dim), dtype=np.float16)
        self.rnn_stoch_state = np.empty((E, cap, rnn_stoch_dim), dtype=np.float16)

        self.position = 0  # next slot to write, shared by all rings
        self.size = 0  # filled slots per ring
        self._last_done = np.ones(E, dtype=bool)  # first transition of each env starts an episode
        self._steps_since_sample = 0

    def push(self, states, actions, rewards, dones, deter_states, stoch_states, is_first=None):
        """Store one transition per environment. All arguments have leading dim num_envs."""
        p = self.position
        self.states[:, p] = states
        self.actions[:, p] = actions
        self.rewards[:, p] = rewards
        self.dones[:, p] = dones
        if is_first is not None:
            self.is_first[:, p] = is_first
        else:
            self.is_first[:, p] = self._last_done
        self.rnn_deter_state[:, p] = deter_states
        self.rnn_stoch_state[:, p] = stoch_states

        np.copyto(self._last_done, dones)
        self.position = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._steps_since_sample += 1

    def sample(self, batch_size: int, seq_len: int, return_rnn_indices: bool = False) -> dict:
        if self.size < seq_len:
            raise ValueError("Buffer contains fewer items than the requested sequence length!")

        # Random sequences: a logical start is an offset from the oldest stored item, so
        # physical = (position + logical) % capacity once the ring is full (position then
        # points at the oldest item), and physical = logical before that.
        starts = np.random.randint(0, self.size - seq_len + 1, size=batch_size)
        if self.size == self.capacity:
            starts = (self.position + starts) % self.capacity
        if self.num_envs > 1:
            env_idx = np.random.randint(0, self.num_envs, size=batch_size)
        else:
            env_idx = np.zeros(batch_size, dtype=np.int64)

        data_idx = (starts[:, None] + np.arange(seq_len)) % self.size

        # Online rows: row e starts with env e's freshest m transitions, then continues
        # into the beginning of that row's random sequence.
        m = min(self._steps_since_sample, seq_len, self.size)
        n_online = min(self.num_envs, batch_size)
        if m > 0:
            data_idx[:n_online, :m] = (self.position - m + np.arange(m)) % self.size
            data_idx[:n_online, m:] = (starts[:n_online, None] + np.arange(seq_len - m)) % self.size
            env_idx[:n_online] = np.arange(n_online)
        self._steps_since_sample = 0

        # Arrival alignment: action/reward/done come from the previous slot.
        action_idx = (data_idx - 1) % self.size
        rows = env_idx[:, None]

        prev_actions = self.actions[rows, action_idx]
        batch = {
            "states": self.states[rows, data_idx],
            "actions": np.zeros((batch_size, seq_len, self.action_dim), dtype=np.float32),
            "rewards": self.rewards[rows, action_idx],
            "dones": self.dones[rows, action_idx],
            "is_first": self.is_first[rows, data_idx],
            "rnn_states": {
                "deter": _to_float32(self.rnn_deter_state[env_idx, data_idx[:, 0]]),
                "stoch": _to_float32(self.rnn_stoch_state[env_idx, data_idx[:, 0]]),
            },
        }
        b_idx = np.arange(batch_size)[:, None]
        t_idx = np.arange(seq_len)[None, :]
        batch["actions"][b_idx, t_idx, prev_actions] = 1.0

        # The head is the oldest stored slot; reading the "previous" slot there wraps to
        # garbage, so treat it like an episode start with unknown history.
        head = self.position if self.size == self.capacity else 0
        at_head = data_idx == head

        batch["actions"][batch["is_first"] | at_head] = 0.0  # no valid previous action
        batch["rewards"][at_head] = 0.0  # previous reward lost
        batch["dones"][batch["is_first"]] = True  # previous episode ended here
        batch["dones"][at_head & ~batch["is_first"]] = False  # history lost, not a real episode end

        if return_rnn_indices:
            return batch, env_idx[:, None] * self.capacity + data_idx
        return batch

    def update_rnn_states(self, indices: np.ndarray, deter_states: np.ndarray, stoch_states: np.ndarray):
        """Write back refreshed RNN states at flat indices (env * capacity + slot) from sample()."""
        if deter_states.dtype != np.float16:
            deter_states = _to_float16(deter_states)
            stoch_states = _to_float16(stoch_states)
        self.rnn_deter_state.reshape(-1, self.rnn_deter_state.shape[-1])[indices] = deter_states
        self.rnn_stoch_state.reshape(-1, self.rnn_stoch_state.shape[-1])[indices] = stoch_states

    _ARRAY_KEYS = ("states", "actions", "rewards", "dones", "is_first", "rnn_deter_state", "rnn_stoch_state")

    def _logical(self, arr: np.ndarray) -> np.ndarray:
        """Filled slots in oldest-to-newest order, shape (num_envs, size, ...)."""
        if self.size == self.capacity and self.position != 0:
            return np.concatenate([arr[:, self.position :], arr[:, : self.position]], axis=1)
        return arr[:, : self.size]

    def save_obs(self, path, k):
        obs = self.states[:, : self.size : k]
        np.save(path, obs.reshape(-1, *obs.shape[2:]))

    def save(self, path):
        np.savez(
            path,
            **{key: self._logical(getattr(self, key)) for key in self._ARRAY_KEYS},
            last_done=self._last_done,
            metadata=np.array([self.size, self.capacity, self.action_dim]),
        )

    @classmethod
    def load(cls, path, reset_rnn_states=False, capacity_override=None):
        data = np.load(path, allow_pickle=False)
        meta = data["metadata"].astype(int)

        if len(meta) == 11:
            # Legacy SingleEnvFastReplayBuffer format: single-env arrays in physical
            # order, metadata = [position, size, capacity, C, W, H, action_dim,
            # rnn_deter_dim, rnn_stoch_dim, last_done, episode_step].
            position, size, capacity, action_dim = meta[0], meta[1], meta[2], meta[6]
            last_done = np.array([bool(meta[9])])
            roll = -position if size == capacity else 0  # to oldest-first order
            arrays = {key: np.roll(data[key], roll, axis=0)[None] for key in cls._ARRAY_KEYS}
        else:
            size, capacity, action_dim = meta
            last_done = data["last_done"]
            arrays = {key: data[key] for key in cls._ARRAY_KEYS}

        num_envs = arrays["states"].shape[0]
        C, W, H = arrays["states"].shape[2:]
        total_capacity = capacity_override if capacity_override is not None else capacity * num_envs
        buf = cls(
            total_capacity,
            num_envs,
            C,
            W,
            H,
            action_dim,
            rnn_deter_dim=arrays["rnn_deter_state"].shape[-1],
            rnn_stoch_dim=arrays["rnn_stoch_state"].shape[-1],
        )

        keep = min(size, buf.capacity)  # if capacity shrank, keep only the newest transitions
        for key in cls._ARRAY_KEYS:
            getattr(buf, key)[:, :keep] = arrays[key][:, size - keep : size]
        buf.size = keep
        buf.position = keep % buf.capacity  # 0 when full: oldest slot is index 0 after logical-order save
        buf._last_done = last_done.astype(bool)

        if reset_rnn_states:
            buf.rnn_deter_state[:, :keep] = 0
            buf.rnn_stoch_state[:, :keep] = 0
        return buf

    def __len__(self):
        return self.size * self.num_envs
