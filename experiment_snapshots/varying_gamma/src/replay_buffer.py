import random
from collections import deque

import gin
import numpy as np
from episode import Episode, get_episode_length


@gin.configurable
class ReplayBuffer:
    def __init__(self, capacity, C, W, H, max_actions):
        assert False, "This class is deprecated. Use FastReplayBuffer instead."
        self.capacity = capacity
        self.buffer = []
        self.total_items = 0

        self.C = C
        self.W = W
        self.H = H
        self.max_actions = max_actions

    def push(self, episode: Episode):
        """Saves a transition."""
        while self.total_items >= self.capacity:
            ep = self.buffer.pop(0)
            self.total_items -= get_episode_length(ep)
        self.buffer.append(episode)
        self.total_items += get_episode_length(episode)

    def sample(self, batch_size: int, seq_len: int) -> dict:
        batch = self._init_batch(batch_size, seq_len)

        for b in range(batch_size):
            seq_idx = 0
            while seq_idx < seq_len:
                episode = random.choice(self.buffer)
                ep_len = get_episode_length(episode)

                start_idx = random.randint(0, ep_len - 1)
                end_idx = min(start_idx + (seq_len - seq_idx), ep_len)
                next_seq_idx = seq_idx + (end_idx - start_idx)

                time_indices = np.arange(seq_idx, next_seq_idx)

                # Set the starting index for RNN reset, the rest is False by default
                batch["is_first"][b, seq_idx] = True

                batch["states"][b, time_indices] = episode.states[start_idx:end_idx]
                batch["rewards"][b, time_indices] = episode.rewards[start_idx:end_idx]
                batch["dones"][b, time_indices] = episode.dones[start_idx:end_idx]

                if start_idx == 0:
                    start_idx += 1
                    seq_idx += 1

                # Do not update the first element as it is a dummy action
                time_indices = np.arange(seq_idx, next_seq_idx)
                action_values = episode.actions[start_idx - 1 : end_idx - 1]

                batch["actions"][b, time_indices, action_values] = 1.0

                seq_idx = next_seq_idx

            assert seq_idx == seq_len

        return batch

    def _init_batch(self, batch_size: int, seq_len: int, dtype=np.float32) -> dict:
        batch = {
            "states": np.zeros((batch_size, seq_len, self.C, self.W, self.H), dtype=np.float32),
            "actions": np.zeros((batch_size, seq_len), dtype=np.float32),
            "rewards": np.zeros((batch_size, seq_len), dtype=np.float32),
            "dones": np.zeros((batch_size, seq_len), dtype=bool),
            "is_first": np.zeros((batch_size, seq_len), dtype=bool),
        }
        return batch


@gin.configurable
class FastReplayBuffer:
    def __init__(self, capacity, C, W, H, action_dim, rnn_deter_dim, rnn_stoch_dim, store_env_states: bool = False):
        self.capacity = capacity
        self.position = 0
        self.last_item_pos = 0
        self.size = 0

        self.C = C
        self.W = W
        self.H = H
        self.action_dim = action_dim
        self.rnn_deter_dim = rnn_deter_dim
        self.rnn_stoch_dim = rnn_stoch_dim

        self.states = np.empty((capacity, C, W, H), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.uint8)
        self.rewards = np.empty(capacity)
        self.dones = np.empty(capacity)
        self.is_first = np.empty(capacity)
        self.rnn_deter_state = np.empty((capacity, rnn_deter_dim), dtype=np.float16)
        self.rnn_stoch_state = np.empty((capacity, rnn_stoch_dim), dtype=np.float16)
        if store_env_states:
            self.env_states = np.empty((capacity,), dtype=object)

        # The idea of episode steps is to be able to identify the beginning of episodes
        # A buffer in which episodes are stored sequentially has a problem that arises when
        # an old episode is partially overwritten by a new episode. In that case, when sampling
        # we must be extra careful to fix cases when we cross episode boundaries.
        self.episode_steps = np.empty(capacity, dtype=np.int32)

    def push(self, episode: Episode):
        ep_len = get_episode_length(episode)
        idxs = np.arange(self.position, self.position + ep_len) % self.capacity

        # print(episode.states.shape, episode.actions.shape, episode.rewards.shape, episode.dones.shape)

        self.states[idxs] = episode.states
        self.actions[idxs] = episode.actions
        self.rewards[idxs] = episode.rewards
        self.dones[idxs] = episode.dones
        if hasattr(self, "env_states"):
            self.env_states[idxs] = episode.env_states
        is_first = np.zeros(ep_len, dtype=bool)
        is_first[0] = True
        self.is_first[idxs] = is_first

        self.rnn_deter_state[idxs] = episode.deter_states
        self.rnn_stoch_state[idxs] = episode.stoch_states
        self.episode_steps[idxs] = np.arange(ep_len, dtype=np.int32)

        self.position = (self.position + ep_len) % self.capacity
        self.size = min(self.size + ep_len, self.capacity)
        if self.size == self.capacity:
            self.last_item_pos = self.capacity - 1
        else:
            self.last_item_pos = self.position

    def sample(self, batch_size: int, seq_len: int, return_rnn_indices: bool = False) -> dict:
        valid_logical_max = self.size - seq_len

        if valid_logical_max < 0:
            raise ValueError("Buffer contains fewer items than the requested sequence length!")

        logical_starts = np.random.randint(0, valid_logical_max + 1, size=batch_size)

        # 3. Map 'Logical' indices to 'Physical' indices
        #    Physical Index = (Write_Head + Logical_Index) % Capacity
        #    The 'Write_Head' (self.position) always points to the Oldest item (or 0 if not full).

        if self.size < self.capacity:
            # If not full, oldest is simply 0
            start_physical_idx = logical_starts
        else:
            # If full, oldest is at self.position
            start_physical_idx = (self.position + logical_starts) % self.capacity

        # 4. Generate the full sequence of indices
        #    We create the time offsets [0, 1, 2...] and add them to physical starts
        time_indices = np.arange(seq_len)

        data_indices = start_physical_idx[:, None] + time_indices
        action_indices = start_physical_idx[:, None] + time_indices - 1
        # Wrap around the physical buffer
        data_indices %= self.size
        action_indices %= self.size

        batch = self.init_batch(batch_size, seq_len)

        a = self.actions[action_indices]
        b_indices = np.arange(batch_size)[:, None]
        t_indices = np.arange(seq_len)[None, :]
        batch["actions"][b_indices, t_indices, a] = 1.0

        batch["states"][:, time_indices] = self.states[data_indices]
        batch["rewards"][:, time_indices] = self.rewards[action_indices]
        batch["dones"][:, time_indices] = self.dones[action_indices]
        batch["is_first"][:, time_indices] = self.is_first[data_indices]
        batch["rnn_states"]["deter"] = self.rnn_deter_state[data_indices[:, 0]]
        batch["rnn_states"]["stoch"] = self.rnn_stoch_state[data_indices[:, 0]]
        if hasattr(self, "env_states"):
            batch["env_states"] = self.env_states[data_indices[:, 0]]

        head_idx = self.position if self.size == self.capacity else 0

        # Find where we are trying to read the action "before" the head index
        # This occurs when data_indices is exactly head_idx
        boundary_mask = data_indices == head_idx

        # 3. Apply the dummy action to:
        #    a) Actual episode starts (is_first)
        #    b) Buffer boundary crossings (where we lost history)

        # Combine masks: is_first OR it is the oldest element in memory
        should_reset = batch["is_first"] | boundary_mask

        # Actions: always zero at is_first/boundary (no valid previous action)
        reset_indices = np.where(should_reset)
        for b, t in zip(*reset_indices):
            batch["actions"][b, t] = 0.0

        # Rewards: zero only at boundary positions where action_indices wraps to
        # garbage. At is_first-only positions (internal episode boundaries), the
        # arrival-aligned reward is the valid terminal reward of the previous
        # episode — GAE needs it for the terminal state's return.
        boundary_indices = np.where(boundary_mask)
        for b, t in zip(*boundary_indices):
            batch["rewards"][b, t] = 0.0

        # Dones: at is_first the previous episode ended, so done must be True
        # (arrival alignment). At boundary-only positions (lost history) default
        # to False.
        is_first_indices = np.where(batch["is_first"])
        for b, t in zip(*is_first_indices):
            batch["dones"][b, t] = True

        boundary_only = boundary_mask & ~batch["is_first"]
        boundary_only_indices = np.where(boundary_only)
        for b, t in zip(*boundary_only_indices):
            batch["dones"][b, t] = False

        # rnn_indices = data_indices[:, 0]

        if return_rnn_indices:
            return batch, data_indices
        return batch

    def update_rnn_states(self, indices: np.ndarray, deter_states: np.ndarray, stoch_states: np.ndarray):
        self.rnn_deter_state[indices] = deter_states
        self.rnn_stoch_state[indices] = stoch_states

    def init_batch(self, batch_size: int, seq_len: int, dtype=np.float32) -> dict:
        batch = {
            "states": np.zeros((batch_size, seq_len, self.C, self.W, self.H), dtype=np.uint8),
            "actions": np.zeros((batch_size, seq_len, self.action_dim), dtype=np.float32),
            "rewards": np.zeros((batch_size, seq_len), dtype=np.float32),
            "dones": np.zeros((batch_size, seq_len), dtype=bool),
            "is_first": np.zeros((batch_size, seq_len), dtype=bool),
            "rnn_states": {
                "deter": np.zeros((batch_size, self.rnn_deter_dim), dtype=np.float32),
                "stoch": np.zeros((batch_size, self.rnn_stoch_dim), dtype=np.float32),
            },
        }
        if hasattr(self, "env_states"):
            batch["env_states"] = np.empty(batch_size, dtype=object)
        return batch

    def __len__(self):
        return self.size


@gin.configurable
class SingleEnvFastReplayBuffer(FastReplayBuffer):
    """
    Replay buffer supporting individual transitions from a single environment.

    Unlike FastReplayBuffer which requires complete episodes, this buffer allows
    pushing transitions one at a time, enabling online learning without waiting
    for episodes to finish. Automatically tracks episode boundaries via is_first.

    TODO: The ability to push only parts of episodes seems useful, we should implement it also in multri-env buffer.
    Essentially, we can keep |ENV| of theese buffers and push to them based on env id.
    """

    def __init__(self, capacity, C, W, H, action_dim, rnn_deter_dim, rnn_stoch_dim, store_env_states: bool = False):
        super().__init__(capacity, C, W, H, action_dim, rnn_deter_dim, rnn_stoch_dim, store_env_states)

        # Internal state tracking for automatic is_first computation
        self._last_done = True  # First transition will be marked as episode start
        self._current_episode_step = 0

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        deter_state: np.ndarray,
        stoch_state: np.ndarray,
        env_state=None,
        is_first=None,
        prev_action=None,
        prev_reward=None,
        prev_done=None,
    ):
        """Push a single transition to the buffer.

        Args:
            state: Observation array of shape (C, W, H), dtype uint8
            action: Action taken (integer)
            reward: Reward received
            done: Whether episode terminated
            deter_state: RNN deterministic state, shape (rnn_deter_dim,)
            stoch_state: RNN stochastic state, shape (rnn_stoch_dim,)
            env_state: Optional environment state for debugging
        """
        pos = self.position

        # Compute is_first from previous done
        is_first = self._last_done

        # Write transition data
        self.states[pos] = state
        self.actions[pos] = action
        self.rewards[pos] = reward
        self.dones[pos] = done
        self.is_first[pos] = is_first
        self.rnn_deter_state[pos] = deter_state
        self.rnn_stoch_state[pos] = stoch_state
        self.episode_steps[pos] = self._current_episode_step

        if hasattr(self, "env_states"):
            self.env_states[pos] = env_state

        # Update internal tracking
        self._last_done = done
        if done:
            self._current_episode_step = 0
        else:
            self._current_episode_step += 1

        # Advance position with wrap-around
        self.position = (pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.size == self.capacity:
            self.last_item_pos = self.capacity - 1
        else:
            self.last_item_pos = self.position

    def push(self, episode: Episode):
        """Push a complete episode to the buffer (backward compatibility).

        Iterates through the episode and calls push_transition for each step.

        WARNING: Do not use this method together with push_transition in the same buffer,
        as it may lead to inconsistent state.
        """
        ep_len = get_episode_length(episode)

        for i in range(ep_len):
            env_state = None
            if episode.env_states is not None:
                env_state = episode.env_states[i]

            self.push_transition(
                state=episode.states[i],
                action=episode.actions[i],
                reward=episode.rewards[i],
                done=episode.dones[i],
                deter_state=episode.deter_states[i],
                stoch_state=episode.stoch_states[i],
                env_state=env_state,
            )

    def save_obs(self, path, k):
        np.save(path, self.states[: self.size : k])

    def save(self, path):
        metadata = np.array(
            [
                self.position,
                self.size,
                self.capacity,
                self.C,
                self.W,
                self.H,
                self.action_dim,
                self.rnn_deter_dim,
                self.rnn_stoch_dim,
                int(self._last_done),
                self._current_episode_step,
            ]
        )
        np.savez(
            path,
            states=self.states[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            dones=self.dones[: self.size],
            is_first=self.is_first[: self.size],
            rnn_deter_state=self.rnn_deter_state[: self.size],
            rnn_stoch_state=self.rnn_stoch_state[: self.size],
            episode_steps=self.episode_steps[: self.size],
            metadata=metadata,
        )

    @classmethod
    def load(cls, path, reset_rnn_states=False, capacity_override=None):
        data = np.load(path, allow_pickle=False)

        if "metadata" not in data:
            raise ValueError(f"Buffer file {path} has no metadata. Re-save with the updated save() method.")

        meta = data["metadata"].astype(int)
        saved_position, saved_size, saved_capacity = meta[0], meta[1], meta[2]
        C, W, H, action_dim = meta[3], meta[4], meta[5], meta[6]
        rnn_deter_dim, rnn_stoch_dim = meta[7], meta[8]
        last_done, episode_step = bool(meta[9]), meta[10]

        target_capacity = capacity_override if capacity_override is not None else saved_capacity

        buf = cls(target_capacity, C, W, H, action_dim, rnn_deter_dim, rnn_stoch_dim)

        actual_size = len(data["states"])
        if actual_size != saved_size:
            raise ValueError(
                f"Corrupted buffer: metadata says size={saved_size} but states array has {actual_size} rows."
            )

        array_keys = [
            "states",
            "actions",
            "rewards",
            "dones",
            "is_first",
            "rnn_deter_state",
            "rnn_stoch_state",
            "episode_steps",
        ]

        if saved_size <= target_capacity:
            for key in array_keys:
                getattr(buf, key)[:saved_size] = data[key]
            buf.size = saved_size
            # If the saved buffer was full and same capacity, restore position; otherwise data starts at 0
            if saved_size == saved_capacity and saved_size == target_capacity:
                buf.position = saved_position
            else:
                buf.position = saved_size % target_capacity
        else:
            # Truncate: keep the most recent target_capacity items
            # In the saved array, data is in physical order. The newest item is at saved_position - 1 (mod saved_size).
            # Roll so newest is at the end, then take the last target_capacity items.
            shift = -saved_position
            for key in array_keys:
                rolled = np.roll(data[key], shift, axis=0)
                getattr(buf, key)[:target_capacity] = rolled[-target_capacity:]
            buf.size = target_capacity
            buf.position = 0
            # Recompute internal state from the last element in kept data
            last_done = bool(buf.dones[buf.size - 1])
            episode_step = int(buf.episode_steps[buf.size - 1]) + 1 if not last_done else 0

        buf._last_done = last_done
        buf._current_episode_step = episode_step
        buf.last_item_pos = buf.capacity - 1 if buf.size == buf.capacity else buf.position

        if reset_rnn_states:
            buf.rnn_deter_state[: buf.size] = 0
            buf.rnn_stoch_state[: buf.size] = 0

        return buf


@gin.configurable
class SingleEnvOnlineReplayBuffer:
    def __init__(self, capacity, C, W, H, action_dim, rnn_deter_dim, rnn_stoch_dim, store_env_states: bool = False):
        self.queue = deque()
        self.buffer = SingleEnvFastReplayBuffer(
            capacity, C, W, H, action_dim, rnn_deter_dim, rnn_stoch_dim, store_env_states
        )

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        deter_state: np.ndarray,
        stoch_state: np.ndarray,
        env_state=None,
        is_first=None,
        prev_action=None,
        prev_reward=None,
        prev_done=None,
    ):
        self.buffer.push_transition(state, action, reward, done, deter_state, stoch_state, env_state)
        self.queue.append(
            (
                state,
                action,
                reward,
                done,
                deter_state,
                stoch_state,
                env_state,
                is_first,
                prev_action,
                prev_reward,
                prev_done,
            )
        )

    def sample(self, batch_size: int, seq_len: int, return_rnn_indices: bool = False) -> dict:
        b, rnn_indices = self.buffer.sample(batch_size, seq_len, return_rnn_indices=True)

        # Cap the queue to the most recent seq_len transitions
        self.queue = deque(self.queue, maxlen=seq_len)

        m = len(self.queue)
        for k in b:
            # move row 0 by m so online transitions fill the start
            if k == "rnn_states":
                continue
            b[k][0] = np.roll(b[k][0], shift=m, axis=0)

        seq_idx = 0
        while len(self.queue) > 0 and seq_idx < seq_len:
            (
                state,
                action,
                reward,
                done,
                deter_state,
                stoch_state,
                env_state,
                is_first,
                prev_action,
                prev_reward,
                prev_done,
            ) = self.queue.popleft()
            b["states"][0, seq_idx] = state
            b["actions"][0, seq_idx] = 0.0  # zero out rolled data
            if prev_action is not None:
                b["actions"][0, seq_idx, prev_action] = 1.0
                b["rewards"][0, seq_idx] = prev_reward
                b["dones"][0, seq_idx] = prev_done
            b["is_first"][0, seq_idx] = is_first
            if seq_idx == 0:
                b["rnn_states"]["deter"][0] = deter_state
                b["rnn_states"]["stoch"][0] = stoch_state
                rnn_indices[0, :m] = -1
            seq_idx += 1
        if return_rnn_indices:
            return b, rnn_indices
        return b

    def push(self, episode: Episode):
        raise NotImplementedError(
            "Pushing complete episodes is not supported in SingleEnvOnlineReplayBuffer. Use push_transition instead."
        )

    def __len__(self):
        return len(self.buffer)

    def update_rnn_states(self, indices: np.ndarray, deter_states: np.ndarray, stoch_states: np.ndarray):
        mask = indices != -1
        self.buffer.update_rnn_states(indices[mask], deter_states[mask], stoch_states[mask])

    def save_obs(self, path, k):
        self.buffer.save_obs(path, k)

    def save(self, path):
        self.buffer.save(path)

    @classmethod
    def load(cls, path, reset_rnn_states=False, capacity_override=None):
        buf = cls.__new__(cls)
        buf.buffer = SingleEnvFastReplayBuffer.load(path, reset_rnn_states, capacity_override)
        buf.queue = deque()
        return buf
