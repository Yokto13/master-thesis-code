import numpy as np
import torch
from dreamer import continuation_target
from multi_env_replay_buffer import MultiEnvReplayBuffer

GAMMA = 0.997


def test_continuation_target_slot_types() -> None:
    # slots: [mid-episode, terminal, spawn (buffer-masked done), first-with-lost-history]
    dones = torch.tensor([[False, True, True, False]])
    is_first = torch.tensor([[False, False, True, True]])

    target = continuation_target(dones, is_first, GAMMA)

    expected = torch.tensor([[GAMMA, 0.0, GAMMA, GAMMA]])
    torch.testing.assert_close(target, expected)


def test_continuation_target_through_buffer() -> None:
    """End-to-end: terminal-push convention -> sampled batch -> cont targets."""
    buf = MultiEnvReplayBuffer(capacity=8, num_envs=1, C=1, W=1, H=1, action_dim=4, rnn_deter_dim=2, rnn_stoch_dim=3)

    def push(obs, action, reward, done, first):
        buf.push(
            np.full((1, 1, 1, 1), obs, dtype=np.uint8),
            np.array([action], dtype=np.uint8),
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=bool),
            np.zeros((1, 2), dtype=np.float16),
            np.zeros((1, 3), dtype=np.float16),
            is_first=np.array([first], dtype=bool),
        )

    push(10, 1, 0.1, False, True)  # episode 1 spawn
    push(11, 2, 0.5, True, False)  # fatal action taken here
    push(12, 0, 0.0, False, False)  # terminal state
    push(20, 3, 0.2, False, True)  # episode 2 spawn

    buf._steps_since_sample = 0  # disable online row
    batch = buf.sample(batch_size=1, seq_len=4)

    dones = torch.from_numpy(batch["dones"])
    is_first = torch.from_numpy(batch["is_first"])

    # Replay GAE still sees episode boundaries at both spawns and the terminal slot.
    np.testing.assert_array_equal(batch["dones"][0], [True, False, True, True])

    # The cont head sees a terminal only at the actual terminal state.
    target = continuation_target(dones, is_first, GAMMA)
    expected = torch.tensor([[GAMMA, GAMMA, 0.0, GAMMA]])
    torch.testing.assert_close(target, expected)
