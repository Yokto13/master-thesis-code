import numpy as np
from multi_env_replay_buffer import MultiEnvReplayBuffer


def make_buffer(capacity=8, num_envs=1, action_dim=4):
    return MultiEnvReplayBuffer(
        capacity=capacity, num_envs=num_envs, C=1, W=1, H=1, action_dim=action_dim, rnn_deter_dim=2, rnn_stoch_dim=3
    )


def push_all(buf, state_values, action_values, reward_values, done_values, env_offset=0):
    """Push scalar-per-step transitions, broadcast across envs with per-env state offsets."""
    offsets = env_offset * np.arange(buf.num_envs)
    for i, (s, a, r, d) in enumerate(zip(state_values, action_values, reward_values, done_values)):
        buf.push(
            (s + offsets).astype(np.uint8).reshape(buf.num_envs, 1, 1, 1),
            np.full(buf.num_envs, a, dtype=np.uint8),
            np.full(buf.num_envs, r, dtype=np.float32),
            np.full(buf.num_envs, d, dtype=bool),
            np.full((buf.num_envs, 2), float(i), dtype=np.float16),
            np.full((buf.num_envs, 3), float(i), dtype=np.float16),
        )


def test_arrival_alignment_and_dummy_action_at_episode_start() -> None:
    buf = make_buffer()
    push_all(buf, [10, 11, 12], [1, 2, 3], [0.1, 0.2, 0.3], [False, False, True])

    # All 3 transitions are fresh, so the single online row is exactly [0, 1, 2].
    batch, indices = buf.sample(batch_size=1, seq_len=3, return_rnn_indices=True)

    np.testing.assert_array_equal(indices, np.array([[0, 1, 2]]))
    np.testing.assert_array_equal(batch["states"][0, :, 0, 0, 0], np.array([10, 11, 12], dtype=np.uint8))
    np.testing.assert_array_equal(batch["is_first"][0], np.array([True, False, False], dtype=bool))

    expected_actions = np.zeros((3, 4), dtype=np.float32)
    expected_actions[1, 1] = 1.0
    expected_actions[2, 2] = 1.0
    np.testing.assert_array_equal(batch["actions"][0], expected_actions)

    np.testing.assert_allclose(batch["rewards"][0], np.array([0.0, 0.1, 0.2], dtype=np.float32))
    # At is_first positions, done=True in arrival alignment (previous episode ended)
    np.testing.assert_array_equal(batch["dones"][0], np.array([True, False, False], dtype=bool))
    np.testing.assert_array_equal(batch["rnn_states"]["deter"][0], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(batch["rnn_states"]["stoch"][0], np.zeros(3, dtype=np.float32))


def test_episode_boundary_preserves_terminal_reward_and_done() -> None:
    """When a sequence spans two episodes, the is_first position must have
    done=True (episode boundary) and the terminal reward of the previous
    episode must be preserved for correct GAE computation."""
    buf = make_buffer(capacity=16)
    # Episode A (terminal reward 5.0) then episode B.
    push_all(buf, [10, 11, 12], [1, 2, 3], [0.0, 0.0, 5.0], [False, False, True])
    push_all(buf, [20, 21], [0, 1], [1.0, 2.0], [False, True])
    buf._steps_since_sample = 0  # disable the online row so all rows are random

    # Sample until we get the sequence starting at slot 1: [11, 12, 20, 21].
    np.random.seed(0)
    for _ in range(200):
        batch, indices = buf.sample(batch_size=1, seq_len=4, return_rnn_indices=True)
        if indices[0, 0] == 1:
            break
    else:
        raise AssertionError("Could not sample sequence starting at index 1")

    np.testing.assert_array_equal(batch["states"][0, :, 0, 0, 0], [11, 12, 20, 21])
    np.testing.assert_array_equal(batch["is_first"][0], [False, False, True, False])

    expected_actions = np.zeros((4, 4), dtype=np.float32)
    expected_actions[0, 1] = 1.0  # action 1 led to state 11
    expected_actions[1, 2] = 1.0  # action 2 led to state 12
    # position 2: is_first → dummy action (zeros)
    expected_actions[3, 0] = 1.0  # action 0 led to state 21
    np.testing.assert_array_equal(batch["actions"][0], expected_actions)

    # Position 2 (is_first, not head): arrival reward = terminal reward of episode A
    np.testing.assert_allclose(batch["rewards"][0], [0.0, 0.0, 5.0, 1.0])
    np.testing.assert_array_equal(batch["dones"][0], [False, False, True, False])


def test_online_rows_contain_freshest_transitions() -> None:
    buf = make_buffer(capacity=16)
    push_all(buf, [10, 11, 12, 13, 14], [0, 0, 0, 0, 0], [0.0] * 5, [False] * 5)
    buf.sample(batch_size=1, seq_len=3)  # consume the fresh counter
    assert buf._steps_since_sample == 0

    push_all(buf, [15, 16], [1, 2], [0.5, 0.6], [False, False])
    batch = buf.sample(batch_size=2, seq_len=4)

    # Row 0 starts with the 2 transitions pushed since the last sample.
    np.testing.assert_array_equal(batch["states"][0, :2, 0, 0, 0], [15, 16])
    # Arrival-aligned action at the second fresh slot is the first fresh action.
    assert batch["actions"][0, 1, 1] == 1.0
    np.testing.assert_allclose(batch["rewards"][0, 1], 0.5)
    assert buf._steps_since_sample == 0


def test_wraparound_head_is_treated_as_unknown_history() -> None:
    buf = make_buffer(capacity=4)
    push_all(buf, [10, 11, 12, 13, 14, 15], [1, 1, 1, 1, 1, 1], [1.0] * 6, [False] * 6)
    assert buf.size == 4 and buf.position == 2
    buf._steps_since_sample = 0

    # seq_len == size forces the single valid sequence: oldest-to-newest [12..15].
    batch = buf.sample(batch_size=1, seq_len=4)
    np.testing.assert_array_equal(batch["states"][0, :, 0, 0, 0], [12, 13, 14, 15])
    # Head slot: previous transition was overwritten → dummy action, zero reward, not done.
    np.testing.assert_array_equal(batch["actions"][0, 0], np.zeros(4, dtype=np.float32))
    assert batch["rewards"][0, 0] == 0.0
    assert not batch["dones"][0, 0]
    # Non-head slots keep real arrival-aligned data.
    np.testing.assert_allclose(batch["rewards"][0, 1:], 1.0)


def test_sequences_never_cross_environments() -> None:
    buf = make_buffer(capacity=32, num_envs=2)
    # env 0 holds values 0..9, env 1 holds 100..109
    push_all(buf, list(range(10)), [0] * 10, [0.0] * 10, [False] * 10, env_offset=100)
    buf._steps_since_sample = 0

    np.random.seed(0)
    for _ in range(20):
        batch = buf.sample(batch_size=4, seq_len=5)
        states = batch["states"][:, :, 0, 0, 0]
        per_row_env = states // 100  # 0 for env 0, 1 for env 1
        assert (per_row_env == per_row_env[:, :1]).all(), "sequence mixed data from two envs"


def test_save_load_round_trip(tmp_path) -> None:
    buf = make_buffer(capacity=16, num_envs=2)
    push_all(buf, [10, 11, 12], [1, 2, 3], [0.1, 0.2, 0.3], [False, False, True], env_offset=50)

    path = str(tmp_path / "buf.npz")
    buf.save(path)
    loaded = MultiEnvReplayBuffer.load(path)

    assert len(loaded) == len(buf) == 6
    assert loaded.capacity == buf.capacity
    assert loaded.position == buf.position
    np.testing.assert_array_equal(loaded._last_done, buf._last_done)
    for key in MultiEnvReplayBuffer._ARRAY_KEYS:
        np.testing.assert_array_equal(getattr(loaded, key)[:, : loaded.size], getattr(buf, key)[:, : buf.size])

    batch = loaded.sample(batch_size=2, seq_len=3)
    assert batch["states"].shape == (2, 3, 1, 1, 1)
    assert batch["actions"].shape == (2, 3, 4)


def test_save_load_full_ring_restores_logical_order(tmp_path) -> None:
    buf = make_buffer(capacity=4)
    push_all(buf, [10, 11, 12, 13, 14, 15], [0] * 6, [0.0] * 6, [False] * 6)
    assert buf.position == 2  # ring has wrapped

    path = str(tmp_path / "buf.npz")
    buf.save(path)
    loaded = MultiEnvReplayBuffer.load(path)

    # Saved in oldest-first order: slot 0 is the oldest kept transition.
    np.testing.assert_array_equal(loaded.states[0, :, 0, 0, 0], [12, 13, 14, 15])
    assert loaded.size == 4 and loaded.position == 0  # full ring, oldest at index 0


def test_load_with_reset_rnn_states(tmp_path) -> None:
    buf = make_buffer(capacity=16)
    push_all(buf, [10, 11, 12], [1, 2, 3], [0.1, 0.2, 0.3], [False, False, True])

    path = str(tmp_path / "buf.npz")
    buf.save(path)
    loaded = MultiEnvReplayBuffer.load(path, reset_rnn_states=True)

    np.testing.assert_array_equal(loaded.rnn_deter_state[:, : loaded.size], 0)
    np.testing.assert_array_equal(loaded.rnn_stoch_state[:, : loaded.size], 0)
    np.testing.assert_array_equal(loaded.states[:, : loaded.size], buf.states[:, : buf.size])


def test_load_with_smaller_capacity_keeps_newest(tmp_path) -> None:
    buf = make_buffer(capacity=6)
    push_all(buf, [10, 11, 12, 20, 21, 22], [1, 2, 3, 0, 1, 2], [0.0] * 6, [False, False, True, False, False, True])
    assert buf.size == 6

    path = str(tmp_path / "buf.npz")
    buf.save(path)
    loaded = MultiEnvReplayBuffer.load(path, capacity_override=4)

    assert loaded.size == 4 and loaded.capacity == 4
    np.testing.assert_array_equal(loaded.states[0, :4, 0, 0, 0], [12, 20, 21, 22])
    np.testing.assert_array_equal(loaded._last_done, [True])


def test_load_legacy_single_env_format(tmp_path) -> None:
    """Buffers saved by the old SingleEnvFastReplayBuffer.save() must still load."""
    size, capacity = 3, 16
    path = str(tmp_path / "legacy.npz")
    np.savez(
        path,
        states=np.array([10, 11, 12], dtype=np.uint8).reshape(size, 1, 1, 1),
        actions=np.array([1, 2, 3], dtype=np.uint8),
        rewards=np.array([0.1, 0.2, 0.3]),
        dones=np.array([0.0, 0.0, 1.0]),
        is_first=np.array([1.0, 0.0, 0.0]),
        rnn_deter_state=np.zeros((size, 2), dtype=np.float16),
        rnn_stoch_state=np.zeros((size, 3), dtype=np.float16),
        episode_steps=np.arange(size, dtype=np.int32),
        # [position, size, capacity, C, W, H, action_dim, deter, stoch, last_done, episode_step]
        metadata=np.array([size, size, capacity, 1, 1, 1, 4, 2, 3, 1, 0]),
    )

    loaded = MultiEnvReplayBuffer.load(path)

    assert loaded.num_envs == 1
    assert len(loaded) == size and loaded.capacity == capacity
    np.testing.assert_array_equal(loaded.states[0, :size, 0, 0, 0], [10, 11, 12])
    np.testing.assert_array_equal(loaded.dones[0, :size], [False, False, True])
    np.testing.assert_array_equal(loaded.is_first[0, :size], [True, False, False])
    np.testing.assert_array_equal(loaded._last_done, [True])

    batch = loaded.sample(batch_size=1, seq_len=3)
    assert batch["states"].shape == (1, 3, 1, 1, 1)
    assert batch["actions"].shape == (1, 3, 4)


def test_terminal_state_push_and_alignment() -> None:
    buf = make_buffer(capacity=8)

    # Episode 1: length 2 (obs 10, 11), terminated at step 1 (obs 11)
    # Step 0:
    buf.push(
        np.array([[10]], dtype=np.uint8).reshape(1, 1, 1, 1),
        np.array([1], dtype=np.uint8),
        np.array([0.1], dtype=np.float32),
        np.array([False], dtype=bool),
        np.zeros((1, 2), dtype=np.float16),
        np.zeros((1, 3), dtype=np.float16),
        is_first=np.array([True], dtype=bool),
    )
    # Step 1 (leads to terminal state 12):
    buf.push(
        np.array([[11]], dtype=np.uint8).reshape(1, 1, 1, 1),
        np.array([2], dtype=np.uint8),
        np.array([0.5], dtype=np.float32),
        np.array([True], dtype=bool),  # terminated
        np.zeros((1, 2), dtype=np.float16),
        np.zeros((1, 3), dtype=np.float16),
        is_first=np.array([False], dtype=bool),
    )
    # Push terminal state 12:
    buf.push(
        np.array([[12]], dtype=np.uint8).reshape(1, 1, 1, 1),
        np.array([0], dtype=np.uint8),  # dummy action
        np.array([0.0], dtype=np.float32),  # dummy reward
        np.array([False], dtype=bool),
        np.zeros((1, 2), dtype=np.float16),
        np.zeros((1, 3), dtype=np.float16),
        is_first=np.array([False], dtype=bool),
    )
    # Episode 2: spawn step (obs 20)
    buf.push(
        np.array([[20]], dtype=np.uint8).reshape(1, 1, 1, 1),
        np.array([3], dtype=np.uint8),
        np.array([0.2], dtype=np.float32),
        np.array([False], dtype=bool),
        np.zeros((1, 2), dtype=np.float16),
        np.zeros((1, 3), dtype=np.float16),
        is_first=np.array([True], dtype=bool),
    )

    # Disable online row
    buf._steps_since_sample = 0

    # Sample a sequence of length 4: [10, 11, 12, 20]
    # Starts at index 0.
    batch = buf.sample(batch_size=1, seq_len=4)

    np.testing.assert_array_equal(batch["states"][0, :, 0, 0, 0], [10, 11, 12, 20])
    np.testing.assert_array_equal(batch["is_first"][0], [True, False, False, True])

    # Arrival-aligned actions:
    # Step 0 (10): is_first -> action masked to 0
    # Step 1 (11): action 1 (from step 0)
    # Step 2 (12): action 2 (from step 1)
    # Step 3 (20): is_first -> action masked to 0
    expected_actions = np.zeros((4, 4), dtype=np.float32)
    expected_actions[1, 1] = 1.0
    expected_actions[2, 2] = 1.0
    np.testing.assert_array_equal(batch["actions"][0], expected_actions)

    # Arrival-aligned rewards:
    # Step 0 (10): head/is_first -> reward masked to 0
    # Step 1 (11): reward 0.1 (from step 0)
    # Step 2 (12): reward 0.5 (from step 1)
    # Step 3 (20): reward 0.0 (dummy reward from step 2)
    np.testing.assert_allclose(batch["rewards"][0], [0.0, 0.1, 0.5, 0.0])

    # Arrival-aligned dones:
    # Step 0 (10): is_first -> done=True
    # Step 1 (11): done=False (from step 0)
    # Step 2 (12): done=True (from step 1 - terminal state correctly gets done=True!)
    # Step 3 (20): is_first -> done=True
    np.testing.assert_array_equal(batch["dones"][0], [True, False, True, True])
