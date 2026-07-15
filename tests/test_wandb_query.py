from wandb_query import compute_group_averages, get_metric_after_step, group_runs_by_name


class MockRun:
    """Minimal mock of a wandb Run for testing."""

    def __init__(self, name, history_data):
        self.name = name
        self._history = history_data

    def history(self, keys, samples=10000, pandas=False):
        return [{k: row[k] for k in ["_step"] + keys if k in row} for row in self._history]


# ── get_metric_after_step ──


def test_get_metric_after_step_filters_correctly():
    run = MockRun(
        "run_a",
        [
            {"_step": 100, "train/episode_return": 10.0},
            {"_step": 200, "train/episode_return": 20.0},
            {"_step": 300, "train/episode_return": 30.0},
        ],
    )
    vals = get_metric_after_step(run, "train/episode_return", min_step=200)
    assert vals == [20.0, 30.0]


def test_get_metric_after_step_returns_empty_when_no_points():
    run = MockRun(
        "run_a",
        [
            {"_step": 100, "train/episode_return": 10.0},
            {"_step": 200, "train/episode_return": 20.0},
        ],
    )
    vals = get_metric_after_step(run, "train/episode_return", min_step=500)
    assert vals == []


def test_get_metric_after_step_includes_exact_boundary():
    run = MockRun(
        "run_a",
        [
            {"_step": 100, "train/episode_return": 10.0},
            {"_step": 200, "train/episode_return": 20.0},
        ],
    )
    vals = get_metric_after_step(run, "train/episode_return", min_step=200)
    assert vals == [20.0]


def test_get_metric_after_step_works_for_eval():
    run = MockRun(
        "run_a",
        [
            {"_step": 50000, "eval/avg_return": 100.0},
            {"_step": 90000, "eval/avg_return": 200.0},
            {"_step": 99000, "eval/avg_return": 300.0},
        ],
    )
    vals = get_metric_after_step(run, "eval/avg_return", min_step=80000)
    assert vals == [200.0, 300.0]


def test_get_metric_after_step_skips_rows_missing_metric():
    run = MockRun(
        "run_a",
        [
            {"_step": 100, "train/episode_return": 10.0},
            {"_step": 200},  # no metric logged at this step
            {"_step": 300, "train/episode_return": 30.0},
        ],
    )
    vals = get_metric_after_step(run, "train/episode_return", min_step=0)
    assert vals == [10.0, 30.0]


# ── group_runs_by_name ──


def test_group_runs_single_run():
    runs = [MockRun("exp_a", [])]
    groups = group_runs_by_name(runs)
    assert list(groups.keys()) == ["exp_a"]
    assert len(groups["exp_a"]) == 1


def test_group_runs_two_same_name():
    r1 = MockRun("exp_a", [])
    r2 = MockRun("exp_a", [])
    groups = group_runs_by_name([r1, r2])
    assert len(groups) == 1
    assert groups["exp_a"] == [r1, r2]


def test_group_runs_different_names():
    r1 = MockRun("exp_a", [])
    r2 = MockRun("exp_b", [])
    r3 = MockRun("exp_a", [])
    groups = group_runs_by_name([r1, r2, r3])
    assert set(groups.keys()) == {"exp_a", "exp_b"}
    assert len(groups["exp_a"]) == 2
    assert len(groups["exp_b"]) == 1


# ── compute_group_averages ──


def test_compute_group_averages_train_episode_return():
    runs = [
        MockRun(
            "pong",
            [
                {"_step": 90000, "train/episode_return": 5.0},
                {"_step": 98000, "train/episode_return": 10.0},
                {"_step": 99000, "train/episode_return": 20.0},
            ],
        ),
        MockRun(
            "pong",
            [
                {"_step": 90000, "train/episode_return": 3.0},
                {"_step": 98000, "train/episode_return": 14.0},
                {"_step": 99000, "train/episode_return": 16.0},
            ],
        ),
    ]
    result = compute_group_averages(runs, "train/episode_return", min_step=97500)
    assert result == {"pong": 15.0}  # mean([10, 20, 14, 16]) = 15


def test_compute_group_averages_eval_avg_return():
    runs = [
        MockRun(
            "alien",
            [
                {"_step": 50000, "eval/avg_return": 100.0},
                {"_step": 98000, "eval/avg_return": 400.0},
            ],
        ),
        MockRun(
            "alien",
            [
                {"_step": 50000, "eval/avg_return": 200.0},
                {"_step": 98000, "eval/avg_return": 600.0},
            ],
        ),
    ]
    result = compute_group_averages(runs, "eval/avg_return", min_step=97500)
    assert result == {"alien": 500.0}  # mean([400, 600])


def test_compute_group_averages_partial_data():
    """One run in a group has no data after min_step — only the other contributes."""
    runs = [
        MockRun(
            "pong",
            [
                {"_step": 98000, "train/episode_return": 10.0},
            ],
        ),
        MockRun(
            "pong",
            [
                {"_step": 80000, "train/episode_return": 5.0},
            ],
        ),
    ]
    result = compute_group_averages(runs, "train/episode_return", min_step=97500)
    assert result == {"pong": 10.0}


def test_compute_group_averages_excludes_empty_groups():
    """If no run in a group has data after min_step, the group is excluded."""
    runs = [
        MockRun("pong", [{"_step": 1000, "train/episode_return": 5.0}]),
        MockRun("pong", [{"_step": 2000, "train/episode_return": 6.0}]),
    ]
    result = compute_group_averages(runs, "train/episode_return", min_step=97500)
    assert result == {}


def test_compute_group_averages_multiple_groups():
    runs = [
        MockRun("pong", [{"_step": 99000, "train/episode_return": 10.0}]),
        MockRun("pong", [{"_step": 99000, "train/episode_return": 20.0}]),
        MockRun("alien", [{"_step": 99000, "train/episode_return": 100.0}]),
        MockRun("alien", [{"_step": 99000, "train/episode_return": 200.0}]),
    ]
    result = compute_group_averages(runs, "train/episode_return", min_step=97500)
    assert result == {"pong": 15.0, "alien": 150.0}
