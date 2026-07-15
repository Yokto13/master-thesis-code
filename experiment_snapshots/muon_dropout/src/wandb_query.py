from collections import defaultdict
from pathlib import Path

import numpy as np
from bootstrap import bootstrapped_CI

ATARI26_GAMES = [
    "alien",
    "amidar",
    "assault",
    "asterix",
    "bank_heist",
    "battle_zone",
    "boxing",
    "breakout",
    "chopper_command",
    "crazy_climber",
    "demon_attack",
    "freeway",
    "frostbite",
    "gopher",
    "hero",
    "jamesbond",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "ms_pacman",
    "pong",
    "private_eye",
    "qbert",
    "road_runner",
    "seaquest",
    "up_n_down",
]


def _load_scores(path: Path) -> dict[str, float]:
    values = [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    assert len(values) == len(ATARI26_GAMES), f"Expected {len(ATARI26_GAMES)} scores, got {len(values)}"
    return dict(zip(ATARI26_GAMES, values))


def get_metric_after_step(run, metric: str, min_step: int) -> list[float]:
    """Get all values of a metric logged at or after min_step."""
    hist = run.history(keys=[metric], samples=10000, pandas=False)
    return [row[metric] for row in hist if row["_step"] >= min_step and metric in row]


def group_runs_by_name(runs) -> dict[str, list]:
    """Group runs by their name."""
    groups = defaultdict(list)
    for run in runs:
        groups[run.name].append(run)
    return dict(groups)


def compute_group_averages(runs, metric: str, min_step: int) -> dict[str, float]:
    """Compute average metric value (at or after min_step) per group."""
    groups = group_runs_by_name(runs)
    result = {}
    for name, group_runs in groups.items():
        run_means = []
        for run in group_runs:
            vals = get_metric_after_step(run, metric, min_step)
            if vals:
                run_means.append(sum(vals) / len(vals))
        if run_means:
            result[name] = sum(run_means) / len(run_means)
    return result


def format_results(averages: dict[str, float], bare: bool = False) -> str:
    """Format group averages as a string, sorted alphabetically by name."""
    lines = []
    for name in sorted(averages):
        if bare:
            lines.append(f"{averages[name]:.2f}")
        else:
            lines.append(f"{name}: {averages[name]:.2f}")
    return "\n".join(lines)


def _extract_game(run_name: str, suffix: str) -> str | None:
    """Extract game name from a run name by stripping the suffix and matching known games."""
    stem = run_name.removesuffix(f"_{suffix}")
    for game in sorted(ATARI26_GAMES, key=len, reverse=True):
        if stem == game or stem.endswith(f"_{game}"):
            return game
    return None


def compute_all_human_normalized_scores(
    runs, metric: str, min_step: int, suffix: str, human_path: Path, random_path: Path
) -> np.ndarray:
    """Compute per-run HNS grouped by game: shape (num_games, num_seeds)."""
    human = _load_scores(human_path)
    random = _load_scores(random_path)

    per_game: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        game = _extract_game(run.name, suffix)
        if game is None:
            continue
        denom = human[game] - random[game]
        if denom == 0:
            continue
        vals = get_metric_after_step(run, metric, min_step)
        if vals:
            agent_score = sum(vals) / len(vals)
            per_game[game].append((agent_score - random[game]) / denom)

    counts = [len(v) for v in per_game.values()]
    min_seeds = min(counts)
    if min_seeds < max(counts):
        print(f"Warning: unequal seeds per game ({min_seeds}–{max(counts)}), truncating to {min_seeds}")
    return np.array([v[:min_seeds] for v in per_game.values()])


def format_hns_results(hns_values: np.ndarray, bare: bool = False) -> str:
    """Format aggregate HNS with mean and 95% bootstrapped CI."""
    data = np.array(hns_values)
    mean = float(data.mean())
    lower, upper = bootstrapped_CI(data, n=10000, ci=95)

    mean, lower, upper = mean * 100, lower * 100, upper * 100

    if bare:
        return f"{mean:.4f}\n{lower:.4f}\n{upper:.4f}"
    return f"Mean HNS: {mean:.4f}\n95% CI: [{lower:.4f}, {upper:.4f}]"
