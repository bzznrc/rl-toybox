"""Shared game catalog, spec builders, and run preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Callable, Iterable

from core.algorithms.exploration import compute_eps_decay
from core.envs.base import Env
from core.envs.scaffold import ScaffoldEnv
from core.envs.spaces import Space
from core.io.runs import RunPaths, normalize_model_kind, resolve_run_paths

if TYPE_CHECKING:
    from core.algorithms.base import Algorithm


EXPLORATION_AVG_WINDOW_EPISODES = 100
MIN_EPISODES_FOR_STATS = 100

OFF_POLICY_TRAIN_DEFAULTS: dict[str, Any] = {
    "train_after_steps": 0,
    "update_every_steps": 1,
    "updates_per_step": 1,
    "reward_window": int(EXPLORATION_AVG_WINDOW_EPISODES),
    "min_episodes_for_stats": int(MIN_EPISODES_FOR_STATS),
}

ACTIVE_GAME_ORDER: tuple[str, ...] = (
    "snake",
    "bang",
    "tower",
    "vroom",
    "frogger",
    "card",
    "osero",
    "kick",
)


@dataclass(frozen=True)
class GameSpec:
    game_id: str
    display_name: str
    default_algo: str
    make_env: Callable[..., Env]
    obs_dim: int
    action_space: Space
    run_name: str
    family: str
    role: str
    summary: str
    primary_algo_label: str
    status: str = "active"
    implementation_stage: str = "implemented"
    train_config: dict[str, object] = field(default_factory=dict)
    algo_config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedRun:
    spec: GameSpec
    algo_id: str
    run_paths: RunPaths
    algorithm: "Algorithm"


def build_env_factory(env_type: type[Env]) -> Callable[..., Env]:
    def make_env(mode: str, render: bool, level: int | None = None) -> Env:
        return env_type(mode=mode, render=render, level=level)

    return make_env


def build_scaffold_env_factory(
    *,
    game_id: str,
    obs_dim: int,
    note: str,
) -> Callable[..., Env]:
    def make_env(mode: str, render: bool, level: int | None = None) -> Env:
        return ScaffoldEnv(
            game_id=game_id,
            obs_dim=int(obs_dim),
            mode=mode,
            render=render,
            level=level,
            note=note,
        )

    return make_env


def build_hidden_run_name(hidden_sizes: Iterable[int]) -> str:
    sizes = [int(size) for size in hidden_sizes]
    if not sizes:
        raise ValueError("Run-name hidden sizes must not be empty.")
    return "_".join(str(size) for size in sizes)


def build_actor_critic_run_name(
    actor_hidden_sizes: Iterable[int],
    critic_hidden_sizes: Iterable[int],
) -> str:
    return f"a{build_hidden_run_name(actor_hidden_sizes)}_c{build_hidden_run_name(critic_hidden_sizes)}"


def build_recurrent_run_name(
    encoder_hidden_sizes: Iterable[int],
    *,
    recurrent_type: str,
    recurrent_hidden_size: int,
    actor_head_hidden_sizes: Iterable[int],
    critic_head_hidden_sizes: Iterable[int],
) -> str:
    recurrent_key = str(recurrent_type).strip().lower()
    if recurrent_key not in {"lstm", "gru"}:
        raise ValueError("Recurrent run name requires recurrent_type 'lstm' or 'gru'.")
    return (
        f"e{build_hidden_run_name(encoder_hidden_sizes)}_"
        f"{recurrent_key}{int(recurrent_hidden_size)}_"
        f"a{build_hidden_run_name(actor_head_hidden_sizes)}_"
        f"c{build_hidden_run_name(critic_head_hidden_sizes)}"
    )


def build_exploration_config(
    eps_start: float,
    eps_min: float,
    eps_decay_steps: int,
    *,
    patience_episodes: int,
    min_improvement: float,
    eps_bump_cap: float,
    bump_cooldown_steps: int,
    avg_window_episodes: int = EXPLORATION_AVG_WINDOW_EPISODES,
) -> dict[str, Any]:
    return {
        "eps_start": float(eps_start),
        "eps_min": float(eps_min),
        "eps_decay": compute_eps_decay(
            eps_start=float(eps_start),
            eps_min=float(eps_min),
            eps_decay_steps=int(eps_decay_steps),
        ),
        "avg_window_episodes": int(avg_window_episodes),
        "patience_episodes": int(patience_episodes),
        "min_improvement": float(min_improvement),
        "eps_bump_cap": float(eps_bump_cap),
        "bump_cooldown_steps": int(bump_cooldown_steps),
    }


def build_off_policy_train_config(
    *,
    max_steps: int,
    checkpoint_every_steps: int,
    reward_window: int,
    min_episodes_for_stats: int | None = None,
    train_after_steps: int | None = None,
    update_every_steps: int | None = None,
    updates_per_step: int | None = None,
) -> dict[str, object]:
    config: dict[str, object] = {
        **OFF_POLICY_TRAIN_DEFAULTS,
        "max_steps": int(max_steps),
        "checkpoint_every_steps": int(checkpoint_every_steps),
        "reward_window": int(reward_window),
        "min_episodes_for_stats": int(
            reward_window if min_episodes_for_stats is None else min_episodes_for_stats
        ),
    }
    if train_after_steps is not None:
        config["train_after_steps"] = int(train_after_steps)
    if update_every_steps is not None:
        config["update_every_steps"] = int(update_every_steps)
    if updates_per_step is not None:
        config["updates_per_step"] = int(updates_per_step)
    return config


def build_on_policy_train_config(
    *,
    max_iterations: int,
    rollout_steps: int,
    checkpoint_every_iterations: int,
    reward_window: int,
    min_episodes_for_stats: int,
) -> dict[str, object]:
    return {
        "max_iterations": int(max_iterations),
        "rollout_steps": int(rollout_steps),
        "checkpoint_every_iterations": int(checkpoint_every_iterations),
        "reward_window": int(reward_window),
        "min_episodes_for_stats": int(min_episodes_for_stats),
    }


_GAME_SPECS: dict[str, GameSpec] | None = None


def _build_game_specs() -> dict[str, GameSpec]:
    # Import game specs lazily so the shared builders above can be imported
    # from each game spec without triggering a circular import.
    from games.bang.spec import SPEC as bang_spec
    from games.card.spec import SPEC as card_spec
    from games.frogger.spec import SPEC as frogger_spec
    from games.kick.spec import SPEC as kick_spec
    from games.osero.spec import SPEC as osero_spec
    from games.snake.spec import SPEC as snake_spec
    from games.tower.spec import SPEC as tower_spec
    from games.vroom.spec import SPEC as vroom_spec

    specs = (
        snake_spec,
        bang_spec,
        tower_spec,
        vroom_spec,
        frogger_spec,
        card_spec,
        osero_spec,
        kick_spec,
    )
    return {spec.game_id: spec for spec in specs}


def all_game_specs() -> dict[str, GameSpec]:
    global _GAME_SPECS
    if _GAME_SPECS is None:
        _GAME_SPECS = _build_game_specs()
    return dict(_GAME_SPECS)


def get_game_spec(game_id: str) -> GameSpec:
    game_key = str(game_id).strip().lower()
    specs = all_game_specs()
    if game_key not in specs:
        valid = ", ".join(sorted(specs.keys()))
        raise KeyError(f"Unknown game '{game_id}'. Valid options: {valid}")
    return specs[game_key]


def prepare_run(game_id: str, algo_override: str | None = None) -> PreparedRun:
    from core.algorithms.factory import build_algorithm

    spec = get_game_spec(game_id)
    algo_id = str(algo_override or spec.default_algo).strip().lower()
    run_paths = resolve_run_paths(spec.game_id, algo_id, spec.run_name, create=True)
    algorithm = build_algorithm(
        algo_id=algo_id,
        obs_dim=spec.obs_dim,
        action_space=spec.action_space,
        algo_config=spec.algo_config,
    )
    return PreparedRun(spec=spec, algo_id=algo_id, run_paths=run_paths, algorithm=algorithm)


def resolve_current_level(env: object, *, default: int = 1) -> int:
    level_value = getattr(env, "_current_level", None)
    if level_value is None:
        game = getattr(env, "game", None)
        level_value = getattr(game, "level", None)
    try:
        return max(1, int(level_value))
    except (TypeError, ValueError):
        return max(1, int(default))


def normalize_resume_mode(mode: str) -> str:
    mode_key = str(mode).strip().lower()
    if mode_key == "new":
        return "none"
    if mode_key == "checkpoint":
        return "check"
    return mode_key


def resume_level_bounds(env: object) -> tuple[int, int]:
    curriculum = getattr(env, "_curriculum", None)
    if curriculum is None:
        return 1, 3
    curriculum_config = getattr(curriculum, "config", None)
    min_level = max(1, int(getattr(curriculum_config, "min_level", 1)))
    max_level = max(min_level, int(getattr(curriculum_config, "max_level", 3)))
    return int(min_level), int(max_level)


def resolve_best_resume_level(
    env: object,
    *,
    explicit_level: int | None = None,
    allow_prompt: bool = True,
) -> int:
    min_level, max_level = resume_level_bounds(env)
    if explicit_level is not None:
        return max(int(min_level), min(int(max_level), int(explicit_level)))
    if not bool(allow_prompt) or not bool(sys.stdin.isatty()):
        raise ValueError(
            "Resume mode 'best' requires --resume-level in non-interactive mode. "
            f"Expected {int(min_level)}..{int(max_level)}."
        )

    prompt = f"Resume from BEST model level ({int(min_level)}-{int(max_level)}): "
    while True:
        try:
            raw_value = input(prompt).strip()
        except EOFError as exc:
            raise ValueError(
                "Resume mode 'best' requires --resume-level when interactive input is unavailable."
            ) from exc
        if raw_value == "":
            print("Level is required.")
            continue
        try:
            parsed_level = int(raw_value)
        except ValueError:
            print("Invalid input. Enter a numeric level.")
            continue
        if int(min_level) <= int(parsed_level) <= int(max_level):
            return int(parsed_level)
        print(f"Invalid level. Choose a value between {int(min_level)} and {int(max_level)}.")


def apply_training_start_level(env: object, level: int) -> int:
    level_value = max(1, int(level))
    curriculum = getattr(env, "_curriculum", None)
    if curriculum is not None:
        curriculum_config = getattr(curriculum, "config", None)
        min_level = int(getattr(curriculum_config, "min_level", 1))
        max_level = int(getattr(curriculum_config, "max_level", max(min_level, level_value)))
        level_value = max(min_level, min(level_value, max_level))
        set_level = getattr(curriculum, "set_level", None)
        if callable(set_level):
            set_level(int(level_value), reset_progress=True)
        else:
            reset_curriculum = getattr(curriculum, "reset", None)
            if callable(reset_curriculum):
                reset_curriculum()
            if hasattr(curriculum, "_level"):
                setattr(curriculum, "_level", int(level_value))
            if hasattr(curriculum, "_consecutive_passes"):
                setattr(curriculum, "_consecutive_passes", 0)

    apply_level = getattr(env, "_apply_level_change", None)
    if not callable(apply_level):
        apply_level = getattr(env, "_apply_level_settings", None)
    if callable(apply_level):
        apply_level(int(level_value))
    else:
        if hasattr(env, "_current_level"):
            setattr(env, "_current_level", int(level_value))
        game = getattr(env, "game", None)
        if game is not None and hasattr(game, "level"):
            setattr(game, "level", int(level_value))
            configure_level = getattr(game, "configure_level", None)
            if callable(configure_level):
                configure_level()
    return int(level_value)


def resolve_resume_path(mode: str, run_paths: RunPaths, level: int) -> Path | None:
    mode_key = str(mode).strip().lower()
    level_value = max(1, int(level))
    best_path = run_paths.model_path(level_value, "best")
    check_path = run_paths.model_path(level_value, "check")
    if mode_key == "none":
        return None
    if mode_key in {"check", "checkpoint"}:
        return check_path if check_path.exists() else None
    if mode_key == "best":
        return best_path if best_path.exists() else None
    if mode_key == "auto":
        if best_path.exists():
            return best_path
        if check_path.exists():
            return check_path
        return None
    raise ValueError(f"Unsupported resume mode: {mode}")


def resolve_play_model_path(run_paths: RunPaths, model_choice: str, level: int) -> Path:
    level_value = max(1, int(level))
    model_kind = normalize_model_kind(str(model_choice))
    if model_kind == "check":
        path = run_paths.model_path(level_value, "check")
    else:
        path = run_paths.model_path(level_value, "best")
        fallback_check = run_paths.model_path(level_value, "check")
        if not path.exists() and fallback_check.exists():
            path = fallback_check
    if not path.exists():
        raise FileNotFoundError(f"No model found at '{path}'.")
    return path


__all__ = [
    "EXPLORATION_AVG_WINDOW_EPISODES",
    "MIN_EPISODES_FOR_STATS",
    "OFF_POLICY_TRAIN_DEFAULTS",
    "ACTIVE_GAME_ORDER",
    "GameSpec",
    "PreparedRun",
    "build_env_factory",
    "build_scaffold_env_factory",
    "build_hidden_run_name",
    "build_actor_critic_run_name",
    "build_recurrent_run_name",
    "build_exploration_config",
    "build_off_policy_train_config",
    "build_on_policy_train_config",
    "all_game_specs",
    "get_game_spec",
    "prepare_run",
    "resolve_current_level",
    "normalize_resume_mode",
    "resume_level_bounds",
    "resolve_best_resume_level",
    "apply_training_start_level",
    "resolve_resume_path",
    "resolve_play_model_path",
]
