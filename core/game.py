"""Shared game catalog, spec builders, and run preparation helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import TYPE_CHECKING, Any, Callable, Iterable

from core.algorithms.exploration import compute_eps_decay
from core.curriculum import DEFAULT_CURRICULUM_MAX_LEVEL, DEFAULT_CURRICULUM_MIN_LEVEL
from core.envs.base import Env
from core.envs.spaces import Box, Discrete, Space
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
DEFAULT_ALGO_TOKENS = frozenset({"", "auto", "default", "game"})

ACTIVE_GAME_ORDER: tuple[str, ...] = (
    "snake",
    "bang",
    "jump",
    "vroom",
    "osero",
    "kick",
)
DEFAULT_TRAINING_LAUNCH_LEVEL = int(DEFAULT_CURRICULUM_MIN_LEVEL)
DEFAULT_PLAY_LAUNCH_LEVEL = int(DEFAULT_CURRICULUM_MAX_LEVEL)
DEFAULT_OSERO_LAUNCH_LEVEL = 2
GENERIC_LAUNCH_LEVEL_TO_OSERO_BOARD_SIZE = {
    1: 4,
    2: 6,
    3: 8,
}


@dataclass(frozen=True)
class GameSpec:
    game_id: str
    default_algo: str
    make_env: Callable[..., Env]
    obs_dim: int
    action_space: Space
    capabilities: "GameCapabilities"
    device: str = "cpu"
    env_metadata: dict[str, object] = field(default_factory=dict)
    default_model_config: dict[str, object] = field(default_factory=dict)
    algo_config_overrides: dict[str, dict[str, object]] = field(default_factory=dict)
    default_train_config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GameCapabilities:
    masked_actions: bool = False
    multi_agent: bool = False
    self_play: bool = False
    recurrent_friendly: bool = False
    centralized_critic_required: bool = False


@dataclass(frozen=True)
class AlgoCapabilities:
    supported_action_spaces: tuple[str, ...]
    supports_masked_actions: bool = False
    supports_multi_agent: bool = False
    supports_self_play: bool = False
    supports_centralized_critic: bool = False
    requires_multi_agent: bool = False
    requires_self_play: bool = False
    requires_recurrent_friendly: bool = False


@dataclass(frozen=True)
class AlgoSpec:
    algo_id: str
    runner_kind: str
    capabilities: AlgoCapabilities
    defaults: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedRun:
    run_paths: RunPaths
    config: dict[str, object]


def build_env_factory(env_type: type[Env]) -> Callable[..., Env]:
    def make_env(mode: str, render: bool, level: int | None = None) -> Env:
        return env_type(mode=mode, render=render, level=level)

    return make_env


def _derive_game_id_from_config(config_module: object) -> str:
    explicit = getattr(config_module, "GAME_ID", None)
    if explicit is not None:
        return str(explicit).strip().lower()
    module_name = str(getattr(config_module, "__name__", "")).strip()
    parts = [part for part in module_name.split(".") if part]
    if len(parts) >= 2 and str(parts[-1]).strip().lower() == "config":
        return str(parts[-2]).strip().lower()
    raise ValueError(f"Unable to derive game_id from config module '{module_name}'.")


def _build_action_space_from_config(config_module: object) -> Space:
    bounds = getattr(config_module, "ACTION_SPACE_BOUNDS", None)
    if bounds is not None:
        low = dict(bounds).get("low")
        high = dict(bounds).get("high")
        return Box(
            shape=(int(getattr(config_module, "ACT_DIM")),),
            low=low,
            high=high,
        )
    return Discrete(int(getattr(config_module, "ACT_DIM")))


def build_game_spec_from_config(*, config_module: object, env_type: type[Env]) -> GameSpec:
    algo_config_overrides = {
        str(algo_id).strip().lower(): dict(config_values)
        for algo_id, config_values in dict(getattr(config_module, "ALGO_CONFIG_OVERRIDES", {})).items()
    }
    return GameSpec(
        game_id=_derive_game_id_from_config(config_module),
        default_algo=str(getattr(config_module, "DEFAULT_ALGO")).strip().lower(),
        make_env=build_env_factory(env_type),
        obs_dim=int(getattr(config_module, "OBS_DIM")),
        action_space=_build_action_space_from_config(config_module),
        capabilities=GameCapabilities(**dict(getattr(config_module, "GAME_CAPABILITIES", {}))),
        device="cuda" if bool(getattr(config_module, "USE_GPU", False)) else "cpu",
        env_metadata=dict(getattr(config_module, "ENV_METADATA", {})),
        default_model_config=dict(getattr(config_module, "DEFAULT_MODEL_CONFIG", {})),
        algo_config_overrides=algo_config_overrides,
        default_train_config=dict(getattr(config_module, "DEFAULT_TRAIN_CONFIG", {})),
    )


def _generic_launch_level_bounds(game_id: str) -> tuple[int, int]:
    game_key = str(game_id).strip().lower()
    if game_key == "osero":
        return 1, int(len(GENERIC_LAUNCH_LEVEL_TO_OSERO_BOARD_SIZE))
    return int(DEFAULT_CURRICULUM_MIN_LEVEL), int(DEFAULT_CURRICULUM_MAX_LEVEL)


def _default_generic_launch_level(game_id: str, *, mode: str | None = None) -> int:
    game_key = str(game_id).strip().lower()
    if game_key == "osero":
        return int(DEFAULT_OSERO_LAUNCH_LEVEL)
    mode_key = "" if mode is None else str(mode).strip().lower()
    if mode_key == "train":
        return int(DEFAULT_TRAINING_LAUNCH_LEVEL)
    return int(DEFAULT_PLAY_LAUNCH_LEVEL)


def _resolve_osero_board_size_from_env() -> int | None:
    raw_value = os.getenv("OSERO_BOARD_SIZE")
    if raw_value is None:
        return None
    normalized = str(raw_value).strip().lower()
    if "x" in normalized:
        normalized = normalized.split("x", 1)[0].strip()
    try:
        board_size = int(normalized)
    except (TypeError, ValueError):
        return None
    if board_size not in set(int(value) for value in GENERIC_LAUNCH_LEVEL_TO_OSERO_BOARD_SIZE.values()):
        return None
    return int(board_size)


def resolve_generic_launch_level(
    game_id: str,
    generic_level: int | None,
    *,
    mode: str | None = None,
) -> int:
    game_key = str(game_id).strip().lower()
    min_level, max_level = _generic_launch_level_bounds(game_key)
    candidate = _default_generic_launch_level(game_key, mode=mode) if generic_level is None else int(generic_level)
    level_key = max(int(min_level), min(int(max_level), int(candidate)))
    if game_key == "osero":
        return 1
    return int(level_key)


def _refresh_osero_launch_modules() -> None:
    global _GAME_SPECS
    _GAME_SPECS = None
    for module_name in (
        "games.osero",
        "games.osero.config",
        "games.osero.rules",
        "games.osero.env",
    ):
        sys.modules.pop(module_name, None)


def apply_generic_launch_level(
    game_id: str,
    generic_level: int | None,
    *,
    mode: str | None = None,
) -> int:
    game_key = str(game_id).strip().lower()
    min_level, max_level = _generic_launch_level_bounds(game_key)
    candidate = _default_generic_launch_level(game_key, mode=mode) if generic_level is None else int(generic_level)
    level_key = max(int(min_level), min(int(max_level), int(candidate)))
    if game_key == "osero":
        env_board_size = _resolve_osero_board_size_from_env()
        board_size = (
            int(env_board_size)
            if generic_level is None and env_board_size is not None
            else int(GENERIC_LAUNCH_LEVEL_TO_OSERO_BOARD_SIZE[int(level_key)])
        )
        os.environ["OSERO_BOARD_SIZE"] = f"{int(board_size)}x{int(board_size)}"
        _refresh_osero_launch_modules()
    return int(resolve_generic_launch_level(game_key, level_key, mode=mode))


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
        "budget": int(max_steps),
        "checkpoint_every": int(checkpoint_every_steps),
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
        "budget": int(max_iterations),
        "rollout_steps": int(rollout_steps),
        "checkpoint_every": int(checkpoint_every_iterations),
        "reward_window": int(reward_window),
        "min_episodes_for_stats": int(min_episodes_for_stats),
    }


ALGO_RUNNER_OFF_POLICY = "off_policy"
ALGO_RUNNER_ON_POLICY = "on_policy"
ALGO_RUNNER_SEARCH_PLAY = "search_play"

_ALGO_SPECS: dict[str, AlgoSpec] | None = None
_GAME_SPECS: dict[str, GameSpec] | None = None


def _build_algo_specs() -> dict[str, AlgoSpec]:
    return {
        "qlearn": AlgoSpec(
            algo_id="qlearn",
            runner_kind=ALGO_RUNNER_OFF_POLICY,
            capabilities=AlgoCapabilities(supported_action_spaces=("discrete",)),
            defaults={
                "algo": {
                    "config": {
                        "hidden_sizes": [32],
                        "learning_rate": 1e-3,
                        "gamma": 0.95,
                        "max_memory": 100_000,
                        "batch_size": 512,
                        "exploration": build_exploration_config(
                            1.0,
                            0.05,
                            300_000,
                            patience_episodes=200,
                            min_improvement=0.10,
                            eps_bump_cap=0.25,
                            bump_cooldown_steps=150_000,
                        ),
                    }
                },
                "run": {
                    "train": build_off_policy_train_config(
                        max_steps=3_000_000,
                        checkpoint_every_steps=100_000,
                        reward_window=100,
                    )
                },
            },
        ),
        "dqn": AlgoSpec(
            algo_id="dqn",
            runner_kind=ALGO_RUNNER_OFF_POLICY,
            capabilities=AlgoCapabilities(
                supported_action_spaces=("discrete",),
                supports_masked_actions=True,
            ),
            defaults={
                "algo": {
                    "config": {
                        "hidden_sizes": [64, 64],
                        "learning_rate": 2.5e-4,
                        "weight_decay": 0.0,
                        "gamma": 0.99,
                        "batch_size": 128,
                        "replay_size": 80_000,
                        "target_sync_every": 500,
                        "grad_clip_norm": 10.0,
                        "exploration": build_exploration_config(
                            1.0,
                            0.05,
                            80_000,
                            patience_episodes=40,
                            min_improvement=0.08,
                            eps_bump_cap=0.30,
                            bump_cooldown_steps=4_000,
                        ),
                        "dueling": True,
                        "double_dqn": True,
                        "prioritized_replay": False,
                    }
                },
                "run": {
                    "train": build_off_policy_train_config(
                        max_steps=250_000,
                        checkpoint_every_steps=20_000,
                        reward_window=100,
                        train_after_steps=2_000,
                        update_every_steps=1,
                        updates_per_step=1,
                    )
                },
            },
        ),
        "sac": AlgoSpec(
            algo_id="sac",
            runner_kind=ALGO_RUNNER_OFF_POLICY,
            capabilities=AlgoCapabilities(supported_action_spaces=("continuous",)),
            defaults={
                "algo": {
                    "config": {
                        "hidden_sizes": [64, 64],
                        "learning_rate": 3e-4,
                        "gamma": 0.99,
                        "batch_size": 256,
                        "replay_size": 200_000,
                        "tau": 0.005,
                        "grad_clip_norm": 10.0,
                        "init_alpha": 0.20,
                    }
                },
                "run": {
                    "train": build_off_policy_train_config(
                        max_steps=10_000_000,
                        checkpoint_every_steps=100_000,
                        reward_window=100,
                        train_after_steps=10_000,
                        update_every_steps=1,
                        updates_per_step=1,
                    )
                },
            },
        ),
        "ppo": AlgoSpec(
            algo_id="ppo",
            runner_kind=ALGO_RUNNER_ON_POLICY,
            capabilities=AlgoCapabilities(
                supported_action_spaces=("discrete", "continuous"),
                supports_masked_actions=True,
                supports_multi_agent=True,
                supports_centralized_critic=True,
            ),
            defaults={
                "algo": {
                    "config": {
                        "hidden_sizes": [64, 64],
                        "learning_rate": 3e-4,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_ratio": 0.2,
                        "update_epochs": 4,
                        "minibatch_size": 256,
                        "entropy_coef": 0.015,
                        "value_coef": 0.5,
                        "max_grad_norm": 0.5,
                    }
                },
                "run": {
                    "train": build_on_policy_train_config(
                        max_iterations=8_000,
                        rollout_steps=1_024,
                        checkpoint_every_iterations=10,
                        reward_window=100,
                        min_episodes_for_stats=100,
                    )
                },
            },
        ),
        "a2c": AlgoSpec(
            algo_id="a2c",
            runner_kind=ALGO_RUNNER_ON_POLICY,
            capabilities=AlgoCapabilities(
                supported_action_spaces=("discrete", "continuous"),
                supports_masked_actions=True,
                supports_multi_agent=True,
                supports_centralized_critic=True,
            ),
            defaults={
                "algo": {
                    "config": {
                        "hidden_sizes": [64, 64],
                        "learning_rate": 3e-4,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_ratio": 999.0,
                        "update_epochs": 1,
                        "minibatch_size": 256,
                        "entropy_coef": 0.01,
                        "value_coef": 0.5,
                        "max_grad_norm": 0.5,
                        "policy_loss_mode": "a2c",
                    }
                },
                "run": {
                    "train": build_on_policy_train_config(
                        max_iterations=8_000,
                        rollout_steps=512,
                        checkpoint_every_iterations=10,
                        reward_window=100,
                        min_episodes_for_stats=100,
                    )
                },
            },
        ),
        "recurrent_ppo": AlgoSpec(
            algo_id="recurrent_ppo",
            runner_kind=ALGO_RUNNER_ON_POLICY,
            capabilities=AlgoCapabilities(
                supported_action_spaces=("discrete", "continuous"),
                supports_masked_actions=True,
                requires_recurrent_friendly=True,
            ),
            defaults={
                "algo": {
                    "config": {
                        "hidden_sizes": [32],
                        "learning_rate": 3e-4,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_ratio": 0.2,
                        "update_epochs": 4,
                        "minibatch_size": 256,
                        "entropy_coef": 0.04,
                        "value_coef": 0.5,
                        "max_grad_norm": 0.5,
                        "recurrent_type": "lstm",
                        "recurrent_hidden_size": 64,
                        "actor_head_hidden_sizes": [32],
                        "critic_head_hidden_sizes": [32],
                        "recurrent_seq_len": 64,
                    }
                },
                "run": {
                    "train": build_on_policy_train_config(
                        max_iterations=12_000,
                        rollout_steps=1_024,
                        checkpoint_every_iterations=10,
                        reward_window=100,
                        min_episodes_for_stats=100,
                    )
                },
            },
        ),
        "mappo": AlgoSpec(
            algo_id="mappo",
            runner_kind=ALGO_RUNNER_ON_POLICY,
            capabilities=AlgoCapabilities(
                supported_action_spaces=("discrete", "continuous"),
                supports_masked_actions=True,
                supports_multi_agent=True,
                supports_centralized_critic=True,
                requires_multi_agent=True,
            ),
            defaults={
                "algo": {
                    "config": {
                        "hidden_sizes": [64, 64],
                        "critic_hidden_sizes": [64, 64],
                        "centralized_critic": True,
                        "critic_condition_on_agent_obs": True,
                        "learning_rate": 3e-4,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_ratio": 0.2,
                        "update_epochs": 4,
                        "minibatch_size": 256,
                        "entropy_coef": 0.01,
                        "value_coef": 0.5,
                        "max_grad_norm": 0.5,
                    }
                },
                "run": {
                    "train": build_on_policy_train_config(
                        max_iterations=8_000,
                        rollout_steps=1_024,
                        checkpoint_every_iterations=10,
                        reward_window=100,
                        min_episodes_for_stats=100,
                    )
                },
            },
        ),
        "search_play": AlgoSpec(
            algo_id="search_play",
            runner_kind=ALGO_RUNNER_SEARCH_PLAY,
            capabilities=AlgoCapabilities(
                supported_action_spaces=("discrete",),
                supports_masked_actions=True,
                supports_self_play=True,
                requires_self_play=True,
            ),
            defaults={
                "algo": {
                    "config": {
                        "hidden_sizes": [64, 64],
                        "learning_rate": 1e-3,
                        "weight_decay": 1e-4,
                        "batch_size": 128,
                        "replay_size": 20_000,
                        "min_replay_to_train": 128,
                        "value_loss_weight": 1.0,
                        "grad_clip_norm": 5.0,
                        "simulations_per_move": 48,
                        "c_puct": 1.25,
                        "dirichlet_alpha": 0.35,
                        "dirichlet_epsilon": 0.25,
                        "temperature_sample_moves": 10,
                    }
                },
                "run": {
                    "train": {
                        "budget": 10_000,
                        "train_after_games": 8,
                        "updates_per_game": 2,
                        "checkpoint_every": 25,
                        "arena_every_games": 25,
                        "arena_games_per_opponent": 2,
                    }
                },
            },
        ),
}


def _algo_specs() -> dict[str, AlgoSpec]:
    global _ALGO_SPECS
    if _ALGO_SPECS is None:
        _ALGO_SPECS = _build_algo_specs()
    return _ALGO_SPECS


def get_algo_spec(algo_id: str) -> AlgoSpec:
    algo_key = str(algo_id).strip().lower()
    specs = _algo_specs()
    if algo_key not in specs:
        valid = ", ".join(sorted(specs.keys()))
        raise KeyError(f"Unknown algorithm '{algo_id}'. Valid options: {valid}")
    return specs[algo_key]


def _space_kind(space: Space) -> str:
    if isinstance(space, Discrete):
        return "discrete"
    if isinstance(space, Box):
        return "continuous"
    raise TypeError(f"Unsupported action space type '{type(space).__name__}'.")


def _capability_mismatch_reasons(spec: GameSpec, algo_spec: AlgoSpec) -> list[str]:
    game_caps = spec.capabilities
    algo_caps = algo_spec.capabilities
    space_kind = _space_kind(spec.action_space)
    reasons: list[str] = []

    if str(space_kind) not in tuple(str(value) for value in algo_caps.supported_action_spaces):
        reasons.append(
            f"game action_space={space_kind} is not supported by {algo_spec.algo_id} "
            f"({', '.join(algo_caps.supported_action_spaces)})"
        )
    if bool(game_caps.masked_actions) and not bool(algo_caps.supports_masked_actions):
        reasons.append("game requires masked action support")
    if bool(game_caps.multi_agent) and not bool(algo_caps.supports_multi_agent):
        reasons.append("game requires multi-agent support")
    if bool(game_caps.self_play) and not bool(algo_caps.supports_self_play):
        reasons.append("game requires self-play support")
    if bool(game_caps.centralized_critic_required) and not bool(algo_caps.supports_centralized_critic):
        reasons.append("game requires centralized critic support")
    if bool(algo_caps.requires_multi_agent) and not bool(game_caps.multi_agent):
        reasons.append(f"{algo_spec.algo_id} requires multi-agent game support")
    if bool(algo_caps.requires_self_play) and not bool(game_caps.self_play):
        reasons.append(f"{algo_spec.algo_id} requires self-play game support")
    if bool(algo_caps.requires_recurrent_friendly) and not bool(game_caps.recurrent_friendly):
        reasons.append(f"{algo_spec.algo_id} requires recurrent-friendly game support")
    return reasons


def validate_game_algo_compatibility(game_id: str, algo_id: str) -> None:
    spec = get_game_spec(game_id)
    algo_spec = get_algo_spec(algo_id)
    reasons = _capability_mismatch_reasons(spec, algo_spec)
    if reasons:
        reason_text = "; ".join(reasons)
        raise ValueError(
            f"Unsupported game/algo combination '{spec.game_id}/{algo_spec.algo_id}': {reason_text}."
        )
def _deep_merge_dicts(*layers: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for layer in layers:
        for key, value in dict(layer).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _deep_merge_dicts(
                    dict(merged[key]),
                    dict(value),
                )
            else:
                merged[key] = deepcopy(value)
    return merged


def _int_list(values: object | None, *, default: Iterable[int] | None = None) -> list[int]:
    if values is None:
        values = [] if default is None else list(default)
    if isinstance(values, (list, tuple)):
        return [int(value) for value in values]
    return [int(values)]


def parse_override_value(raw_value: str) -> object:
    text = str(raw_value).strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def set_nested_override(target: dict[str, object], path: str, value: object) -> None:
    parts = [part.strip() for part in str(path).split(".") if part.strip()]
    if not parts:
        raise ValueError("Override path must not be empty.")
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


def parse_override_assignments(assignments: Iterable[str] | None) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for assignment in assignments or ():
        if "=" not in str(assignment):
            raise ValueError(
                f"Invalid override '{assignment}'. Expected dotted.path=value, for example algo.config.learning_rate=0.0003."
            )
        path, raw_value = str(assignment).split("=", 1)
        set_nested_override(overrides, path.strip(), parse_override_value(raw_value))
    return overrides


def _normalize_mode(mode: str) -> str:
    mode_key = str(mode).strip().lower()
    if mode_key not in {"train", "eval", "play"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected train, eval, or play.")
    return mode_key


def _normalize_device(device: str) -> str:
    device_key = str(device).strip().lower()
    if device_key in {"cuda", "gpu"}:
        return "cuda"
    if device_key in {"cpu", ""}:
        return "cpu"
    if device_key == "auto":
        return "auto"
    raise ValueError(f"Unsupported device '{device}'. Expected cpu, cuda, or auto.")


def _runtime_uses_gpu(device: str) -> bool:
    normalized = _normalize_device(device)
    if normalized == "cuda":
        return True
    if normalized != "auto":
        return False
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def _game_defaults(spec: GameSpec) -> dict[str, object]:
    env_block = {
        "obs_dim": int(spec.obs_dim),
        "action_space": spec.action_space,
    }
    if spec.env_metadata:
        env_block = _deep_merge_dicts(env_block, dict(spec.env_metadata))
    return {
        "game": {
            "id": spec.game_id,
            "capabilities": asdict(spec.capabilities),
            "env": env_block,
        }
    }


def _algo_defaults(algo_spec: AlgoSpec) -> dict[str, object]:
    algo_layer: dict[str, object] = {
        "algo": {
            "id": algo_spec.algo_id,
            "runner_kind": algo_spec.runner_kind,
            "capabilities": asdict(algo_spec.capabilities),
            "config": {},
        },
        "run": {
            "name": "",
            "train": {},
            "paths": {},
        },
    }
    if algo_spec.defaults:
        algo_layer = _deep_merge_dicts(algo_layer, deepcopy(algo_spec.defaults))
    return algo_layer


def _game_model_defaults(spec: GameSpec) -> dict[str, object]:
    if not spec.default_model_config:
        return {}
    return {
        "algo": {
            "config": deepcopy(spec.default_model_config),
        }
    }


def _game_algo_defaults(spec: GameSpec, algo_id: str) -> dict[str, object]:
    algo_key = str(algo_id).strip().lower()
    layer: dict[str, object] = {}
    if algo_key in spec.algo_config_overrides:
        layer = _deep_merge_dicts(
            layer,
            {
                "algo": {
                    "config": deepcopy(spec.algo_config_overrides[algo_key]),
                }
            },
        )
    if algo_key == str(spec.default_algo).strip().lower() and spec.default_train_config:
        layer = _deep_merge_dicts(
            layer,
            {
                "run": {
                    "train": deepcopy(spec.default_train_config),
                }
            },
        )
    return layer


def _game_train_defaults(spec: GameSpec) -> dict[str, object]:
    if not spec.default_train_config or "budget" not in spec.default_train_config:
        return {}
    return {
        "run": {
            "train": {
                "budget": int(spec.default_train_config["budget"]),
            },
        }
    }


def _resolve_algo_id(spec: GameSpec, algo_override: str | None) -> str:
    override_key = "" if algo_override is None else str(algo_override).strip().lower()
    if override_key in DEFAULT_ALGO_TOKENS:
        return str(spec.default_algo).strip().lower()
    return str(override_key)


def _materialize_runner_train_block(
    *,
    common: dict[str, object],
    runner_kind: str,
    train_block: dict[str, object],
) -> dict[str, object]:
    resolved = dict(train_block)
    budget = resolved.pop("budget", None)
    checkpoint_every = resolved.pop("checkpoint_every", None)

    if runner_kind == ALGO_RUNNER_OFF_POLICY:
        if budget is not None and "max_steps" not in resolved:
            resolved["max_steps"] = max(1, int(budget))
        if checkpoint_every is not None and "checkpoint_every_steps" not in resolved:
            resolved["checkpoint_every_steps"] = max(1, int(checkpoint_every))
        total_steps = common.get("total_steps")
        if total_steps is not None:
            resolved["max_steps"] = max(1, int(total_steps))
        episodes = common.get("episodes")
        if episodes is not None:
            resolved["max_episodes"] = max(1, int(episodes))
    elif runner_kind == ALGO_RUNNER_ON_POLICY:
        rollout_steps = max(1, int(resolved.get("rollout_steps", 1024)))
        if budget is not None and "max_iterations" not in resolved:
            resolved["max_iterations"] = max(1, int(math.ceil(float(budget) / float(rollout_steps))))
        if checkpoint_every is not None and "checkpoint_every_iterations" not in resolved:
            resolved["checkpoint_every_iterations"] = max(1, int(checkpoint_every))
        total_steps = common.get("total_steps")
        if total_steps is not None:
            resolved["max_iterations"] = max(1, int(math.ceil(float(total_steps) / float(rollout_steps))))
    elif runner_kind == ALGO_RUNNER_SEARCH_PLAY:
        if budget is not None and "max_games" not in resolved:
            resolved["max_games"] = max(1, int(budget))
        if checkpoint_every is not None and "checkpoint_every_games" not in resolved:
            resolved["checkpoint_every_games"] = max(1, int(checkpoint_every))
        episodes = common.get("episodes")
        if episodes is not None:
            resolved["max_games"] = max(1, int(episodes))
        elif common.get("total_steps") is not None:
            resolved["max_games"] = max(1, int(common["total_steps"]))

    save_every = common.get("save_every")
    if save_every is not None:
        cadence = max(1, int(save_every))
        if runner_kind == ALGO_RUNNER_OFF_POLICY:
            resolved["checkpoint_every_steps"] = cadence
        elif runner_kind == ALGO_RUNNER_ON_POLICY:
            resolved["checkpoint_every_iterations"] = cadence
        elif runner_kind == ALGO_RUNNER_SEARCH_PLAY:
            resolved["checkpoint_every_games"] = cadence

    return resolved


def _derive_search_play_board_size(game_env: dict[str, object], algo_config: dict[str, object]) -> int:
    board_size = algo_config.get("board_size", game_env.get("board_size"))
    if board_size is not None:
        return max(1, int(board_size))
    obs_dim = int(game_env.get("obs_dim", 0))
    if obs_dim <= 0:
        raise ValueError("Unable to derive search_play board_size without game.env.board_size or obs_dim.")
    inferred = max(1, int(round(math.sqrt(float(obs_dim)))))
    if int(inferred) * int(inferred) != int(obs_dim):
        raise ValueError(f"Unable to derive square board_size from obs_dim={obs_dim}.")
    return int(inferred)


def _derive_run_name(spec: GameSpec, algo_spec: AlgoSpec, composed: dict[str, object]) -> str:
    game_env = dict(dict(composed.get("game", {})).get("env", {}))
    algo_config = dict(dict(composed.get("algo", {})).get("config", {}))
    hidden_sizes = _int_list(algo_config.get("hidden_sizes"))

    if algo_spec.algo_id == "search_play":
        board_size = _derive_search_play_board_size(game_env, algo_config)
        if hidden_sizes:
            return f"b{int(board_size)}_{'_'.join(str(size) for size in hidden_sizes)}"
        return f"b{int(board_size)}"

    recurrent_type = str(algo_config.get("recurrent_type", "none")).strip().lower()
    if recurrent_type in {"lstm", "gru"}:
        actor_head_hidden_sizes = _int_list(algo_config.get("actor_head_hidden_sizes"), default=[32])
        critic_head_hidden_sizes = _int_list(algo_config.get("critic_head_hidden_sizes"), default=[32])
        recurrent_hidden_size = int(algo_config.get("recurrent_hidden_size", 64))
        encoder_hidden_sizes = hidden_sizes or [32]
        return build_recurrent_run_name(
            encoder_hidden_sizes,
            recurrent_type=recurrent_type,
            recurrent_hidden_size=recurrent_hidden_size,
            actor_head_hidden_sizes=actor_head_hidden_sizes,
            critic_head_hidden_sizes=critic_head_hidden_sizes,
        )

    critic_hidden_sizes = _int_list(algo_config.get("critic_hidden_sizes"))
    if hidden_sizes and critic_hidden_sizes and critic_hidden_sizes != hidden_sizes:
        return build_actor_critic_run_name(hidden_sizes, critic_hidden_sizes)
    if hidden_sizes:
        return build_hidden_run_name(hidden_sizes)
    return f"{spec.game_id}_{algo_spec.algo_id}"


def compose_run_config(
    game_id: str,
    algo_override: str | None = None,
    *,
    mode: str = "train",
    user_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    spec = get_game_spec(game_id)
    resolved_mode = _normalize_mode(mode)
    algo_id = _resolve_algo_id(spec, algo_override)
    algo_spec = get_algo_spec(algo_id)
    validate_game_algo_compatibility(spec.game_id, algo_spec.algo_id)

    composed = _deep_merge_dicts(
        {
            "common": {
                "mode": resolved_mode,
                "seed": None,
                "device": spec.device,
                "render": False,
                "headless": False,
                "total_steps": None,
                "episodes": None,
                "checkpoint_path": None,
                "save_every": None,
            }
        },
        _game_defaults(spec),
        _algo_defaults(algo_spec),
        _game_train_defaults(spec),
        _game_model_defaults(spec),
        _game_algo_defaults(spec, algo_id),
        {} if user_overrides is None else dict(user_overrides),
    )

    common = dict(composed.setdefault("common", {}))
    common["mode"] = resolved_mode
    common["device"] = _normalize_device(str(common.get("device", "cpu")))
    common["render"] = bool(common.get("render", False))
    common["headless"] = bool(common.get("headless", False))
    if bool(common["headless"]):
        common["render"] = False
    composed["common"] = common

    algo_block = dict(composed.setdefault("algo", {}))
    algo_block["id"] = algo_spec.algo_id
    algo_block["runner_kind"] = algo_spec.runner_kind
    composed["algo"] = algo_block

    run_block = dict(composed.setdefault("run", {}))
    run_block.setdefault("train", {})
    run_block.setdefault("paths", {})
    if not str(run_block.get("name", "")).strip():
        run_block["name"] = _derive_run_name(spec, algo_spec, composed)
    composed["run"] = run_block
    return composed


def _build_game_specs() -> dict[str, GameSpec]:
    # Import config/env lazily so active games can be built straight from
    # their declarative config without separate spec shims.
    from games.bang import config as bang_config
    from games.bang.env import BangEnv
    from games.jump import config as jump_config
    from games.jump.env import JumpEnv
    from games.kick import config as kick_config
    from games.kick.env import KickEnv
    from games.osero import config as osero_config
    from games.osero.env import OseroEnv
    from games.snake import config as snake_config
    from games.snake.env import SnakeEnv
    from games.vroom import config as vroom_config
    from games.vroom.env import VroomEnv

    specs = (
        build_game_spec_from_config(config_module=snake_config, env_type=SnakeEnv),
        build_game_spec_from_config(config_module=bang_config, env_type=BangEnv),
        build_game_spec_from_config(config_module=jump_config, env_type=JumpEnv),
        build_game_spec_from_config(config_module=vroom_config, env_type=VroomEnv),
        build_game_spec_from_config(config_module=osero_config, env_type=OseroEnv),
        build_game_spec_from_config(config_module=kick_config, env_type=KickEnv),
    )
    return {spec.game_id: spec for spec in specs}


def _game_specs() -> dict[str, GameSpec]:
    global _GAME_SPECS
    if _GAME_SPECS is None:
        _GAME_SPECS = _build_game_specs()
    return _GAME_SPECS


def get_game_spec(game_id: str) -> GameSpec:
    game_key = str(game_id).strip().lower()
    specs = _game_specs()
    if game_key not in specs:
        valid = ", ".join(sorted(specs.keys()))
        raise KeyError(f"Unknown game '{game_id}'. Valid options: {valid}")
    return specs[game_key]


def _attach_run_paths(config: dict[str, object], run_paths: RunPaths) -> dict[str, object]:
    attached = _deep_merge_dicts(config)
    run_block = dict(attached.get("run", {}))
    run_paths_block = dict(run_block.get("paths", {}))
    run_paths_block.update(
        {
            "dir": str(run_paths.run_dir),
            "metrics": str(run_paths.metrics_path),
        }
    )
    run_block["paths"] = run_paths_block
    attached["run"] = run_block
    return attached


def build_env_from_config(
    config: dict[str, object],
    *,
    mode: str | None = None,
    render: bool | None = None,
    level: int | None = None,
) -> Env:
    game_id = str(dict(config.get("game", {})).get("id", "")).strip().lower()
    spec = get_game_spec(game_id)
    common = dict(config.get("common", {}))
    if mode is None:
        resolved_mode = _normalize_mode(str(common.get("mode", "train")))
    else:
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in {"train", "eval", "play", "human"}:
            raise ValueError(f"Unsupported env mode '{mode}'. Expected train, eval, play, or human.")
    resolved_render = bool(common.get("render", False) if render is None else render)
    if bool(common.get("headless", False)):
        resolved_render = False
    return spec.make_env(mode=resolved_mode, render=bool(resolved_render), level=level)


def _resolve_algo_runtime_config(config: dict[str, object]) -> tuple[str, int, Space, dict[str, object]]:
    game_block = dict(config.get("game", {}))
    game_env = dict(game_block.get("env", {}))
    common = dict(config.get("common", {}))
    algo_block = dict(config.get("algo", {}))
    algo_id = str(algo_block.get("id", "")).strip().lower()
    algo_config = dict(algo_block.get("config", {}))
    algo_config["use_gpu"] = _runtime_uses_gpu(str(common.get("device", "cpu")))

    if bool(dict(game_block.get("capabilities", {})).get("centralized_critic_required", False)):
        algo_config.setdefault("centralized_critic", True)
        algo_config.setdefault("critic_condition_on_agent_obs", True)
        if "critic_obs_dim" not in algo_config and game_env.get("central_obs_dim") is not None:
            algo_config["critic_obs_dim"] = int(game_env["central_obs_dim"])

    if algo_id == "search_play" and "board_size" not in algo_config:
        algo_config["board_size"] = _derive_search_play_board_size(game_env, algo_config)

    obs_dim = int(game_env.get("obs_dim", 0))
    action_space = game_env.get("action_space")
    if not isinstance(action_space, Space):
        raise TypeError("Composed config is missing a valid game.env.action_space.")
    return algo_id, int(obs_dim), action_space, algo_config


def build_algo_from_config(config: dict[str, object]):
    from core.algorithms.factory import build_algorithm

    algo_id, obs_dim, action_space, algo_config = _resolve_algo_runtime_config(config)
    return build_algorithm(
        algo_id=algo_id,
        obs_dim=int(obs_dim),
        action_space=action_space,
        algo_config=algo_config,
    )


def build_runner_from_config(config: dict[str, object]) -> Callable[[Env, "Algorithm", RunPaths], dict[str, object]]:
    common = dict(config.get("common", {}))
    algo_block = dict(config.get("algo", {}))
    runner_kind = str(algo_block.get("runner_kind", "")).strip().lower()
    train_block = _materialize_runner_train_block(
        common=common,
        runner_kind=runner_kind,
        train_block=dict(dict(config.get("run", {})).get("train", {})),
    )

    if runner_kind == ALGO_RUNNER_OFF_POLICY:
        from core.runners.off_policy import OffPolicyConfig, run_off_policy_training

        runner_config = OffPolicyConfig(**train_block)
        return lambda env, algorithm, run_paths: run_off_policy_training(env, algorithm, run_paths, runner_config)

    if runner_kind == ALGO_RUNNER_ON_POLICY:
        from core.runners.on_policy import OnPolicyConfig, run_on_policy_training

        runner_config = OnPolicyConfig(**train_block)
        return lambda env, algorithm, run_paths: run_on_policy_training(env, algorithm, run_paths, runner_config)

    if runner_kind == ALGO_RUNNER_SEARCH_PLAY:
        from core.search_play.interfaces import SearchPlayTrainConfig
        from core.search_play.trainer import run_search_play_training

        runner_config = SearchPlayTrainConfig(**train_block)
        return lambda env, algorithm, run_paths: run_search_play_training(env, algorithm, run_paths, runner_config)

    raise ValueError(f"Unsupported runner kind '{runner_kind}'.")


def apply_seed_from_config(config: dict[str, object]) -> int | None:
    seed = dict(config.get("common", {})).get("seed")
    if seed is None:
        return None
    seed_value = int(seed)
    random.seed(seed_value)
    try:
        import numpy as np

        np.random.seed(seed_value)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed_value)
        if bool(torch.cuda.is_available()):
            torch.cuda.manual_seed_all(seed_value)
    except ModuleNotFoundError:
        pass
    return seed_value


def prepare_run(
    game_id: str,
    algo_override: str | None = None,
    *,
    mode: str = "train",
    user_overrides: dict[str, object] | None = None,
) -> PreparedRun:
    config = compose_run_config(game_id, algo_override, mode=mode, user_overrides=user_overrides)
    algo_id = str(config["algo"]["id"]).strip().lower()
    game_id = str(config["game"]["id"]).strip().lower()
    run_name = str(dict(config.get("run", {})).get("name", "")).strip()
    run_paths = resolve_run_paths(game_id, algo_id, run_name, create=True)
    attached_config = _attach_run_paths(config, run_paths)
    return PreparedRun(run_paths=run_paths, config=attached_config)


def resolve_current_level(env: object, *, default: int = 1) -> int:
    level_value = getattr(env, "_current_level", None)
    if level_value is None:
        game = getattr(env, "game", None)
        level_value = getattr(game, "level", None)
    try:
        return max(1, int(level_value))
    except (TypeError, ValueError):
        return max(1, int(default))


def resume_level_bounds(env: object) -> tuple[int, int]:
    curriculum = getattr(env, "_curriculum", None)
    if curriculum is None:
        current_level = resolve_current_level(env, default=1)
        return int(current_level), int(current_level)
    curriculum_config = getattr(curriculum, "config", None)
    min_level = max(1, int(getattr(curriculum_config, "min_level", 1)))
    max_level = max(min_level, int(getattr(curriculum_config, "max_level", DEFAULT_CURRICULUM_MAX_LEVEL)))
    return int(min_level), int(max_level)


def resolve_best_resume_level(
    env: object,
    *,
    explicit_level: int | None = None,
    allow_prompt: bool = False,
) -> int:
    min_level, max_level = resume_level_bounds(env)
    if explicit_level is not None:
        return max(int(min_level), min(int(max_level), int(explicit_level)))
    if min_level == max_level:
        return int(min_level)
    if not bool(allow_prompt) or not bool(sys.stdin.isatty()):
        raise ValueError(
            "Resume mode 'best' requires --resume-level for games with curriculum levels. "
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
    if mode_key == "check":
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


def resolve_latest_play_model_path(run_paths: RunPaths, level: int) -> Path:
    level_value = max(1, int(level))
    candidates: list[tuple[int, int, Path]] = []
    for kind, priority in (("best", 1), ("check", 0)):
        path = run_paths.model_path(level_value, kind)
        if path.exists():
            candidates.append((int(path.stat().st_mtime_ns), int(priority), path))
    if not candidates:
        expected_best = run_paths.model_path(level_value, "best")
        raise FileNotFoundError(f"No model found for level {level_value} at '{expected_best.parent}'.")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


__all__ = [
    "ALGO_RUNNER_OFF_POLICY",
    "ALGO_RUNNER_ON_POLICY",
    "ALGO_RUNNER_SEARCH_PLAY",
    "EXPLORATION_AVG_WINDOW_EPISODES",
    "MIN_EPISODES_FOR_STATS",
    "OFF_POLICY_TRAIN_DEFAULTS",
    "ACTIVE_GAME_ORDER",
    "GameSpec",
    "GameCapabilities",
    "AlgoSpec",
    "AlgoCapabilities",
    "PreparedRun",
    "build_env_factory",
    "build_exploration_config",
    "build_off_policy_train_config",
    "build_on_policy_train_config",
    "apply_seed_from_config",
    "build_algo_from_config",
    "build_env_from_config",
    "build_runner_from_config",
    "compose_run_config",
    "get_algo_spec",
    "get_game_spec",
    "parse_override_assignments",
    "prepare_run",
    "set_nested_override",
    "validate_game_algo_compatibility",
    "resolve_current_level",
    "resume_level_bounds",
    "resolve_best_resume_level",
    "apply_training_start_level",
    "resolve_resume_path",
    "resolve_play_model_path",
    "resolve_latest_play_model_path",
]
