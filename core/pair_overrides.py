"""Pair-specific tuning deltas for the composition layer."""

from __future__ import annotations

from core.game import build_exploration_config
from games.cardz import config as cardz_config
from games.kick import config as kick_config
from games.osero import config as osero_config


PAIR_OVERRIDES: dict[tuple[str, str], dict[str, object]] = {
    (
        "bang",
        "dqn",
    ): {
        "algo": {
            "config": {
                "batch_size": 256,
                "replay_size": 500_000,
                "target_sync_every": 10_000,
                "weight_decay": 1e-5,
                "exploration": build_exploration_config(
                    1.0,
                    0.05,
                    2_500_000,
                    patience_episodes=150,
                    min_improvement=0.10,
                    eps_bump_cap=0.35,
                    bump_cooldown_steps=1_250_000,
                ),
                "prioritized_replay": True,
                "per_alpha": 0.6,
                "per_beta_start": 0.4,
                "per_beta_frames": 10_000_000,
                "per_epsilon": 1e-4,
            }
        },
        "run": {
            "train": {
                "max_steps": 10_000_000,
                "train_after_steps": 50_000,
                "update_every_steps": 4,
                "updates_per_step": 1,
                "checkpoint_every_steps": 200_000,
            }
        },
    },
    (
        "osero",
        "search_play",
    ): {
        "algo": {
            "config": {
                "hidden_sizes": list(osero_config.POLICY_VALUE_HIDDEN_DIMENSIONS),
                "simulations_per_move": int(osero_config.SIMULATIONS_PER_MOVE),
                "dirichlet_alpha": float(osero_config.DIRICHLET_ALPHA),
            }
        },
    },
    (
        "cardz",
        "a2c",
    ): {
        "algo": {
            "config": {
                "share_backbone": True,
                "entropy_coef": float(cardz_config.LEVEL_SETTINGS[int(cardz_config.MIN_LEVEL)]["entropy_coef"]),
            }
        },
    },
    (
        "kick",
        "ppo",
    ): {
        "algo": {
            "config": {
                "hidden_sizes": [96, 96],
                "critic_hidden_sizes": [192, 192],
                "minibatch_size": 512,
                "entropy_coef": float(kick_config.LEVEL_SETTINGS[int(kick_config.MIN_LEVEL)]["entropy_coef"]),
            }
        },
        "run": {
            "train": {
                "max_iterations": 12_000,
                "rollout_steps": 2_048,
                "checkpoint_every_iterations": 10,
            }
        },
    },
}
