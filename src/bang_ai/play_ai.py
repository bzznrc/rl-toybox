"""Run a trained model in the arena for quick evaluation."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from bang_ai.config import (
    LOAD_MODEL,
    MAX_LEVEL,
    MIN_LEVEL,
    MODEL_BEST_PATH,
    MODEL_CHECKPOINT_PATH,
    NUM_ACTIONS,
    PLAY_OPPONENT_LEVEL,
    SHOW_GAME_OVERRIDE,
)
from bang_ai.game import TrainingGame
from bang_ai.logging_utils import configure_logging, format_display_path, log_run_context
from bang_ai.model import build_loaded_q_network, device
from bang_ai.utils import resolve_show_game


class GameModelRunner:
    """Loads a trained model and plays episodes greedily."""

    def __init__(self, model_path: str = MODEL_BEST_PATH):
        self.level = max(MIN_LEVEL, min(PLAY_OPPONENT_LEVEL, MAX_LEVEL))
        self.model_path = model_path
        self.game = TrainingGame(
            level=self.level,
            show_game=resolve_show_game(SHOW_GAME_OVERRIDE, default_value=True),
            end_on_player_death=False,
        )
        self.model, _ = build_loaded_q_network(load_path=model_path, strict=True)
        self.model.eval()

    def select_action(self, state) -> list[int]:
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32, device=device))
            action_idx = int(torch.argmax(q_values).item())
        action = [0] * NUM_ACTIONS
        action[action_idx] = 1
        return action

    def run(self, episodes: int = 10) -> None:
        log_run_context(
            "play-ai",
            {
                "episodes": episodes,
                "model": self.model_path,
                "load_mode": LOAD_MODEL,
                "level": self.level,
            },
        )
        wins = 0
        for _ in range(episodes):
            self.game.reset()
            done = False
            while not done:
                state = self.game.get_state_vector()
                action = self.select_action(state)
                _, done, _ = self.game.play_step(action)
            if self.game.is_player_last_alive():
                wins += 1
        self.game.close()
        print(f"Model win rate: {wins}/{episodes} ({(wins / episodes) * 100:.1f}%)")


def run_ai(episodes: int = 10) -> None:
    configure_logging()
    runner = None
    try:
        runner = GameModelRunner(MODEL_BEST_PATH)
    except (FileNotFoundError, RuntimeError):
        try:
            runner = GameModelRunner(MODEL_CHECKPOINT_PATH)
        except (FileNotFoundError, RuntimeError):
            raise FileNotFoundError(
                "No compatible model found at "
                f"'{format_display_path(MODEL_BEST_PATH)}' or '{format_display_path(MODEL_CHECKPOINT_PATH)}'."
            )
    runner.run(episodes=episodes)


if __name__ == "__main__":
    run_ai()
