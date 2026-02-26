"""Run a trained Snake model for quick evaluation."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
else:
    from pathlib import Path

import torch

from snake_ai.config import (
    HIDDEN_DIMENSIONS,
    LOAD_MODEL,
    MODEL_BEST_PATH,
    MODEL_CHECKPOINT_PATH,
    NUM_ACTIONS,
    NUM_INPUT_FEATURES,
    resolve_show_game,
)
from snake_ai.game import TrainingSnakeGame
from snake_ai.logging_utils import configure_logging, log_key_values, log_run_context
from snake_ai.model import LinearQNet


class GameModelRunner:
    """Load a trained model and run greedy evaluation episodes."""

    def __init__(self, model_path: str = MODEL_BEST_PATH):
        self.model_path = model_path
        self.model = LinearQNet(NUM_INPUT_FEATURES, HIDDEN_DIMENSIONS, NUM_ACTIONS)
        self.model.load(model_path)
        self.model.eval()
        self.game = TrainingSnakeGame(show_game=resolve_show_game(default_value=True))

    def select_action(self, state) -> list[int]:
        with torch.no_grad():
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        action = [0] * NUM_ACTIONS
        action[move] = 1
        return action

    def run(self, episodes: int = 10) -> None:
        total_score = 0
        best_score = 0

        for episode in range(1, episodes + 1):
            self.game.reset()
            done = False

            while not done:
                state = self.game.get_state_vector()
                action = self.select_action(state)
                _, done, score = self.game.play_step(action)

            total_score += score
            best_score = max(best_score, score)
            avg_score = total_score / episode
            log_key_values(
                "snake_ai.play_ai",
                {
                    "Episode": f"{episode:>4}",
                    "Score": f"{score:>4}",
                    "Best": f"{best_score:>4}",
                    "Avg": f"{avg_score:>7.3f}",
                },
            )

        self.game.close()
        avg_score = total_score / max(1, episodes)
        log_key_values(
            "snake_ai.play_ai",
            {"Episodes": episodes, "Avg Score": avg_score, "Best Score": best_score},
            prefix="Play AI Summary",
        )


def run_ai(episodes: int = 10) -> None:
    configure_logging()
    if Path(MODEL_BEST_PATH).exists():
        model_path = MODEL_BEST_PATH
    elif Path(MODEL_CHECKPOINT_PATH).exists():
        model_path = MODEL_CHECKPOINT_PATH
    else:
        raise FileNotFoundError(f"No model found at '{MODEL_BEST_PATH}' or '{MODEL_CHECKPOINT_PATH}'.")
    runner = GameModelRunner(model_path=model_path)
    log_run_context("play-ai", {"episodes": episodes, "model": model_path, "load_mode": LOAD_MODEL})
    runner.run(episodes=episodes)


if __name__ == "__main__":
    run_ai()
