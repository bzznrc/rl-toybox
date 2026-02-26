"""Training entrypoint for Snake AI."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import deque
import logging
import random

import torch

from snake_ai.config import (
    AVG100_WINDOW,
    BATCH_SIZE,
    BEST_MODEL_MIN_EPISODES,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    HIDDEN_DIMENSIONS,
    LEARNING_RATE,
    LOAD_MODEL,
    MAX_MEMORY,
    MODEL_BEST_PATH,
    MODEL_CHECKPOINT_PATH,
    NUM_ACTIONS,
    NUM_INPUT_FEATURES,
    resolve_show_game,
)
from snake_ai.game import TrainingSnakeGame
from snake_ai.logging_utils import configure_logging, log_run_context
from snake_ai.model import LinearQNet, QTrainer

LOGGER = logging.getLogger("snake_ai.train")


def resolve_model_load_path() -> str | None:
    if LOAD_MODEL is False:
        return None
    if LOAD_MODEL == "B":
        return MODEL_BEST_PATH
    if LOAD_MODEL == "L":
        return MODEL_CHECKPOINT_PATH
    raise ValueError('Invalid LOAD_MODEL value. Use False, "B", or "L".')


def try_save_model(model, path: str, success_message: str) -> bool:
    try:
        model.save(path)
    except RuntimeError as exc:
        LOGGER.warning("save failed (%s): %s", path, exc)
        return False
    LOGGER.info("%s", success_message)
    return True


class DQNAgent:
    """Agent that interacts with the environment and learns from it."""

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(NUM_INPUT_FEATURES, HIDDEN_DIMENSIONS, NUM_ACTIONS)
        load_path = resolve_model_load_path()
        if load_path is not None:
            self.model.load(load_path)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=GAMMA)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> float:
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done) -> None:
        self.trainer.train_step(state, action, reward, next_state, done)

    def select_action(self, state) -> list[int]:
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        if random.random() < self.epsilon:
            move = random.randint(0, NUM_ACTIONS - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        action = [0] * NUM_ACTIONS
        action[move] = 1
        return action


def train() -> None:
    configure_logging()
    record = 0
    score_window = deque(maxlen=AVG100_WINDOW)
    best_avg100 = float("-inf")

    agent = DQNAgent()
    game = TrainingSnakeGame(show_game=resolve_show_game(default_value=False))
    log_run_context(
        "train-ai",
        {
            "model": MODEL_CHECKPOINT_PATH,
            "model_best": MODEL_BEST_PATH,
            "load_mode": LOAD_MODEL,
            "batch_size": BATCH_SIZE,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
        },
    )

    while True:
        state_old = game.get_state_vector()
        action = agent.select_action(state_old)

        reward, done, score = game.play_step(action)
        state_new = game.get_state_vector()

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            score_window.append(score)
            avg100 = sum(score_window) / len(score_window)

            if agent.n_games % 50 == 0:
                try_save_model(agent.model, MODEL_CHECKPOINT_PATH, "Model Saved")

            if agent.n_games >= BEST_MODEL_MIN_EPISODES and avg100 > best_avg100:
                best_avg100 = avg100
                try_save_model(agent.model, MODEL_BEST_PATH, "New Best Model")

            if score > record:
                record = score

            LOGGER.info(
                "Episode: %s\tScore: %s\tRecord: %s\tAvg%s: %.2f\tBestAvg: %.2f\tEpsilon = %.3f",
                agent.n_games,
                score,
                record,
                AVG100_WINDOW,
                avg100,
                best_avg100 if best_avg100 > float("-inf") else avg100,
                agent.epsilon,
            )


if __name__ == "__main__":
    train()
