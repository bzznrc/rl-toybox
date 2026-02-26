"""Training entrypoint for Bang AI."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import deque
import logging
import random

import torch

from bang_ai.config import (
    BATCH_SIZE,
    BEST_MODEL_MIN_EPISODES,
    CHECKPOINT_EVERY_STEPS,
    CURRICULUM_CONSECUTIVE_CHECKS,
    CURRICULUM_MIN_EPISODES_PER_LEVEL,
    CURRICULUM_REWARD_THRESHOLDS,
    EPISODE_CHECKPOINT_EVERY,
    EPSILON_DECAY_EPISODES,
    EPSILON_EXPLORATION_CAP,
    EPSILON_LEVEL_UP_RESET,
    EPSILON_MIN,
    EPSILON_STAGNATION_BOOST,
    EPSILON_START_RESUME,
    EPSILON_START_SCRATCH,
    GRADIENT_STEPS_PER_UPDATE,
    LEARN_START_STEPS,
    LOAD_MODEL,
    MAX_LEVEL,
    MIN_LEVEL,
    MODEL_BEST_PATH,
    MODEL_CHECKPOINT_PATH,
    NUM_ACTIONS,
    PATIENCE,
    PER_ALPHA,
    PER_BETA_FRAMES,
    PER_BETA_START,
    PER_EPSILON,
    REPLAY_BUFFER_SIZE,
    RESUME_LEVEL,
    REWARD_ROLLING_WINDOW,
    SHOW_GAME_OVERRIDE,
    STAGNATION_IMPROVEMENT_THRESHOLD,
    STAGNATION_WINDOW,
    TARGET_SYNC_EVERY,
    TOTAL_TRAINING_STEPS,
    TRAIN_EVERY_STEPS,
)
from bang_ai.game import TrainingGame
from bang_ai.logging_utils import configure_logging, format_display_path, log_key_values, log_run_context
from bang_ai.model import DQNTrainer, build_loaded_q_network, device
from bang_ai.utils import resolve_show_game

LOGGER = logging.getLogger("bang_ai.train")


def resolve_training_start_level(resume_level: int | None) -> int:
    if resume_level is None:
        level = int(MIN_LEVEL)
    else:
        level = int(resume_level)
    if not (MIN_LEVEL <= level <= MAX_LEVEL):
        raise ValueError(
            f"Training start level must be in [{MIN_LEVEL}, {MAX_LEVEL}], got {level}. "
            "Set RESUME_LEVEL accordingly (or None to use MIN_LEVEL)."
        )
    return level


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
        LOGGER.warning("save failed (%s): %s", format_display_path(path), exc)
        return False
    LOGGER.info("%s", success_message)
    return True


class SumTree:
    """Binary sum tree to sample proportional to priority in O(log N)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    @property
    def total(self):
        return self.tree[1]

    def add(self, priority, data):
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 1:
            idx //= 2
            self.tree[idx] += change

    def get(self, value):
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """PER with proportional prioritization and importance sampling."""

    def __init__(self, capacity, alpha, beta_start, beta_frames, epsilon):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(1, beta_frames)
        self.epsilon = epsilon
        self.frame = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.size

    def add(self, transition):
        priority = self.max_priority**self.alpha
        return self.tree.add(priority, transition)

    def sample(self, batch_size):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total / batch_size
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            idx, priority, data = self.tree.get(value)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        sampling_probabilities = [p / self.tree.total for p in priorities]
        is_weights = [(self.tree.size * p) ** (-beta) for p in sampling_probabilities]
        max_weight = max(is_weights) if is_weights else 1.0
        is_weights = [w / max_weight for w in is_weights]
        return batch, indices, is_weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


class DQNAgent:
    """Replay-buffer DQN agent with target network synchronization."""

    def __init__(self):
        self.episodes_played = 0
        self.epsilon = 0.0
        self.epsilon_boost = 0.0
        self.epsilon_start = EPSILON_START_SCRATCH
        self.epsilon_decay_per_episode = (EPSILON_START_SCRATCH - EPSILON_MIN) / max(1, EPSILON_DECAY_EPISODES)
        self.stagnation_rewards = deque(maxlen=STAGNATION_WINDOW)
        self.best_stagnation_average = None
        self.stagnation_episodes = 0
        self.memory = PrioritizedReplayBuffer(
            capacity=REPLAY_BUFFER_SIZE,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START,
            beta_frames=PER_BETA_FRAMES,
            epsilon=PER_EPSILON,
        )
        self.total_env_steps = 0

        load_path = resolve_model_load_path()
        self.requested_model_path = load_path
        self.online_model, self.loaded_model_path = build_loaded_q_network(load_path=load_path, strict=False)
        self.target_model = self.online_model.copy().to(device)
        self.target_model.eval()

        self.epsilon_start = EPSILON_START_RESUME if self.loaded_model_path else EPSILON_START_SCRATCH
        self.epsilon = self.epsilon_start

        self.trainer = DQNTrainer(self.online_model, self.target_model)
        self.training_steps = 0

    def get_state(self, game: TrainingGame) -> list[float]:
        return [float(value) for value in game.get_state_vector()]

    def remember(self, state, action, reward, next_state, done):
        return self.memory.add((state, action, reward, next_state, done))

    def train_long_memory(self) -> float:
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        batch, indices, is_weights = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        loss, td_errors = self._train_batch(states, actions, rewards, next_states, dones, is_weights=is_weights)
        self.memory.update_priorities(indices, td_errors)
        return loss

    def _train_batch(self, states, actions, rewards, next_states, dones, is_weights=None):
        loss, td_errors = self.trainer.train_step(states, actions, rewards, next_states, dones, is_weights=is_weights)
        self.training_steps += 1
        if self.training_steps % TARGET_SYNC_EVERY == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())
        return loss, td_errors

    def select_action(self, state) -> list[int]:
        if random.random() < self.epsilon:
            action_idx = random.randint(0, NUM_ACTIONS - 1)
        else:
            with torch.no_grad():
                q_values = self.online_model(torch.tensor(state, dtype=torch.float32, device=device))
                action_idx = int(torch.argmax(q_values).item())

        action = [0] * NUM_ACTIONS
        action[action_idx] = 1
        return action

    def _base_epsilon(self):
        decayed = self.epsilon_start - self.episodes_played * self.epsilon_decay_per_episode
        return max(EPSILON_MIN, decayed)

    def update_epsilon(self):
        base_epsilon = self._base_epsilon()
        epsilon_upper_bound = max(self.epsilon_start, EPSILON_EXPLORATION_CAP)
        self.epsilon = min(epsilon_upper_bound, max(EPSILON_MIN, base_epsilon + self.epsilon_boost))
        if self.epsilon_boost > 0.0:
            self.epsilon_boost = max(0.0, self.epsilon_boost - self.epsilon_decay_per_episode)

    def _set_exploration_target(self, target_epsilon: float):
        target_epsilon = max(EPSILON_MIN, min(EPSILON_EXPLORATION_CAP, target_epsilon))
        base_epsilon = self._base_epsilon()
        self.epsilon_boost = max(0.0, target_epsilon - base_epsilon)
        self.epsilon = target_epsilon

    def apply_stagnation_boost(self):
        if self.epsilon >= EPSILON_EXPLORATION_CAP:
            return False
        self._set_exploration_target(self.epsilon + EPSILON_STAGNATION_BOOST)
        return True

    def reset_epsilon_for_level_up(self):
        self._set_exploration_target(EPSILON_LEVEL_UP_RESET)
        self.stagnation_rewards.clear()
        self.best_stagnation_average = None
        self.stagnation_episodes = 0

    def update_stagnation_state(self, episode_reward: float) -> bool:
        self.stagnation_rewards.append(episode_reward)
        if len(self.stagnation_rewards) < STAGNATION_WINDOW:
            return False

        moving_avg = sum(self.stagnation_rewards) / len(self.stagnation_rewards)
        if self.best_stagnation_average is None:
            self.best_stagnation_average = moving_avg
            self.stagnation_episodes = 0
            return False

        if moving_avg > self.best_stagnation_average + STAGNATION_IMPROVEMENT_THRESHOLD:
            self.best_stagnation_average = moving_avg
            self.stagnation_episodes = 0
            return False

        self.stagnation_episodes += 1
        if self.stagnation_episodes >= PATIENCE:
            boosted = self.apply_stagnation_boost()
            self.stagnation_episodes = 0
            self.best_stagnation_average = moving_avg
            return boosted
        return False


class PerformanceCurriculum:
    """Progress levels using rolling reward thresholds."""

    def __init__(self, level: int):
        self.level = level
        self.episodes_at_level = 0
        self.consecutive_passes = 0
        expected_transitions = max(0, MAX_LEVEL - MIN_LEVEL)
        if len(CURRICULUM_REWARD_THRESHOLDS) < expected_transitions:
            raise ValueError(
                f"CURRICULUM_REWARD_THRESHOLDS must define at least {expected_transitions} values."
            )

    def threshold_for_current_level(self):
        if self.level >= MAX_LEVEL:
            return None
        idx = self.level - MIN_LEVEL
        if idx < 0 or idx >= len(CURRICULUM_REWARD_THRESHOLDS):
            return None
        return CURRICULUM_REWARD_THRESHOLDS[idx]

    def on_episode_end(self, avg_reward: float, rolling_ready: bool) -> bool:
        self.episodes_at_level += 1
        threshold = self.threshold_for_current_level()
        if threshold is None:
            return False

        if not rolling_ready or self.episodes_at_level < CURRICULUM_MIN_EPISODES_PER_LEVEL:
            self.consecutive_passes = 0
            return False

        if avg_reward > threshold:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        if self.consecutive_passes < CURRICULUM_CONSECUTIVE_CHECKS:
            return False

        self.level += 1
        self.episodes_at_level = 0
        self.consecutive_passes = 0
        return True


def train() -> None:
    configure_logging()
    reward_window = deque(maxlen=REWARD_ROLLING_WINDOW)

    agent = DQNAgent()
    level = resolve_training_start_level(RESUME_LEVEL)
    curriculum = PerformanceCurriculum(level=level)
    game = TrainingGame(
        level=curriculum.level,
        show_game=resolve_show_game(SHOW_GAME_OVERRIDE, default_value=False),
    )

    if agent.loaded_model_path:
        model_status = agent.loaded_model_path
    elif agent.requested_model_path:
        model_status = f"missing:{agent.requested_model_path}"
    else:
        model_status = "scratch"

    log_run_context(
        "train-ai",
        {
            "model": model_status,
            "load_mode": LOAD_MODEL,
            "steps": TOTAL_TRAINING_STEPS,
            "batch": BATCH_SIZE,
            "buffer": REPLAY_BUFFER_SIZE,
            "train_every": TRAIN_EVERY_STEPS,
            "level": curriculum.level,
        },
    )

    best_average_reward = float("-inf")

    while True:
        agent.update_epsilon()
        episode_reward = 0.0
        episode_losses = []
        done = False

        while not done:
            state_old = agent.get_state(game)
            action = agent.select_action(state_old)
            reward, done, _ = game.play_step(action)
            state_new = agent.get_state(game)

            agent.remember(state_old, action, reward, state_new, done)
            episode_reward += reward
            agent.total_env_steps += 1

            if agent.total_env_steps >= LEARN_START_STEPS and agent.total_env_steps % TRAIN_EVERY_STEPS == 0:
                for _ in range(GRADIENT_STEPS_PER_UPDATE):
                    loss = agent.train_long_memory()
                    if loss > 0.0:
                        episode_losses.append(loss)

            if agent.total_env_steps % CHECKPOINT_EVERY_STEPS == 0:
                try_save_model(agent.online_model, MODEL_CHECKPOINT_PATH, "Model Saved (step checkpoint)")

        agent.episodes_played += 1
        mean_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0

        reward_window.append(episode_reward)
        avg_reward = sum(reward_window) / len(reward_window)
        exploration_boosted = agent.update_stagnation_state(episode_reward)
        if exploration_boosted:
            log_key_values(
                LOGGER.name,
                {
                    "Event": "Exploration Bump",
                    "Epsilon": f"{agent.epsilon:.3f}",
                },
            )

        log_key_values(
            LOGGER.name,
            {
                "Episode": agent.episodes_played,
                "Level": curriculum.level,
                "Frames": game.frame_count,
                "Reward": f"{episode_reward:.2f}",
                f"Avg{REWARD_ROLLING_WINDOW}": f"{avg_reward:.2f}",
                "Best": f"{best_average_reward:.2f}",
                "Loss": f"{mean_loss:.4f}",
                "Epsilon": f"{agent.epsilon:.3f}",
            },
        )

        rolling_ready = len(reward_window) == REWARD_ROLLING_WINDOW
        if curriculum.on_episode_end(avg_reward, rolling_ready):
            game.level = curriculum.level
            game.configure_level()
            agent.reset_epsilon_for_level_up()
            log_key_values(
                LOGGER.name,
                {
                    "Event": "Level Up",
                    "Level": curriculum.level,
                },
            )

        if agent.episodes_played % EPISODE_CHECKPOINT_EVERY == 0:
            try_save_model(agent.online_model, MODEL_CHECKPOINT_PATH, "Model Saved")

        if agent.episodes_played >= BEST_MODEL_MIN_EPISODES and avg_reward > best_average_reward:
            best_average_reward = avg_reward
            try_save_model(agent.online_model, MODEL_BEST_PATH, "New Best Model")

        game.reset()

        if agent.total_env_steps >= TOTAL_TRAINING_STEPS:
            log_key_values(
                LOGGER.name,
                {
                    "Event": "Training Step Limit Reached",
                    "Steps": agent.total_env_steps,
                },
            )
            break

    game.close()


if __name__ == "__main__":
    train()
