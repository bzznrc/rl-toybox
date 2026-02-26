"""Snake RL model and trainer."""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_ai.config import MODEL_SAVE_RETRIES, MODEL_SAVE_RETRY_DELAY_SECONDS


def _resolve_model_path(path_value: str) -> Path:
    return Path(path_value)


INCOMPATIBLE_CHECKPOINT_MESSAGE = (
    "ERROR: Incompatible model checkpoint for current network architecture. "
    "HIDDEN_DIMENSIONS and checkpoint must match."
)


class LinearQNet(nn.Module):
    """Feedforward Neural Network with customizable hidden layers."""

    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def save(self, file_path_value: str) -> None:
        file_path = _resolve_model_path(file_path_value)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = file_path.with_name(f"{file_path.name}.tmp.{os.getpid()}")
        last_error = None

        for attempt in range(MODEL_SAVE_RETRIES):
            try:
                torch.save(self.state_dict(), temp_file)
                os.replace(temp_file, file_path)
                return
            except (OSError, RuntimeError) as error:
                last_error = error
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass
                if attempt < MODEL_SAVE_RETRIES - 1:
                    delay = MODEL_SAVE_RETRY_DELAY_SECONDS * (attempt + 1)
                    time.sleep(delay)

        raise RuntimeError(
            f"Failed to save model to '{file_path}' after {MODEL_SAVE_RETRIES} attempts."
        ) from last_error

    def load(self, file_path_value: str) -> None:
        file_path = _resolve_model_path(file_path_value)
        if file_path.exists():
            try:
                self.load_state_dict(torch.load(file_path))
                self.eval()
                print(f"Model loaded from {file_path}")
            except RuntimeError:
                print(INCOMPATIBLE_CHECKPOINT_MESSAGE)
                raise SystemExit(1)
        else:
            print(f"No model found at {file_path}. Starting from scratch.")


class QTrainer:
    """Trainer class for the Q-learning model."""

    def __init__(self, model: LinearQNet, lr: float, gamma: float):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done) -> float:
        state = torch.from_numpy(np.asarray(state, dtype=np.float32))
        next_state = torch.from_numpy(np.asarray(next_state, dtype=np.float32))
        action = torch.from_numpy(np.asarray(action, dtype=np.float32))
        reward = torch.from_numpy(np.asarray(reward, dtype=np.float32))
        done = torch.from_numpy(np.asarray(done, dtype=np.bool_))

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
