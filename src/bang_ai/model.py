"""Bang DQN model and trainer."""

from __future__ import annotations

import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from bang_ai.config import (
    GAMMA,
    GRAD_CLIP_NORM,
    HIDDEN_DIMENSIONS,
    LEARNING_RATE,
    MODEL_SAVE_RETRIES,
    MODEL_SAVE_RETRY_DELAY_SECONDS,
    NUM_ACTIONS,
    NUM_INPUT_FEATURES,
    USE_GPU,
    WEIGHT_DECAY,
)
from bang_ai.logging_utils import format_display_path, get_torch_device


device = get_torch_device(prefer_gpu=USE_GPU)
INCOMPATIBLE_CHECKPOINT_MESSAGE = (
    "Incompatible model checkpoint for current network architecture "
    "(input/output size or hidden dimensions mismatch)."
)


class DuelingQNetwork(nn.Module):
    """Compact Q-network with separate value and advantage heads."""

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_size
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_features, hidden), nn.GELU()])
            in_features = hidden

        self.feature_extractor = nn.Sequential(*layers)
        self.value_head = nn.Linear(in_features, 1)
        self.advantage_head = nn.Linear(in_features, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature_extractor(x)
        values = self.value_head(features)
        advantages = self.advantage_head(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)

    def copy(self):
        return copy.deepcopy(self)

    def save(self, file_name: str) -> None:
        directory = os.path.dirname(file_name) or "."
        os.makedirs(directory, exist_ok=True)
        temp_file = f"{file_name}.tmp.{os.getpid()}"
        last_error = None

        for attempt in range(MODEL_SAVE_RETRIES):
            try:
                torch.save(self.state_dict(), temp_file)
                os.replace(temp_file, file_name)
                return
            except (OSError, RuntimeError) as error:
                last_error = error
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass
                if attempt < MODEL_SAVE_RETRIES - 1:
                    delay = MODEL_SAVE_RETRY_DELAY_SECONDS * (attempt + 1)
                    time.sleep(delay)

        raise RuntimeError(
            f"Failed to save model to '{file_name}' after {MODEL_SAVE_RETRIES} attempts."
        ) from last_error

    def load(self, file_name: str) -> None:
        self.load_state_dict(torch.load(file_name, map_location=device))


def build_q_network() -> DuelingQNetwork:
    return DuelingQNetwork(NUM_INPUT_FEATURES, HIDDEN_DIMENSIONS, NUM_ACTIONS).to(device)


def build_loaded_q_network(load_path: str | None = None, strict: bool = False) -> tuple[DuelingQNetwork, str | None]:
    model = build_q_network()
    loaded_path = None
    if load_path:
        if os.path.exists(load_path):
            try:
                model.load(load_path)
            except RuntimeError:
                if strict:
                    print(INCOMPATIBLE_CHECKPOINT_MESSAGE)
                    raise
                print(
                    f"WARNING: {INCOMPATIBLE_CHECKPOINT_MESSAGE} "
                    f"Ignoring '{format_display_path(load_path)}'."
                )
            else:
                loaded_path = load_path
        elif strict:
            raise FileNotFoundError(format_display_path(load_path))
    return model, loaded_path


class DQNTrainer:
    """Trains the online network with Double-DQN targets."""

    def __init__(self, online_model: DuelingQNetwork, target_model: DuelingQNetwork):
        self.online_model = online_model
        self.target_model = target_model
        self.optimizer = optim.AdamW(
            online_model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    def train_step(self, state, action, reward, next_state, done, is_weights=None):
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
        action = torch.as_tensor(action, dtype=torch.long, device=device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.bool, device=device)
        if is_weights is not None:
            is_weights = torch.as_tensor(is_weights, dtype=torch.float32, device=device)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)
            if is_weights is not None:
                is_weights = is_weights.unsqueeze(0)

        action_indices = action.argmax(dim=1)
        current_q = self.online_model(state).gather(1, action_indices.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_model(next_state).argmax(dim=1)
            next_q = self.target_model(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = reward + (~done).float() * GAMMA * next_q

        td_errors = targets - current_q
        per_sample_loss = self.loss_fn(current_q, targets)
        if is_weights is not None:
            per_sample_loss = per_sample_loss * is_weights
        loss = per_sample_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()
        return float(loss.item()), td_errors.detach().abs().cpu().tolist()
