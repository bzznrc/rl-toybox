"""Central configuration for Vroom."""

from __future__ import annotations


INPUT_FEATURE_NAMES = [
    # SELF (8)
    "self_lat_offset",
    "self_lat_offset_delta",
    "self_fwd_speed",
    "self_fwd_speed_delta",
    "self_heading_sin",
    "self_heading_cos",
    "self_in_contact",
    "self_last_action",
    # RAYS (4)
    "ray_fwd_near",
    "ray_fwd_far",
    "ray_fwd_left",
    "ray_fwd_right",
    # TGT (4)
    "tgt_dx",
    "tgt_dy",
    "tgt_dvx",
    "tgt_dvy",
    # TRACK (4)
    "trk_lookahead_sin",
    "trk_lookahead_cos",
    "trk_lookahead_dist",
    "trk_curvature_ahead",
]

ACTION_NAMES = [
    "coast",
    "throttle",
    "left_coast",
    "right_coast",
    "left_throttle",
    "right_throttle",
]

OBS_DIM = len(INPUT_FEATURE_NAMES)
ACT_DIM = len(ACTION_NAMES)

# Model and training
HIDDEN_DIMENSIONS = [48, 48]
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 200_000
TARGET_SYNC_EVERY = 2_000
GRAD_CLIP_NORM = 10.0
LEARN_START_STEPS = 20_000
TRAIN_EVERY_STEPS = 1
UPDATES_PER_TRAIN = 1
MAX_TRAINING_STEPS = 2_000_000
CHECKPOINT_EVERY_STEPS = 100_000
EPSILON_DECAY = 0.9999975036

# Reward shaping
REWARD_WIN = 10.0
PENALTY_LOSE = -5.0
PROGRESS_SCALE = 1.0
PROGRESS_CLIP = 0.2
PENALTY_COLLISION = -1.0
PENALTY_STEP = -0.01

REWARD_COMPONENTS = {
    "outcome.reward_win": REWARD_WIN,
    "outcome.penalty_lose": PENALTY_LOSE,
    "progress.scale": PROGRESS_SCALE,
    "progress.clip": PROGRESS_CLIP,
    "event.penalty_collision": PENALTY_COLLISION,
    "step.penalty_step": PENALTY_STEP,
}
