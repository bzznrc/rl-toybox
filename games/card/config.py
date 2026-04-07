"""Scaffold config for Card, the stochastic hidden-info actor-critic game."""

OBS_DIM = 14
ACT_DIM = 4
ACTION_NAMES = ("fold", "check", "bet_small", "bet_big")

HIDDEN_DIMENSIONS = (64, 64)
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 128
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
USE_GPU = True

MIN_LEVEL = 1
MAX_TRAINING_ITERATIONS = 50
ROLLOUT_STEPS = 256
CHECKPOINT_EVERY_ITERATIONS = 10
REWARD_ROLLING_WINDOW = 20
MIN_EPISODES_FOR_STATS = 10
LEVEL_SETTINGS = {
    1: {"entropy_coef": 0.02},
    2: {"entropy_coef": 0.02},
    3: {"entropy_coef": 0.02},
}
