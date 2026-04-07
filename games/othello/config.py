"""Scaffold config for Othello, the search/self-play capstone."""

OBS_DIM = 64
ACT_DIM = 65
ACTION_NAMES = ("board_move_masked",)

POLICY_VALUE_HIDDEN_DIMENSIONS = (128, 128)
SIMULATIONS_PER_MOVE = 64
CPUCT = 1.25
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
