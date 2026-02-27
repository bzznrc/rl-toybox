"""Central IO configuration for Vroom."""

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
