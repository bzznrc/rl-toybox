
"""Continuous-control side-view biped walker environment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import arcade
import numpy as np
from pyglet.window import key as pyglet_key

from core.arcade_style import (
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CORAL,
    COLOR_DARK_NEUTRAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_LIGHT_NEUTRAL,
    COLOR_SLATE_GRAY,
)
from core.curriculum import (
    ThreeLevelCurriculum,
    advance_curriculum,
    build_curriculum_config,
    validate_curriculum_level_settings,
)
from core.envs.base import Env
from core.io_schema import clip_signed, clip_unit, normalized_ray_first_hit, ordered_feature_vector
from core.rewards import RewardBreakdown
from core.runtime import ArcadeFrameClock, ArcadeWindowController
from core.utils import resolve_play_level
from games.walk import config


validate_curriculum_level_settings(
    min_level=config.MIN_LEVEL,
    max_level=config.MAX_LEVEL,
    level_settings=config.LEVEL_SETTINGS,
)


@dataclass(frozen=True)
class LegKinematics:
    hip_x: float
    hip_y: float
    knee_x: float
    knee_y: float
    ankle_x: float
    ankle_y: float
    heel_x: float
    heel_y: float
    toe_x: float
    toe_y: float
    foot_x: float
    foot_y: float


class WalkEnv(Env):
    """2D side-view continuous-control walker for PPO demos."""

    INPUT_FEATURE_NAMES = tuple(config.INPUT_FEATURE_NAMES)
    ACTION_NAMES = tuple(config.ACTION_NAMES)
    OBS_DIM = int(config.OBS_DIM)
    ACT_DIM = int(config.ACT_DIM)
    REWARD_COMPONENT_ORDER = ("P", "L")

    JOINT_SPEED_NORM = 10.0

    def __init__(self, mode: str = "train", render: bool = False, level: int | None = None) -> None:
        self.mode = str(mode)
        self.show_game = bool(render)

        curriculum_config = build_curriculum_config(
            min_level=int(config.MIN_LEVEL),
            max_level=int(config.MAX_LEVEL),
            promotion_settings=config.CURRICULUM_PROMOTION,
        )
        self._curriculum = (
            ThreeLevelCurriculum(config=curriculum_config, level_settings=config.LEVEL_SETTINGS)
            if self.mode == "train"
            else None
        )
        self._current_level = (
            int(self._curriculum.get_level())
            if self._curriculum is not None
            else resolve_play_level(
                level=level,
                min_level=config.MIN_LEVEL,
                max_level=config.MAX_LEVEL,
                default_level=3,
            )
        )
        self._last_episode_level = int(self._current_level)
        self._last_episode_success = 0

        self.frame_clock = ArcadeFrameClock()
        self.window_controller = ArcadeWindowController(
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            config.WINDOW_TITLE,
            enabled=self.show_game,
            queue_input_events=False,
            vsync=False,
        )

        self._terrain_x = np.zeros((0,), dtype=np.float32)
        self._terrain_h = np.zeros((0,), dtype=np.float32)
        self._terrain_length = 0.0
        self._terrain_bump_amp_primary = 0.0
        self._terrain_bump_amp_secondary = 0.0
        self._terrain_freq_primary = 0.0
        self._terrain_freq_secondary = 0.0
        self._terrain_step_every = 24
        self._terrain_step_chance = 0.0
        self._terrain_step_height = 0.0
        self._terrain_step_clip = 0.0
        self._terrain_noise_scale = 0.0
        self._terrain_goal_distance = 0.0
        self.max_steps = 1
        self._level_entropy_coef = 0.0

        self._episode_counter = 0
        self.steps = 0
        self.done = False

        self.torso_x = float(config.START_X)
        self.torso_y = float(config.START_TORSO_CLEARANCE)
        self.torso_vx = 0.0
        self.torso_vy = 0.0
        self.torso_tilt = 0.0
        self.torso_ang_vel = 0.0

        self.left_hip_angle = float(config.START_LEFT_HIP_ANGLE)
        self.left_hip_speed = 0.0
        self.left_knee_angle = float(config.START_LEFT_KNEE_ANGLE)
        self.left_knee_speed = 0.0
        self.right_hip_angle = float(config.START_RIGHT_HIP_ANGLE)
        self.right_hip_speed = 0.0
        self.right_knee_angle = float(config.START_RIGHT_KNEE_ANGLE)
        self.right_knee_speed = 0.0

        self.left_foot_contact = False
        self.right_foot_contact = False

        self._last_obs = np.zeros((self.OBS_DIM,), dtype=np.float32)
        self._last_ray_values = np.ones((4,), dtype=np.float32)
        self._last_ray_origin = (0.0, 0.0)
        self._last_ray_dirs = [(0.0, -1.0)] * 4
        self._last_x_distance = 0.0
        self._best_x = float(config.START_X)

        self._episode_reward_components = RewardBreakdown()

        self._apply_level_settings(int(self._current_level))
        self.reset()

    def get_entropy_coef_for_level(self, level: int | None = None) -> float | None:
        if level is None or int(level) == int(self._current_level):
            return float(self._level_entropy_coef)

        settings = config.LEVEL_SETTINGS.get(int(level))
        if settings is None:
            raise ValueError(f"Unsupported level '{level}' for Walk.")
        if "entropy_coef" not in settings:
            raise ValueError("Walk LEVEL_SETTINGS entries must define 'entropy_coef'.")
        try:
            return float(settings["entropy_coef"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Walk LEVEL_SETTINGS 'entropy_coef' must be numeric.") from exc

    def _apply_level_settings(self, level: int) -> None:
        settings = config.LEVEL_SETTINGS.get(int(level))
        if settings is None:
            raise ValueError(f"Unsupported level '{level}' for Walk.")

        difficulty = float(np.clip(float(settings["terrain_difficulty"]), 0.0, 1.0))
        goal_distance = max(0.0, float(settings["goal_distance"]))
        view_world_width = float(config.SCREEN_WIDTH) / max(1e-6, float(config.WORLD_PIXELS_PER_METER))
        terrain_buffer = (
            float(config.START_X)
            + float(view_world_width)
            - float(config.CAMERA_LEAD_METERS)
            + float(config.TERRAIN_RENDER_MARGIN_METERS)
            + 1.0
        )
        self._current_level = int(level)
        self._terrain_length = max(16.0, float(goal_distance + terrain_buffer))
        self._terrain_bump_amp_primary = self._lerp(
            config.TERRAIN_BUMP_AMP_PRIMARY_FLAT,
            config.TERRAIN_BUMP_AMP_PRIMARY_RUGGED,
            difficulty,
        )
        self._terrain_bump_amp_secondary = self._lerp(
            config.TERRAIN_BUMP_AMP_SECONDARY_FLAT,
            config.TERRAIN_BUMP_AMP_SECONDARY_RUGGED,
            difficulty,
        )
        self._terrain_freq_primary = self._lerp(
            config.TERRAIN_FREQ_PRIMARY_FLAT,
            config.TERRAIN_FREQ_PRIMARY_RUGGED,
            difficulty,
        )
        self._terrain_freq_secondary = self._lerp(
            config.TERRAIN_FREQ_SECONDARY_FLAT,
            config.TERRAIN_FREQ_SECONDARY_RUGGED,
            difficulty,
        )
        self._terrain_step_every = max(
            1,
            int(
                round(
                    self._lerp(
                        float(config.TERRAIN_STEP_EVERY_FLAT),
                        float(config.TERRAIN_STEP_EVERY_RUGGED),
                        difficulty,
                    )
                )
            ),
        )
        self._terrain_step_chance = float(
            np.clip(
                self._lerp(config.TERRAIN_STEP_CHANCE_FLAT, config.TERRAIN_STEP_CHANCE_RUGGED, difficulty),
                0.0,
                1.0,
            )
        )
        self._terrain_step_height = max(
            0.0,
            self._lerp(config.TERRAIN_STEP_HEIGHT_FLAT, config.TERRAIN_STEP_HEIGHT_RUGGED, difficulty),
        )
        self._terrain_step_clip = max(
            0.0,
            self._lerp(config.TERRAIN_STEP_CLIP_FLAT, config.TERRAIN_STEP_CLIP_RUGGED, difficulty),
        )
        self._terrain_noise_scale = max(
            0.0,
            self._lerp(config.TERRAIN_NOISE_FLAT, config.TERRAIN_NOISE_RUGGED, difficulty),
        )
        self._terrain_goal_distance = float(goal_distance)
        self.max_steps = max(
            1,
            int(
                round(
                    float(config.EPISODE_STEP_BUDGET_BASE)
                    + float(self._terrain_goal_distance) * float(config.EPISODE_STEP_BUDGET_PER_METER)
                )
            ),
        )
        self._level_entropy_coef = float(settings["entropy_coef"])

    def _episode_seed(self) -> int:
        return int(config.BASE_SEED + self._episode_counter * 9_973 + self._current_level * 131)

    def _generate_terrain(self, seed: int) -> None:
        rng = np.random.default_rng(int(seed))
        dx = float(config.TERRAIN_SAMPLE_DX)
        point_count = max(16, int(self._terrain_length / dx) + 3)
        xs = (np.arange(point_count, dtype=np.float32) * dx).astype(np.float32)

        phase_primary = float(rng.uniform(0.0, 2.0 * math.pi))
        phase_secondary = float(rng.uniform(0.0, 2.0 * math.pi))
        step_offset = 0.0
        heights = np.zeros((point_count,), dtype=np.float32)

        for idx, x in enumerate(xs):
            if idx > 0 and (idx % int(self._terrain_step_every) == 0):
                if float(rng.random()) < float(self._terrain_step_chance):
                    step_offset += float(rng.uniform(-self._terrain_step_height, self._terrain_step_height))
                    step_offset = float(np.clip(step_offset, -self._terrain_step_clip, self._terrain_step_clip))

            bump_primary = self._terrain_bump_amp_primary * math.sin(self._terrain_freq_primary * float(x) + phase_primary)
            bump_secondary = self._terrain_bump_amp_secondary * math.sin(
                self._terrain_freq_secondary * float(x) + phase_secondary
            )
            noise = float(rng.normal(loc=0.0, scale=self._terrain_noise_scale))
            heights[idx] = float(step_offset + bump_primary + bump_secondary + noise)

        kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel /= float(kernel.sum())
        heights = np.convolve(heights, kernel, mode="same").astype(np.float32)
        if heights.size > 0:
            heights -= float(heights[0])

        self._terrain_x = xs
        self._terrain_h = heights

    def _terrain_height(self, x_world: float) -> float:
        if self._terrain_x.size <= 0:
            return 0.0

        x_value = float(x_world)
        xs = self._terrain_x
        hs = self._terrain_h
        if x_value <= float(xs[0]):
            return float(hs[0])
        if x_value >= float(xs[-1]):
            return float(hs[-1])

        dx = max(1e-6, float(config.TERRAIN_SAMPLE_DX))
        idx = int((x_value - float(xs[0])) / dx)
        idx = max(0, min(idx, int(xs.size) - 2))
        x0 = float(xs[idx])
        x1 = float(xs[idx + 1])
        h0 = float(hs[idx])
        h1 = float(hs[idx + 1])
        t = (x_value - x0) / max(1e-6, x1 - x0)
        return float(h0 + (h1 - h0) * t)

    def _terrain_surface_height(self, x_world: float) -> float:
        # Terrain surface line is the physical contact surface.
        return float(self._terrain_height(float(x_world)))

    def _terrain_surface_slope(self, x_world: float) -> float:
        dx = max(1e-4, float(config.TERRAIN_SAMPLE_DX))
        h_left = self._terrain_surface_height(float(x_world) - dx)
        h_right = self._terrain_surface_height(float(x_world) + dx)
        return float((h_right - h_left) / (2.0 * dx))

    def _torso_render_half_side_world(self) -> float:
        torso_side = (
            max(float(config.TORSO_WIDTH), float(config.TORSO_HEIGHT))
            * float(config.BIPED_RENDER_SCALE)
            * float(config.TORSO_RENDER_SIDE_SCALE)
        )
        return float(0.5 * torso_side)

    def _torso_surface_penetration(
        self,
        *,
        torso_x: float,
        torso_y: float,
        torso_tilt: float,
    ) -> float:
        half_side = float(self._torso_render_half_side_world())
        max_penetration = 0.0
        for local_x in (-half_side, half_side):
            for local_y in (-half_side, half_side):
                off_x, off_y = self._rotate_local(local_x, local_y, torso_tilt)
                px = float(torso_x) + float(off_x)
                py = float(torso_y) + float(off_y)
                penetration = float(self._terrain_surface_height(px) - py)
                if penetration > max_penetration:
                    max_penetration = float(penetration)
        return float(max_penetration)

    def _clamp_world_point_above_surface(
        self,
        x_world: float,
        y_world: float,
        *,
        clearance_world: float = 0.0,
    ) -> tuple[float, float]:
        min_y = float(self._terrain_surface_height(float(x_world)) + max(0.0, float(clearance_world)))
        return float(x_world), float(max(float(y_world), min_y))

    @staticmethod
    def _rotate_local(local_x: float, local_y: float, angle: float) -> tuple[float, float]:
        cos_a = math.cos(float(angle))
        sin_a = math.sin(float(angle))
        x_rot = (cos_a * float(local_x)) - (sin_a * float(local_y))
        y_rot = (sin_a * float(local_x)) + (cos_a * float(local_y))
        return float(x_rot), float(y_rot)

    @staticmethod
    def _dir_from_down(angle: float) -> tuple[float, float]:
        return float(math.sin(float(angle))), float(-math.cos(float(angle)))

    def _leg_joint_values(self, side: str) -> tuple[float, float, float, float]:
        if str(side) == "left":
            return (
                float(self.left_hip_angle),
                float(self.left_hip_speed),
                float(self.left_knee_angle),
                float(self.left_knee_speed),
            )
        return (
            float(self.right_hip_angle),
            float(self.right_hip_speed),
            float(self.right_knee_angle),
            float(self.right_knee_speed),
        )

    def _leg_kinematics_from_state(
        self,
        side: str,
        *,
        torso_x: float,
        torso_y: float,
        torso_tilt: float,
        hip_angle: float,
        knee_angle: float,
    ) -> LegKinematics:
        hip_local_x = -float(config.HIP_SPACING) * 0.5 if str(side) == "left" else float(config.HIP_SPACING) * 0.5
        hip_local_y = -float(config.TORSO_HEIGHT) * 0.45
        hip_off_x, hip_off_y = self._rotate_local(hip_local_x, hip_local_y, torso_tilt)
        hip_x = float(torso_x) + hip_off_x
        hip_y = float(torso_y) + hip_off_y

        thigh_angle = float(torso_tilt) + float(hip_angle)
        # Positive knee angle means flexion relative to the thigh hinge.
        shank_angle = thigh_angle - float(knee_angle)
        foot_angle = shank_angle + math.radians(float(config.FOOT_ANGLE_OFFSET_DEG))

        thigh_dx, thigh_dy = self._dir_from_down(thigh_angle)
        shank_dx, shank_dy = self._dir_from_down(shank_angle)
        foot_dx, foot_dy = self._dir_from_down(foot_angle)

        knee_x = hip_x + float(config.THIGH_LENGTH) * thigh_dx
        knee_y = hip_y + float(config.THIGH_LENGTH) * thigh_dy

        ankle_x = knee_x + float(config.SHIN_LENGTH) * shank_dx
        ankle_y = knee_y + float(config.SHIN_LENGTH) * shank_dy

        heel_x = ankle_x - float(config.FOOT_LENGTH) * 0.25 * foot_dx
        heel_y = ankle_y - float(config.FOOT_LENGTH) * 0.25 * foot_dy
        toe_x = ankle_x + float(config.FOOT_LENGTH) * 0.75 * foot_dx
        toe_y = ankle_y + float(config.FOOT_LENGTH) * 0.75 * foot_dy
        foot_x = (heel_x + toe_x) * 0.5
        foot_y = (heel_y + toe_y) * 0.5

        return LegKinematics(
            hip_x=float(hip_x),
            hip_y=float(hip_y),
            knee_x=float(knee_x),
            knee_y=float(knee_y),
            ankle_x=float(ankle_x),
            ankle_y=float(ankle_y),
            heel_x=float(heel_x),
            heel_y=float(heel_y),
            toe_x=float(toe_x),
            toe_y=float(toe_y),
            foot_x=float(foot_x),
            foot_y=float(foot_y),
        )

    def _leg_kinematics(self, side: str) -> LegKinematics:
        hip_angle, _hip_speed, knee_angle, _knee_speed = self._leg_joint_values(side)
        return self._leg_kinematics_from_state(
            side,
            torso_x=float(self.torso_x),
            torso_y=float(self.torso_y),
            torso_tilt=float(self.torso_tilt),
            hip_angle=float(hip_angle),
            knee_angle=float(knee_angle),
        )

    def _foot_velocity(self, side: str, pose: LegKinematics) -> tuple[float, float]:
        eps = 1e-3
        hip_angle, hip_speed, knee_angle, knee_speed = self._leg_joint_values(side)
        pred_pose = self._leg_kinematics_from_state(
            side,
            torso_x=float(self.torso_x + self.torso_vx * eps),
            torso_y=float(self.torso_y + self.torso_vy * eps),
            torso_tilt=float(self.torso_tilt + self.torso_ang_vel * eps),
            hip_angle=float(hip_angle + hip_speed * eps),
            knee_angle=float(knee_angle + knee_speed * eps),
        )
        vx = (float(pred_pose.foot_x) - float(pose.foot_x)) / eps
        vy = (float(pred_pose.foot_y) - float(pose.foot_y)) / eps
        return float(vx), float(vy)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(float(low), min(float(high), float(value))))

    @staticmethod
    def _lerp(low: float, high: float, t: float) -> float:
        return float(float(low) + (float(high) - float(low)) * float(t))

    @staticmethod
    def _normalize_interval(value: float, low: float, high: float) -> float:
        mid = 0.5 * (float(low) + float(high))
        half = max(1e-6, 0.5 * (float(high) - float(low)))
        return float(clip_signed((float(value) - mid) / half))

    def _integrate_joint(
        self,
        angle: float,
        speed: float,
        torque: float,
        *,
        accel_scale: float,
        angle_min: float,
        angle_max: float,
    ) -> tuple[float, float]:
        dt = float(config.PHYSICS_DT)
        next_speed = float(speed) + float(torque) * float(accel_scale) * dt
        damping = max(0.0, 1.0 - float(config.JOINT_DAMPING) * dt)
        next_speed *= float(damping)
        next_angle = float(angle) + next_speed * dt
        next_angle = self._clamp(next_angle, float(angle_min), float(angle_max))
        if next_angle <= float(angle_min) and next_speed < 0.0:
            next_speed = 0.0
        if next_angle >= float(angle_max) and next_speed > 0.0:
            next_speed = 0.0
        return float(next_angle), float(next_speed)

    def _apply_joint_dynamics(self, action_vec: np.ndarray) -> None:
        torques = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        if torques.size != int(self.ACT_DIM):
            torques = np.zeros((int(self.ACT_DIM),), dtype=np.float32)

        self.left_hip_angle, self.left_hip_speed = self._integrate_joint(
            self.left_hip_angle,
            self.left_hip_speed,
            float(torques[0]),
            accel_scale=float(config.HIP_TORQUE_ACCEL),
            angle_min=float(config.HIP_ANGLE_MIN),
            angle_max=float(config.HIP_ANGLE_MAX),
        )
        self.left_knee_angle, self.left_knee_speed = self._integrate_joint(
            self.left_knee_angle,
            self.left_knee_speed,
            float(torques[1]),
            accel_scale=float(config.KNEE_TORQUE_ACCEL),
            angle_min=float(config.KNEE_ANGLE_MIN),
            angle_max=float(config.KNEE_ANGLE_MAX),
        )
        self.right_hip_angle, self.right_hip_speed = self._integrate_joint(
            self.right_hip_angle,
            self.right_hip_speed,
            float(torques[2]),
            accel_scale=float(config.HIP_TORQUE_ACCEL),
            angle_min=float(config.HIP_ANGLE_MIN),
            angle_max=float(config.HIP_ANGLE_MAX),
        )
        self.right_knee_angle, self.right_knee_speed = self._integrate_joint(
            self.right_knee_angle,
            self.right_knee_speed,
            float(torques[3]),
            accel_scale=float(config.KNEE_TORQUE_ACCEL),
            angle_min=float(config.KNEE_ANGLE_MIN),
            angle_max=float(config.KNEE_ANGLE_MAX),
        )

    def _contact_for_foot(self, pose: LegKinematics, foot_vel_x: float, foot_vel_y: float) -> tuple[bool, float, float, float]:
        ground_y = self._terrain_surface_height(float(pose.foot_x))
        penetration = float(ground_y - float(pose.foot_y))

        in_contact = bool(penetration >= -float(config.CONTACT_SLOP))
        normal_force = 0.0
        friction_force = 0.0
        if in_contact:
            effective_pen = max(0.0, penetration)
            normal_force = (
                float(config.CONTACT_SPRING) * effective_pen
                + float(config.CONTACT_DAMP) * max(0.0, -float(foot_vel_y))
            )
            normal_force = self._clamp(normal_force, 0.0, float(config.MAX_NORMAL_FORCE))
            if normal_force > 0.0:
                raw_friction = -float(config.FRICTION_STIFFNESS) * float(foot_vel_x)
                friction_cap = min(float(config.MAX_FRICTION_FORCE), float(config.FRICTION_COEF) * normal_force)
                friction_force = self._clamp(raw_friction, -friction_cap, friction_cap)

        return bool(in_contact), float(normal_force), float(friction_force), float(penetration)

    def _simulate_step(self, action_vec: np.ndarray) -> None:
        self._apply_joint_dynamics(action_vec)

        left_pose = self._leg_kinematics("left")
        right_pose = self._leg_kinematics("right")
        left_foot_vx, left_foot_vy = self._foot_velocity("left", left_pose)
        right_foot_vx, right_foot_vy = self._foot_velocity("right", right_pose)

        force_x = -float(config.AIR_DRAG) * float(self.torso_vx)
        force_y = 0.0
        torque_z = 0.0

        left_contact, left_normal, left_friction, left_pen = self._contact_for_foot(left_pose, left_foot_vx, left_foot_vy)
        right_contact, right_normal, right_friction, right_pen = self._contact_for_foot(
            right_pose,
            right_foot_vx,
            right_foot_vy,
        )

        self.left_foot_contact = bool(left_contact)
        self.right_foot_contact = bool(right_contact)

        if left_contact:
            force_x += left_friction
            force_y += left_normal
            rel_x = float(left_pose.foot_x - self.torso_x)
            rel_y = float(left_pose.foot_y - self.torso_y)
            torque_z += (rel_x * left_normal) - (rel_y * left_friction)
        if right_contact:
            force_x += right_friction
            force_y += right_normal
            rel_x = float(right_pose.foot_x - self.torso_x)
            rel_y = float(right_pose.foot_y - self.torso_y)
            torque_z += (rel_x * right_normal) - (rel_y * right_friction)

        max_penetration = max(0.0, float(left_pen), float(right_pen))
        if max_penetration > 0.0:
            self.torso_y += min(0.03, max_penetration * 0.45)
            if self.torso_vy < 0.0:
                self.torso_vy *= 0.2

        dt = float(config.PHYSICS_DT)
        accel_x = force_x / max(1e-6, float(config.TORSO_MASS))
        accel_y = float(config.GRAVITY) + (force_y / max(1e-6, float(config.TORSO_MASS)))
        ang_accel = (
            float(torque_z)
            - float(config.TORSO_ANGULAR_DAMP) * float(self.torso_ang_vel)
            - float(config.TORSO_UPRIGHT_SPRING) * float(self.torso_tilt)
        ) / max(1e-6, float(config.TORSO_INERTIA))

        self.torso_vx = self._clamp(
            self.torso_vx + accel_x * dt,
            -float(config.MAX_TORSO_SPEED_X),
            float(config.MAX_TORSO_SPEED_X),
        )
        self.torso_vy = self._clamp(
            self.torso_vy + accel_y * dt,
            -float(config.MAX_TORSO_SPEED_Y),
            float(config.MAX_TORSO_SPEED_Y),
        )
        self.torso_ang_vel = self._clamp(
            self.torso_ang_vel + ang_accel * dt,
            -float(config.MAX_TORSO_ANG_SPEED),
            float(config.MAX_TORSO_ANG_SPEED),
        )

        self.torso_x = max(0.0, float(self.torso_x + self.torso_vx * dt))
        self.torso_y = float(self.torso_y + self.torso_vy * dt)
        self.torso_tilt = self._clamp(
            self.torso_tilt + self.torso_ang_vel * dt,
            -1.6,
            1.6,
        )

        # Enforce non-penetration against the visible terrain surface line.
        post_left_pose = self._leg_kinematics("left")
        post_right_pose = self._leg_kinematics("right")
        post_penetration = max(
            0.0,
            float(self._terrain_surface_height(post_left_pose.foot_x) - post_left_pose.foot_y),
            float(self._terrain_surface_height(post_right_pose.foot_x) - post_right_pose.foot_y),
        )
        if post_penetration > 0.0:
            self.torso_y += float(post_penetration)
            if self.torso_vy < 0.0:
                self.torso_vy = 0.0

        body_penetration = float(
            self._torso_surface_penetration(
                torso_x=float(self.torso_x),
                torso_y=float(self.torso_y),
                torso_tilt=float(self.torso_tilt),
            )
        )
        if body_penetration > 0.0:
            self.torso_y += float(body_penetration)
            if self.torso_vy < 0.0:
                self.torso_vy = 0.0

    def _ray_origin_world(self) -> tuple[float, float]:
        off_x, off_y = self._rotate_local(
            float(config.RAY_ORIGIN_LOCAL_X),
            float(config.RAY_ORIGIN_LOCAL_Y),
            float(self.torso_tilt),
        )
        return float(self.torso_x + off_x), float(self.torso_y + off_y)

    def _compute_rays(self) -> np.ndarray:
        origin_x, origin_y = self._ray_origin_world()
        ray_values: list[float] = []
        ray_dirs: list[tuple[float, float]] = []

        for angle_deg in config.RAY_ANGLES_DEG:
            angle_rad = math.radians(float(angle_deg))
            dir_x = float(math.sin(angle_rad))
            dir_y = float(-math.cos(angle_rad))
            ray_dirs.append((dir_x, dir_y))
            value = normalized_ray_first_hit(
                origin_x=float(origin_x),
                origin_y=float(origin_y),
                dir_x=dir_x,
                dir_y=dir_y,
                max_distance=float(config.RAY_MAX_DISTANCE),
                is_blocked=lambda px, py: bool(float(py) <= self._terrain_surface_height(float(px))),
                step_size=float(config.RAY_STEP_SIZE),
                start_offset=0.0,
            )
            ray_values.append(float(clip_unit(value)))

        rays = np.asarray(ray_values, dtype=np.float32)
        self._last_ray_values = rays
        self._last_ray_origin = (float(origin_x), float(origin_y))
        self._last_ray_dirs = list(ray_dirs)
        return rays

    def _obs(self) -> np.ndarray:
        rays = self._compute_rays()

        feature_values = {
            "self_torso_tilt": float(clip_signed(self.torso_tilt / max(1e-6, float(config.TERMINAL_TILT_RAD)))),
            "self_torso_ang_vel": float(
                clip_signed(self.torso_ang_vel / max(1e-6, float(config.MAX_TORSO_ANG_SPEED)))
            ),
            "self_vx": float(clip_signed(self.torso_vx / max(1e-6, float(config.MAX_TORSO_SPEED_X)))),
            "self_vy": float(clip_signed(self.torso_vy / max(1e-6, float(config.MAX_TORSO_SPEED_Y)))),
            "self_left_hip_angle": float(
                self._normalize_interval(self.left_hip_angle, config.HIP_ANGLE_MIN, config.HIP_ANGLE_MAX)
            ),
            "self_left_hip_speed": float(clip_signed(self.left_hip_speed / float(self.JOINT_SPEED_NORM))),
            "self_left_knee_angle": float(
                self._normalize_interval(self.left_knee_angle, config.KNEE_ANGLE_MIN, config.KNEE_ANGLE_MAX)
            ),
            "self_left_knee_speed": float(clip_signed(self.left_knee_speed / float(self.JOINT_SPEED_NORM))),
            "self_right_hip_angle": float(
                self._normalize_interval(self.right_hip_angle, config.HIP_ANGLE_MIN, config.HIP_ANGLE_MAX)
            ),
            "self_right_hip_speed": float(clip_signed(self.right_hip_speed / float(self.JOINT_SPEED_NORM))),
            "self_right_knee_angle": float(
                self._normalize_interval(self.right_knee_angle, config.KNEE_ANGLE_MIN, config.KNEE_ANGLE_MAX)
            ),
            "self_right_knee_speed": float(clip_signed(self.right_knee_speed / float(self.JOINT_SPEED_NORM))),
            "self_left_foot_contact": 1.0 if self.left_foot_contact else 0.0,
            "self_right_foot_contact": 1.0 if self.right_foot_contact else 0.0,
            "ray_ground_10": float(rays[0]),
            "ray_ground_25": float(rays[1]),
            "ray_ground_40": float(rays[2]),
            "ray_ground_55": float(rays[3]),
        }

        obs = np.asarray(ordered_feature_vector(self.INPUT_FEATURE_NAMES, feature_values), dtype=np.float32)
        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Walk observation expected {self.OBS_DIM} features, got {obs.shape[0]}")
        return obs

    def _reset_robot_state(self) -> None:
        self.torso_x = float(config.START_X)
        self.torso_y = float(config.START_TORSO_CLEARANCE)
        self.torso_vx = 0.0
        self.torso_vy = 0.0
        self.torso_tilt = 0.0
        self.torso_ang_vel = 0.0

        self.left_hip_angle = float(config.START_LEFT_HIP_ANGLE)
        self.left_hip_speed = 0.0
        self.left_knee_angle = float(config.START_LEFT_KNEE_ANGLE)
        self.left_knee_speed = 0.0
        self.right_hip_angle = float(config.START_RIGHT_HIP_ANGLE)
        self.right_hip_speed = 0.0
        self.right_knee_angle = float(config.START_RIGHT_KNEE_ANGLE)
        self.right_knee_speed = 0.0

        self.left_foot_contact = False
        self.right_foot_contact = False

        for _ in range(12):
            left_pose = self._leg_kinematics("left")
            right_pose = self._leg_kinematics("right")
            left_pen = self._terrain_surface_height(left_pose.foot_x) - left_pose.foot_y
            right_pen = self._terrain_surface_height(right_pose.foot_x) - right_pose.foot_y
            body_pen = self._torso_surface_penetration(
                torso_x=float(self.torso_x),
                torso_y=float(self.torso_y),
                torso_tilt=float(self.torso_tilt),
            )
            correction = max(0.0, float(left_pen), float(right_pen), float(body_pen))
            if correction <= 1e-6:
                break
            self.torso_y += float(correction + 0.002)

    def _parse_action(self, action: object) -> np.ndarray:
        if self.mode == "human":
            return self._human_action_from_keyboard()

        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if int(action_array.size) != int(self.ACT_DIM):
            return np.zeros((int(self.ACT_DIM),), dtype=np.float32)
        return np.clip(action_array, -1.0, 1.0).astype(np.float32, copy=False)

    def _human_action_from_keyboard(self) -> np.ndarray:
        torques = np.zeros((int(self.ACT_DIM),), dtype=np.float32)
        if self.window_controller.window is None:
            return torques

        if self.window_controller.is_key_down(pyglet_key.A):
            torques[0] -= 1.0
        if self.window_controller.is_key_down(pyglet_key.D):
            torques[0] += 1.0
        if self.window_controller.is_key_down(pyglet_key.W):
            torques[1] += 1.0
        if self.window_controller.is_key_down(pyglet_key.S):
            torques[1] -= 1.0

        if self.window_controller.is_key_down(pyglet_key.J):
            torques[2] -= 1.0
        if self.window_controller.is_key_down(pyglet_key.L):
            torques[2] += 1.0
        if self.window_controller.is_key_down(pyglet_key.I):
            torques[3] += 1.0
        if self.window_controller.is_key_down(pyglet_key.K):
            torques[3] -= 1.0
        return np.clip(torques, -1.0, 1.0).astype(np.float32, copy=False)

    def reset(self) -> np.ndarray:
        self._apply_level_settings(int(self._current_level))
        seed = self._episode_seed()
        self._episode_counter += 1
        self._generate_terrain(seed)

        self.steps = 0
        self.done = False
        self._episode_reward_components.reset()
        self._reset_robot_state()
        self._best_x = float(self.torso_x)

        self._last_obs = self._obs()
        self._last_x_distance = float(self.torso_x - float(config.START_X))
        return np.asarray(self._last_obs, dtype=np.float32)

    def _is_fallen(self) -> bool:
        if abs(float(self.torso_tilt)) > float(config.TERMINAL_TILT_RAD):
            return True
        ground_here = self._terrain_surface_height(float(self.torso_x))
        return bool(float(self.torso_y) <= float(ground_here + float(config.TERMINAL_TORSO_CLEARANCE)))

    def step(self, action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.done:
            return (
                np.asarray(self._last_obs, dtype=np.float32),
                0.0,
                True,
                {
                    "win": bool(self._last_episode_success > 0),
                    "success": int(self._last_episode_success),
                    "level": int(self._last_episode_level),
                    "x_distance": float(self._last_x_distance),
                    "reward_components": self._episode_reward_components.totals(),
                },
            )

        self.window_controller.poll_events_or_raise()

        episode_level = int(self._current_level)
        action_vec = self._parse_action(action)
        best_x_prev = float(self._best_x)

        self._simulate_step(action_vec)
        self.steps += 1

        best_x_now = max(float(self._best_x), float(self.torso_x))
        progress_reward = max(0.0, float(best_x_now - best_x_prev))
        self._best_x = float(best_x_now)
        x_distance = float(self.torso_x - float(config.START_X))
        fell = bool(self._is_fallen())
        reached_goal = bool(x_distance >= float(self._terrain_goal_distance))
        timed_out = bool(self.steps >= int(self.max_steps))
        done = bool(fell or reached_goal or timed_out)
        success = int(done and (not fell) and reached_goal)

        reward = 0.0
        reward_breakdown: dict[str, float] = {}

        if self.mode != "human":
            if float(progress_reward) > 0.0:
                reward += float(progress_reward)
                reward_breakdown["P"] = float(progress_reward)
                self._episode_reward_components.add("P", float(progress_reward))

            if done and success <= 0:
                reward += float(config.PENALTY_LOSE)
                reward_breakdown["L"] = float(config.PENALTY_LOSE)
                self._episode_reward_components.add("L", float(config.PENALTY_LOSE))

        self.done = bool(done)
        self._last_obs = self._obs()
        self._last_x_distance = float(x_distance)

        level_changed = False
        if self.done:
            self._last_episode_level = int(episode_level)
            self._last_episode_success = int(success)
            self._current_level, level_changed = advance_curriculum(
                self._curriculum,
                success=int(success),
                current_level=int(self._current_level),
                apply_level=self._apply_level_settings,
            )

        self.render()
        self.frame_clock.tick(config.FPS if self.show_game else config.TRAINING_FPS)

        info: dict[str, object] = {
            "win": bool(success > 0) if self.done else False,
            "success": int(success) if self.done else 0,
            "level": int(episode_level),
            "level_changed": bool(level_changed),
            "x_distance": float(x_distance),
            "fell": bool(fell),
            "reward_breakdown": reward_breakdown if self.mode != "human" else {},
        }
        if self.done:
            info["reward_components"] = self._episode_reward_components.totals()

        return np.asarray(self._last_obs, dtype=np.float32), float(reward), bool(self.done), info

    def _camera_left_world(self) -> float:
        return float(self.torso_x - float(config.CAMERA_LEAD_METERS))

    def _world_to_screen_x(self, x_world: float) -> float:
        camera_left = self._camera_left_world()
        return float((float(x_world) - camera_left) * float(config.WORLD_PIXELS_PER_METER))

    def _world_to_screen_y(self, y_world: float) -> float:
        return float(config.GROUND_BASE_Y_PX + float(y_world) * float(config.WORLD_PIXELS_PER_METER))

    def _draw_oriented_box(
        self,
        *,
        center_x: float,
        center_y: float,
        half_w: float,
        half_h: float,
        angle: float,
        color: tuple[int, int, int],
    ) -> None:
        corners_local = (
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        )
        points: list[tuple[float, float]] = []
        for lx, ly in corners_local:
            rx, ry = self._rotate_local(float(lx), float(ly), float(angle))
            px = self._world_to_screen_x(float(center_x + rx))
            py = self._world_to_screen_y(float(center_y + ry))
            points.append((px, py))
        arcade.draw_polygon_filled(points, color)

    def _draw_terrain(self) -> None:
        camera_left = self._camera_left_world() - float(config.TERRAIN_RENDER_MARGIN_METERS)
        world_width = float(config.SCREEN_WIDTH) / float(config.WORLD_PIXELS_PER_METER)
        camera_right = camera_left + world_width + (2.0 * float(config.TERRAIN_RENDER_MARGIN_METERS))
        x_cursor = camera_left
        points: list[tuple[float, float]] = []
        while x_cursor <= camera_right:
            sx = self._world_to_screen_x(x_cursor)
            sy = self._world_to_screen_y(self._terrain_surface_height(x_cursor))
            points.append((sx, sy))
            x_cursor += float(config.TERRAIN_SAMPLE_DX)
        if not points:
            return

        terrain_poly = [(0.0, float(config.BB_HEIGHT)), *points, (float(config.SCREEN_WIDTH), float(config.BB_HEIGHT))]
        arcade.draw_polygon_filled(terrain_poly, COLOR_DARK_NEUTRAL)

        for idx in range(len(points) - 1):
            x1, y1 = points[idx]
            x2, y2 = points[idx + 1]
            arcade.draw_line(x1, y1, x2, y2, COLOR_LIGHT_NEUTRAL, 3.0)

    def _draw_distance_markers(self) -> None:
        spacing = max(1e-6, float(config.DISTANCE_MARKER_SPACING_METERS))
        major_every = max(1, int(config.DISTANCE_MARKER_MAJOR_EVERY))
        start_x = float(config.START_X)
        inner_size = float(config.DISTANCE_MARKER_INNER_SIZE_PX)
        inset = max(0.0, float(config.DISTANCE_MARKER_OUTLINE_INSET_PX))
        outer_size = float(inner_size + 2.0 * inset)

        visible_left = self._camera_left_world()
        visible_right = visible_left + (float(config.SCREEN_WIDTH) / max(1e-6, float(config.WORLD_PIXELS_PER_METER)))
        marker_index_start = max(0, int(math.floor((visible_left - start_x) / spacing)))
        if (start_x + marker_index_start * spacing) < visible_left:
            marker_index_start += 1
        marker_index_end = int(math.ceil((visible_right - start_x) / spacing))

        for marker_index in range(marker_index_start, marker_index_end + 1):
            marker_x = float(start_x + marker_index * spacing)
            surface_sy = self._world_to_screen_y(self._terrain_surface_height(marker_x))
            center_y = 0.5 * (float(config.BB_HEIGHT) + float(surface_sy))
            center_x = self._world_to_screen_x(marker_x)

            if marker_index % major_every == 0:
                arcade.draw_lbwh_rectangle_filled(
                    float(center_x - outer_size * 0.5),
                    float(center_y - outer_size * 0.5),
                    float(outer_size),
                    float(outer_size),
                    COLOR_FOG_GRAY,
                )
            arcade.draw_lbwh_rectangle_filled(
                float(center_x - inner_size * 0.5),
                float(center_y - inner_size * 0.5),
                float(inner_size),
                float(inner_size),
                COLOR_SLATE_GRAY,
            )

    def _draw_walker(self) -> None:
        left_pose = self._leg_kinematics("left")
        right_pose = self._leg_kinematics("right")

        left_color_outer = COLOR_AQUA
        left_color_inner = COLOR_DEEP_TEAL
        right_color_outer = COLOR_CORAL
        right_color_inner = COLOR_BRICK_RED

        torso_side = (
            max(float(config.TORSO_WIDTH), float(config.TORSO_HEIGHT))
            * float(config.BIPED_RENDER_SCALE)
            * float(config.TORSO_RENDER_SIDE_SCALE)
        )
        torso_half_side = 0.5 * float(torso_side)
        torso_inset_world = float(config.TORSO_BORDER_INSET_PX) / max(1e-6, float(config.WORLD_PIXELS_PER_METER))
        torso_inner_half_side = max(0.02, float(torso_half_side) - float(torso_inset_world))
        self._draw_oriented_box(
            center_x=self.torso_x,
            center_y=self.torso_y,
            half_w=float(torso_half_side),
            half_h=float(torso_half_side),
            angle=float(self.torso_tilt),
            color=COLOR_AQUA,
        )
        self._draw_oriented_box(
            center_x=self.torso_x,
            center_y=self.torso_y,
            half_w=float(torso_inner_half_side),
            half_h=float(torso_inner_half_side),
            angle=float(self.torso_tilt),
            color=COLOR_DEEP_TEAL,
        )

        leg_width_outer = float(max(2.0, float(config.LEG_OUTER_WIDTH_PX)))
        leg_width_inner = float(max(1.0, min(leg_width_outer - 1.0, float(config.LEG_INNER_WIDTH_PX))))
        foot_width_outer = float(max(2.0, float(config.FOOT_OUTER_WIDTH_PX)))
        foot_width_inner = float(max(1.0, min(foot_width_outer - 1.0, float(config.FOOT_INNER_WIDTH_PX))))
        meters_per_pixel = 1.0 / max(1e-6, float(config.WORLD_PIXELS_PER_METER))
        leg_clearance_world = 0.5 * leg_width_outer * meters_per_pixel
        foot_clearance_world = 0.5 * foot_width_outer * meters_per_pixel

        def draw_leg(pose: LegKinematics, outer: tuple[int, int, int], inner: tuple[int, int, int], contact: bool) -> None:
            hip_wx, hip_wy = float(pose.hip_x), float(pose.hip_y)
            knee_wx, knee_wy = float(pose.knee_x), float(pose.knee_y)
            ankle_wx, ankle_wy = float(pose.ankle_x), float(pose.ankle_y)
            heel_wx, heel_wy = float(pose.heel_x), float(pose.heel_y)
            toe_wx, toe_wy = float(pose.toe_x), float(pose.toe_y)
            foot_wx, foot_wy = float(pose.foot_x), float(pose.foot_y)

            if contact:
                slope = self._terrain_surface_slope(ankle_wx)
                tan_x, tan_y = 1.0, float(slope)
                tan_len = max(1e-6, float(math.hypot(tan_x, tan_y)))
                tan_x /= tan_len
                tan_y /= tan_len
                heel_wx = float(ankle_wx - float(config.FOOT_LENGTH) * 0.25 * tan_x)
                heel_wy = float(ankle_wy - float(config.FOOT_LENGTH) * 0.25 * tan_y)
                toe_wx = float(ankle_wx + float(config.FOOT_LENGTH) * 0.75 * tan_x)
                toe_wy = float(ankle_wy + float(config.FOOT_LENGTH) * 0.75 * tan_y)
                foot_wx = float((heel_wx + toe_wx) * 0.5)
                foot_wy = float((heel_wy + toe_wy) * 0.5)

            hip_wx, hip_wy = self._clamp_world_point_above_surface(
                hip_wx,
                hip_wy,
                clearance_world=leg_clearance_world,
            )
            knee_wx, knee_wy = self._clamp_world_point_above_surface(
                knee_wx,
                knee_wy,
                clearance_world=leg_clearance_world,
            )
            ankle_wx, ankle_wy = self._clamp_world_point_above_surface(
                ankle_wx,
                ankle_wy,
                clearance_world=leg_clearance_world,
            )
            heel_wx, heel_wy = self._clamp_world_point_above_surface(
                heel_wx,
                heel_wy,
                clearance_world=foot_clearance_world,
            )
            toe_wx, toe_wy = self._clamp_world_point_above_surface(
                toe_wx,
                toe_wy,
                clearance_world=foot_clearance_world,
            )
            foot_wx, foot_wy = self._clamp_world_point_above_surface(
                foot_wx,
                foot_wy,
                clearance_world=foot_clearance_world,
            )

            hip_x, hip_y = self._world_to_screen_x(hip_wx), self._world_to_screen_y(hip_wy)
            knee_x, knee_y = self._world_to_screen_x(knee_wx), self._world_to_screen_y(knee_wy)
            ankle_x, ankle_y = self._world_to_screen_x(ankle_wx), self._world_to_screen_y(ankle_wy)
            heel_x, heel_y = self._world_to_screen_x(heel_wx), self._world_to_screen_y(heel_wy)
            toe_x, toe_y = self._world_to_screen_x(toe_wx), self._world_to_screen_y(toe_wy)
            foot_x, foot_y = self._world_to_screen_x(foot_wx), self._world_to_screen_y(foot_wy)

            arcade.draw_line(hip_x, hip_y, knee_x, knee_y, outer, leg_width_outer)
            arcade.draw_line(knee_x, knee_y, ankle_x, ankle_y, outer, leg_width_outer)
            arcade.draw_line(hip_x, hip_y, knee_x, knee_y, inner, leg_width_inner)
            arcade.draw_line(knee_x, knee_y, ankle_x, ankle_y, inner, leg_width_inner)

            arcade.draw_line(heel_x, heel_y, toe_x, toe_y, outer, foot_width_outer)
            arcade.draw_line(heel_x, heel_y, toe_x, toe_y, inner, foot_width_inner)
            if contact:
                arcade.draw_circle_filled(foot_x, foot_y, max(3.5, foot_width_inner * 0.75), COLOR_FOG_GRAY)

        draw_leg(left_pose, left_color_outer, left_color_inner, self.left_foot_contact)
        draw_leg(right_pose, right_color_outer, right_color_inner, self.right_foot_contact)

    def _draw_rays(self) -> None:
        if not bool(config.DRAW_RAYS):
            return
        origin_x, origin_y = self._last_ray_origin
        sx0 = self._world_to_screen_x(origin_x)
        sy0 = self._world_to_screen_y(origin_y)

        for idx, (dir_x, dir_y) in enumerate(self._last_ray_dirs):
            value = float(self._last_ray_values[idx])
            distance = float(value) * float(config.RAY_MAX_DISTANCE)
            hit_x = float(origin_x + dir_x * distance)
            hit_y = float(origin_y + dir_y * distance)
            sx1 = self._world_to_screen_x(hit_x)
            sy1 = self._world_to_screen_y(hit_y)
            color = COLOR_FOG_GRAY if value >= 1.0 else COLOR_AQUA
            arcade.draw_line(sx0, sy0, sx1, sy1, color, 1.0)

    def _draw_hud(self) -> None:
        arcade.draw_lbwh_rectangle_filled(0, 0, config.SCREEN_WIDTH, config.BB_HEIGHT, COLOR_DARK_NEUTRAL)
        distance_walked_m = float(self.torso_x - float(config.START_X))
        status = "RUN"
        if self.done:
            status = "WIN" if self._last_episode_success > 0 else "FAIL"

        text_main = (
            f"Lv:{int(self._current_level)}  "
            f"Step:{int(self.steps):>4}/{int(self.max_steps):<4}  "
            f"Dist:{distance_walked_m:>6.2f}m  "
            f"Tilt:{math.degrees(self.torso_tilt):>6.1f}  "
            f"Contact:{int(self.left_foot_contact)}|{int(self.right_foot_contact)}  "
            f"{status}"
        )
        arcade.draw_text(
            text_main,
            8,
            max(2.0, float(config.BB_HEIGHT) * 0.5 - 6.0),
            COLOR_LIGHT_NEUTRAL,
            font_size=max(10.0, float(config.BB_HEIGHT) * 0.42),
            anchor_x="left",
            anchor_y="center",
        )

    def render(self) -> None:
        if self.window_controller.window is None:
            return

        self.window_controller.clear(COLOR_DARK_NEUTRAL)
        arcade.draw_lbwh_rectangle_filled(
            0,
            float(config.BB_HEIGHT),
            float(config.SCREEN_WIDTH),
            float(config.SCREEN_HEIGHT - config.BB_HEIGHT),
            COLOR_SLATE_GRAY,
        )
        self._draw_terrain()
        self._draw_distance_markers()
        self._draw_walker()
        self._draw_hud()
        self.window_controller.flip()

    def close(self) -> None:
        self.window_controller.close()
