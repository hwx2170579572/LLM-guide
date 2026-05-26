from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
try:
    import gymnasium as gym
except Exception:
    gym = None
try:
    import gym as legacy_gym
except Exception:
    legacy_gym = None
import highway_env  # noqa: F401  # registers highway-env ids
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


ALL_MODES = ["goal_sac_pure", "goal_csac_lagrangian", "paper_sac_pure", "paper_csac_lagrangian"]
PENALTY_IMPLS = ["legacy_fixed_penalty", "stage4_constrained_recurrent_v1"]
UPGRADE_STAGES = ["none", "m3m6", "m1m2", "m4m5", "final"]
ACTOR_DECOUPLE_MODES = ["weighted", "alternating_stopgrad"]
LOG_STD_MIN = -20
LOG_STD_MAX = 2


@dataclass
class EnvConfig:
    HIGHWAY_ENV_ID: str = field(default="highway-v0", init=False, repr=False)
    LOCKED_LANES_COUNT: int = field(default=4, init=False, repr=False)
    LOCKED_DURATION: int = field(default=20, init=False, repr=False)
    LOCKED_SIMULATION_FREQUENCY: int = field(default=20, init=False, repr=False)
    LOCKED_POLICY_FREQUENCY: int = field(default=20, init=False, repr=False)
    TRAFFIC_PROFILES: Tuple[str, ...] = field(default=("sparse", "dense"), init=False, repr=False)

    env_id: str = "highway-v0"
    lanes_count: int = 4
    # The concrete vehicle count is sampled on every reset from the selected traffic profile.
    vehicles_count: int = 15
    duration: int = 20
    simulation_frequency: int = 20
    policy_frequency: int = 20
    lane_width: float = 4.0
    vehicles_obs_count: int = 8
    max_speed: float = 30.0
    min_speed: float = 0.0
    speed_limit: float = 30.0
    offscreen_rendering: bool = True
    normalize_obs: bool = True
    absolute_obs: bool = True
    order: str = "sorted"

    # Scenario randomization profile.
    traffic_density: str = "sparse"
    sparse_vehicles_min: int = 10
    sparse_vehicles_max: int = 15
    dense_vehicles_min: int = 20
    dense_vehicles_max: int = 25
    ego_speed_min: float = 20.0
    ego_speed_max: float = 25.0
    vehicle_speed_min: float = 20.0
    vehicle_speed_max: float = 25.0
    ego_init_x_min: float = 0.0
    ego_init_x_max: float = 100.0
    npc_spawn_rear: float = 120.0
    npc_spawn_front: float = 520.0
    npc_min_gap_same_lane: float = 18.0
    npc_min_gap_other_lane: float = 10.0

    # Goal definition: right-most lane, 500 m ahead of the randomized ego start.
    goal_distance: float = 500.0
    goal_lane_id: int = 3
    goal_lane_tolerance: float = 0.45

    # Ego-centered rendering window. highway-env's renderer follows the controlled
    # vehicle; centering at [0.5, 0.5] makes the saved frame symmetric around ego.
    screen_width: int = 900
    screen_height: int = 360
    centering_position_x: float = 0.50
    centering_position_y: float = 0.50
    render_scaling: float = 4.0
    show_trajectories: bool = False

    def traffic_vehicle_range(self) -> Tuple[int, int]:
        profile = str(self.traffic_density).lower().strip()
        if profile == "dense":
            return int(self.dense_vehicles_min), int(self.dense_vehicles_max)
        if profile == "sparse":
            return int(self.sparse_vehicles_min), int(self.sparse_vehicles_max)
        raise ValueError(f"Unsupported traffic_density='{self.traffic_density}'. Use sparse or dense.")

    def assert_highway_lock(self) -> None:
        if str(self.env_id).strip() != self.HIGHWAY_ENV_ID:
            raise ValueError(
                f"Highway-only run expected env_id='{self.HIGHWAY_ENV_ID}', got '{self.env_id}'."
            )
        if int(self.lanes_count) != int(self.LOCKED_LANES_COUNT):
            raise ValueError(f"Highway-only run requires lanes_count={self.LOCKED_LANES_COUNT}.")
        if int(self.duration) != int(self.LOCKED_DURATION):
            raise ValueError(f"Highway-only run requires duration={self.LOCKED_DURATION}s.")
        if int(self.simulation_frequency) != int(self.LOCKED_SIMULATION_FREQUENCY):
            raise ValueError(
                f"Highway-only run requires simulation_frequency={self.LOCKED_SIMULATION_FREQUENCY}Hz."
            )
        if int(self.policy_frequency) != int(self.LOCKED_POLICY_FREQUENCY):
            raise ValueError(
                f"Highway-only run requires policy_frequency={self.LOCKED_POLICY_FREQUENCY}Hz."
            )
        if abs(float(self.speed_limit) - 30.0) > 1e-6:
            raise ValueError("This scenario requires speed_limit=30.0 m/s.")
        if not bool(self.absolute_obs):
            raise ValueError("This scenario requires absolute_obs=True.")
        profile = str(self.traffic_density).lower().strip()
        if profile not in self.TRAFFIC_PROFILES:
            raise ValueError(f"traffic_density must be one of {self.TRAFFIC_PROFILES}.")
        low, high = self.traffic_vehicle_range()
        if low < 0 or high < low:
            raise ValueError(f"Invalid traffic vehicle range: {low}~{high}.")
        if int(self.goal_lane_id) != int(self.lanes_count - 1):
            raise ValueError("The goal must be placed on the right-most lane.")

    def to_gym_config(self) -> Dict[str, Any]:
        self.assert_highway_lock()
        return {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": self.vehicles_obs_count,
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                "normalize": self.normalize_obs,
                "absolute": self.absolute_obs,
                "order": self.order,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
            },
            "lanes_count": self.lanes_count,
            "vehicles_count": self.vehicles_count,
            "controlled_vehicles": 1,
            "duration": self.duration,
            "simulation_frequency": self.simulation_frequency,
            "policy_frequency": self.policy_frequency,
            "speed_limit": self.speed_limit,
            "offscreen_rendering": self.offscreen_rendering,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "centering_position": [self.centering_position_x, self.centering_position_y],
            "scaling": self.render_scaling,
            "show_trajectories": self.show_trajectories,
            "action_masking": False,
            "ego_spacing": 2,
            "offroad_terminal": False
        }


@dataclass
class RewardConfig:
    # reward_type is selected by --mode: goal_shaped or paper_formula.
    reward_type: str = "goal_shaped"
    mission_success_reward: float = 10.0
    mission_failure_penalty: float = -10.0
    goal_progress_weight: float = 1.00
    goal_lane_weight: float = 0.30
    paper_heading_epsilon: float = 0.03
    paper_lane_change_reward: float = 0.4
    paper_wrong_lane_change_penalty: float = -0.5
    paper_speed_reward_scale: float = 0.3
    paper_speed_min: float = 20.0
    paper_ttc_threshold: float = 3.0
    paper_ttc_eps: float = 0.1
    use_env_reward: bool = True
    env_reward_scale: float = 0.15
    progress_weight: float = 1.15
    lane_center_weight: float = 0.18
    heading_weight: float = 0.08
    lateral_speed_weight: float = 0.05
    action_weight: float = 0.02
    smooth_weight: float = 0.15
    lane_change_weight: float = 0.35
    max_lane_center_penalty: float = 2.0
    max_heading_penalty: float = 1.0
    max_action_penalty: float = 2.0
    max_smooth_penalty: float = 2.0


@dataclass
class CostConfig:
    # Clear constraint expressions use distance headway, time headway, TTC, boundary and speed terms.
    time_headway_sec: float = 1.2
    min_headway: float = 12.0
    headway_proactive_distance: float = 19.2
    headway_closing_speed: float = 1.5
    ttc_safe: float = 2.5
    speed_limit: float = 30.0
    overspeed_soft_ratio: float = 0.92
    action_acc_scale: float = 5.0
    action_steer_scale: float = 0.7853981633974483
    acc_limit: float = 2.5
    steer_limit: float = 0.20
    action_delta_limit: float = 1.8
    lane_center_soft_ratio: float = 0.12
    lane_center_hard_ratio: float = 0.45
    road_boundary_soft_margin: float = 1.60
    road_boundary_hard_margin: float = 0.70
    road_boundary_off_margin: float = 0.05
    collision_front_distance: float = 10.0
    collision_front_ttc: float = 4.0
    rear_gap_distance: float = 8.0
    rear_ttc: float = 3.0
    side_gap_front: float = 10.0
    side_gap_rear: float = 8.0
    collision_weight: float = 1.0
    headway_weight: float = 1.3
    overspeed_weight: float = 0.6
    comfort_weight: float = 0.10
    lane_weight: float = 1.15


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 2e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dim: int = 256
    batch_size: int = 128
    buffer_size: int = 100000
    start_steps: int = 800
    update_after: int = 256
    update_every: int = 1
    target_entropy: float = -2.0
    critic_grad_clip_norm: float = 10.0
    actor_grad_clip_norm: float = 10.0
    penalty_collision: float = 1.00
    penalty_headway: float = 0.80
    penalty_overspeed: float = 0.55
    penalty_comfort: float = 0.10
    penalty_lane: float = 0.70
    penalty_impl: str = "stage4_constrained_recurrent_v1"
    upgrade_stage: str = "m1m2"
    experiment_round: int = 1
    enable_priority_safety_replay: bool = True
    enable_lagrangian_safety: bool = True
    enable_recurrent_encoder: bool = True
    enable_decoupled_actor: bool = True
    enable_tail_risk_cvar: bool = False
    actor_decouple_mode: str = "weighted"
    safety_w_collision: float = 1.0
    safety_w_headway: float = 1.0
    # Round-1 initial values only; tune with multi-seed scans (not theoretical constants).
    safety_budget_collision: float = 0.02
    safety_budget_headway: float = 0.08
    # Backward-compatible shared budget; when set from CLI it maps to both constraints.
    safety_budget: float = 0.05
    lambda_init: float = 1.0
    lambda_lr: float = 1e-3
    lambda_max: float = 50.0
    constraint_budget_safety: float = 0.06
    constraint_budget_boundary: float = 0.04
    constraint_budget_speed: float = 0.04
    constraint_lambda_lr: float = 1e-3
    constraint_lambda_max: float = 50.0
    actor_update_interval: int = 2
    alpha_min: float = 0.05
    seq_len: int = 8
    gru_hidden_dim: int = 128
    safety_n_step: int = 12
    danger_ratio: float = 0.60
    near_danger_ratio: float = 0.20
    danger_precollision_steps: int = 10
    cvar_alpha: float = 0.20
    cvar_quantiles: int = 16


@dataclass
class TrainConfig:
    seed: int = 42
    episodes: int = 10
    max_steps_per_episode: int = 400
    device: str = "cuda:0"
    mode: str = "goal_csac_lagrangian"
    render: bool = False
    print_every_step: bool = False


@dataclass
class EvalConfig:
    enabled: bool = True
    episodes: int = 3
    seed_offset: int = 1000
    render: bool = False
    print_step: bool = False
    save_video: bool = False
    video_dir: str = ""


@dataclass
class DiagnosticsConfig:
    export_train_log: bool = False
    train_log_path: str = ""
    save_train_json: bool = False
    train_json_path: str = ""
    run_result_json_path: str = ""
    tensorboard_log_dir: str = ""
    tensorboard_flush_secs: int = 10


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


def _jsonable_plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_plain(v) for v in value]
    return str(value)


def get_config() -> Config:
    cfg = Config()
    cfg.train.mode = str(cfg.train.mode).lower()
    cfg.sac.penalty_impl = str(cfg.sac.penalty_impl).lower()
    cfg.sac.upgrade_stage = str(cfg.sac.upgrade_stage).lower()
    cfg.sac.actor_decouple_mode = str(cfg.sac.actor_decouple_mode).lower()
    cfg.reward.reward_type = str(cfg.reward.reward_type).lower()
    return cfg


def validate_highway_only_config(cfg: Config) -> None:
    cfg.env.assert_highway_lock()


def configure_mode_behavior(cfg: Config, mode: Optional[str] = None) -> None:
    """Map the four experiment modes to reward type and agent family.

    Modes:
      goal_sac_pure:            goal-shaped reward + vanilla SAC
      goal_csac_lagrangian:     goal-shaped reward + Lagrangian constrained SAC
      paper_sac_pure:           paper-style R_ms+R_lc+R_e+R_s reward + vanilla SAC
      paper_csac_lagrangian:    paper-style reward + Lagrangian constrained SAC
    """
    selected = str(mode if mode is not None else cfg.train.mode).lower().strip()
    if selected not in ALL_MODES:
        raise ValueError(f"Unsupported mode: {selected}. Choose from {ALL_MODES}.")
    cfg.train.mode = selected
    cfg.reward.reward_type = "paper_formula" if selected.startswith("paper_") else "goal_shaped"
    # The new constrained modes use the compact Lagrangian CMDP agent below; legacy stage guards are disabled.
    cfg.sac.penalty_impl = "legacy_fixed_penalty"
    cfg.sac.upgrade_stage = "none"
    cfg.sac.enable_priority_safety_replay = False
    cfg.sac.enable_lagrangian_safety = selected.endswith("csac_lagrangian")
    cfg.sac.enable_recurrent_encoder = False
    cfg.sac.enable_decoupled_actor = False
    cfg.sac.enable_tail_risk_cvar = False


def _normalize_stage_controls(cfg: Config) -> None:
    cfg.sac.penalty_impl = str(cfg.sac.penalty_impl).lower().strip() or "stage4_constrained_recurrent_v1"
    if cfg.sac.penalty_impl not in PENALTY_IMPLS:
        cfg.sac.penalty_impl = "stage4_constrained_recurrent_v1"
    cfg.sac.upgrade_stage = str(cfg.sac.upgrade_stage).lower().strip() or "m1m2"
    if cfg.sac.upgrade_stage not in UPGRADE_STAGES:
        cfg.sac.upgrade_stage = "m1m2"
    cfg.sac.actor_decouple_mode = str(cfg.sac.actor_decouple_mode).lower().strip() or "weighted"
    if cfg.sac.actor_decouple_mode not in ACTOR_DECOUPLE_MODES:
        cfg.sac.actor_decouple_mode = "weighted"


def _apply_stage_template(cfg: Config, stage: str) -> None:
    stage = str(stage).lower()
    cfg.sac.enable_priority_safety_replay = False
    cfg.sac.enable_lagrangian_safety = False
    cfg.sac.enable_recurrent_encoder = False
    cfg.sac.enable_decoupled_actor = False
    cfg.sac.enable_tail_risk_cvar = False
    cfg.sac.actor_update_interval = max(1, int(cfg.sac.actor_update_interval))
    cfg.sac.alpha_min = max(0.0, float(cfg.sac.alpha_min))
    if stage == "m3m6":
        cfg.sac.enable_priority_safety_replay = True
        cfg.sac.actor_update_interval = max(2, int(cfg.sac.actor_update_interval))
        cfg.sac.alpha_min = max(float(cfg.sac.alpha_min), 0.03)
    elif stage == "m1m2":
        cfg.sac.enable_priority_safety_replay = True
        cfg.sac.enable_lagrangian_safety = True
        cfg.sac.enable_recurrent_encoder = False
        cfg.sac.enable_decoupled_actor = False
        cfg.sac.enable_tail_risk_cvar = False
        cfg.sac.actor_update_interval = max(2, int(cfg.sac.actor_update_interval))
        cfg.sac.alpha_min = max(float(cfg.sac.alpha_min), 0.03)
    elif stage == "m4m5":
        cfg.sac.enable_priority_safety_replay = True
        cfg.sac.enable_lagrangian_safety = True
        cfg.sac.enable_recurrent_encoder = True
        cfg.sac.enable_decoupled_actor = True
        cfg.sac.actor_update_interval = max(2, int(cfg.sac.actor_update_interval))
        cfg.sac.alpha_min = max(float(cfg.sac.alpha_min), 0.03)
    elif stage == "final":
        cfg.sac.enable_priority_safety_replay = True
        cfg.sac.enable_lagrangian_safety = True
        cfg.sac.enable_recurrent_encoder = True
        cfg.sac.enable_decoupled_actor = True
        cfg.sac.enable_tail_risk_cvar = True
        cfg.sac.actor_update_interval = max(2, int(cfg.sac.actor_update_interval))
        cfg.sac.alpha_min = max(float(cfg.sac.alpha_min), 0.03)


def _apply_round1_guard(cfg: Config) -> List[str]:
    notes: List[str] = ["round1_lock_active"]
    if str(cfg.train.mode) != "penalty_sac_pure":
        return notes
    if int(getattr(cfg.sac, "experiment_round", 1)) != 1:
        return notes

    if str(cfg.sac.penalty_impl).lower() != "stage4_constrained_recurrent_v1":
        cfg.sac.penalty_impl = "stage4_constrained_recurrent_v1"
        notes.append("penalty_impl->stage4_constrained_recurrent_v1")
    if str(cfg.sac.upgrade_stage).lower() != "m1m2":
        cfg.sac.upgrade_stage = "m1m2"
        notes.append("upgrade_stage->m1m2")

    prev_priority = bool(cfg.sac.enable_priority_safety_replay)
    prev_lagrangian = bool(cfg.sac.enable_lagrangian_safety)
    prev_recurrent = bool(cfg.sac.enable_recurrent_encoder)
    prev_decoupled = bool(cfg.sac.enable_decoupled_actor)
    prev_cvar = bool(cfg.sac.enable_tail_risk_cvar)
    _apply_stage_template(cfg, "m1m2")
    if not prev_priority and bool(cfg.sac.enable_priority_safety_replay):
        notes.append("priority_safety_replay->True")
    if not prev_lagrangian and bool(cfg.sac.enable_lagrangian_safety):
        notes.append("lagrangian_safety->True")
    if prev_recurrent and not bool(cfg.sac.enable_recurrent_encoder):
        notes.append("recurrent_encoder->False")
    if prev_decoupled and not bool(cfg.sac.enable_decoupled_actor):
        notes.append("decoupled_actor->False")
    if prev_cvar and not bool(cfg.sac.enable_tail_risk_cvar):
        notes.append("tail_risk_cvar->False")

    clipped_n_step = int(np.clip(int(cfg.sac.safety_n_step), 10, 20))
    if clipped_n_step != int(cfg.sac.safety_n_step):
        notes.append(f"safety_n_step->{clipped_n_step}")
        cfg.sac.safety_n_step = clipped_n_step
    return notes


def _apply_round2_guard(cfg: Config) -> List[str]:
    notes: List[str] = ["round2_long_horizon_lock_active"]
    if str(cfg.train.mode) != "penalty_sac_pure":
        return notes
    if int(getattr(cfg.sac, "experiment_round", 1)) != 2:
        return notes

    if str(cfg.sac.penalty_impl).lower() != "stage4_constrained_recurrent_v1":
        cfg.sac.penalty_impl = "stage4_constrained_recurrent_v1"
        notes.append("penalty_impl->stage4_constrained_recurrent_v1")
    if str(cfg.sac.upgrade_stage).lower() != "m1m2":
        cfg.sac.upgrade_stage = "m1m2"
        notes.append("upgrade_stage->m1m2")

    _apply_stage_template(cfg, "m1m2")
    if not bool(cfg.sac.enable_recurrent_encoder):
        cfg.sac.enable_recurrent_encoder = True
        notes.append("recurrent_encoder->True")
    if bool(cfg.sac.enable_decoupled_actor):
        cfg.sac.enable_decoupled_actor = False
        notes.append("decoupled_actor->False")
    if bool(cfg.sac.enable_tail_risk_cvar):
        cfg.sac.enable_tail_risk_cvar = False
        notes.append("tail_risk_cvar->False")

    seq_len_raw = int(cfg.sac.seq_len)
    if 16 <= seq_len_raw <= 24:
        seq_len_target = seq_len_raw
    elif seq_len_raw <= 0 or seq_len_raw == 8:
        seq_len_target = 20
    else:
        seq_len_target = int(np.clip(seq_len_raw, 16, 24))
    if seq_len_target != seq_len_raw:
        cfg.sac.seq_len = seq_len_target
        notes.append(f"seq_len->{seq_len_target}")
    return notes


def _effective_feature_flags(cfg: Config) -> Dict[str, Any]:
    return {
        "mode": str(cfg.train.mode),
        "reward_type": str(cfg.reward.reward_type),
        "traffic_density": str(cfg.env.traffic_density),
        "absolute_obs": bool(cfg.env.absolute_obs),
        "goal_distance": float(cfg.env.goal_distance),
        "goal_lane_id": int(cfg.env.goal_lane_id),
        "duration": int(cfg.env.duration),
        "simulation_frequency": int(cfg.env.simulation_frequency),
        "policy_frequency": int(cfg.env.policy_frequency),
        "penalty_impl": str(cfg.sac.penalty_impl),
        "upgrade_stage": str(cfg.sac.upgrade_stage),
        "experiment_round": int(cfg.sac.experiment_round),
        "enable_priority_safety_replay": bool(cfg.sac.enable_priority_safety_replay),
        "enable_lagrangian_safety": bool(cfg.sac.enable_lagrangian_safety),
        "enable_recurrent_encoder": bool(cfg.sac.enable_recurrent_encoder),
        "enable_decoupled_actor": bool(cfg.sac.enable_decoupled_actor),
        "enable_tail_risk_cvar": bool(cfg.sac.enable_tail_risk_cvar),
        "actor_decouple_mode": str(cfg.sac.actor_decouple_mode),
        "seq_len": int(cfg.sac.seq_len),
        "safety_n_step": int(cfg.sac.safety_n_step),
        "safety_budget_collision": float(cfg.sac.safety_budget_collision),
        "safety_budget_headway": float(cfg.sac.safety_budget_headway),
        "danger_ratio": float(cfg.sac.danger_ratio),
        "near_danger_ratio": float(cfg.sac.near_danger_ratio),
        "actor_update_interval": int(cfg.sac.actor_update_interval),
    }


def resolve_runtime_device(requested_device: str) -> str:
    requested = str(requested_device or "cpu").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[DEVICE][WARN] CUDA requested but unavailable; using CPU.")
        return "cpu"
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            print(f"[DEVICE][WARN] {requested} requested but CUDA is unavailable; using CPU.")
            return "cpu"
        try:
            index = int(requested.split(":", 1)[1])
        except Exception:
            print(f"[DEVICE][WARN] Invalid device '{requested_device}'; using CPU.")
            return "cpu"
        if index < torch.cuda.device_count():
            return requested
        print(f"[DEVICE][WARN] {requested} requested but only {torch.cuda.device_count()} CUDA device(s) found; using CPU.")
        return "cpu"
    try:
        torch.device(requested)
        return requested
    except Exception:
        print(f"[DEVICE][WARN] Invalid device '{requested_device}'; using CPU.")
        return "cpu"


def describe_runtime_device(device_str: str) -> str:
    dev = torch.device(device_str)
    if dev.type == "cuda" and torch.cuda.is_available():
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        return f"{device_str} ({torch.cuda.get_device_name(idx)})"
    return str(dev)


def configure_torch_runtime(device_str: str) -> None:
    if torch.device(device_str).type == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _wrap_to_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _squared_barrier_cost(value: float, threshold: float) -> float:
    threshold = max(float(threshold), 1e-6)
    gap = max(0.0, threshold - float(value))
    return float((gap / threshold) ** 2)


def _upper_quadratic_cost(value: float, soft_threshold: float, hard_threshold: float) -> float:
    value = float(value)
    soft_threshold = float(soft_threshold)
    hard_threshold = max(float(hard_threshold), soft_threshold + 1e-6)
    if value <= soft_threshold:
        return 0.0
    ratio = (value - soft_threshold) / max(hard_threshold - soft_threshold, 1e-6)
    return _clip01(ratio ** 2)


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_min(values: List[float]) -> float:
    return float(np.min(values)) if values else 0.0


def _safe_max(values: List[float]) -> float:
    return float(np.max(values)) if values else 0.0


class HighwayEnvWrapper:
    def __init__(self, cfg: Config, render: bool = False, record_video: bool = False):
        self.cfg = cfg
        validate_highway_only_config(self.cfg)
        self.render_enabled = bool(render)
        self.record_video = bool(record_video)
        render_mode = "rgb_array" if self.record_video else ("human" if self.render_enabled else None)
        self.api_backend = "unknown"
        self.legacy_api = False
        self.env = self._make_env(
            env_id=cfg.env.env_id,
            env_config=cfg.env.to_gym_config(),
            render_mode=render_mode,
        )
        self.last_obs = None
        self.last_action = np.zeros(self.action_dim(), dtype=np.float32)
        self.prev_scene: Optional[dict] = None
        self.recent_lane_change_countdown = 0
        self.action_low, self.action_high = self._read_action_bounds()
        self.runtime_env_snapshot: Dict[str, Any] = {}
        self.current_traffic_count = int(cfg.env.vehicles_count)
        self.current_goal_x = 0.0
        self.current_goal_y = float(cfg.env.goal_lane_id * cfg.env.lane_width)
        self.current_goal_lane_id = int(cfg.env.goal_lane_id)
        self._episode_rng = np.random.default_rng(int(cfg.train.seed))

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        episode_seed = int(seed) if seed is not None else int(np.random.randint(0, 2**31 - 1))
        self._episode_rng = np.random.default_rng(episode_seed)
        self._apply_episode_traffic_count(self._episode_rng)

        obs = None
        info: Dict[str, Any] = {}
        if not self.legacy_api:
            reset_out = self.env.reset(seed=episode_seed)
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, info = reset_out
            else:
                obs = reset_out
        else:
            try:
                reset_out = self.env.reset(is_training=False, testing_seeds=episode_seed)
            except TypeError:
                reset_out = self.env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, info = reset_out
            else:
                obs = reset_out

        self._randomize_episode_initial_state(self._episode_rng)
        obs = self._observe_current_state(fallback_obs=obs)
        self.last_obs = obs
        self.last_action = np.zeros(self.action_dim(), dtype=np.float32)
        self.prev_scene = None
        self.recent_lane_change_countdown = 0
        state = self.build_low_state(obs)
        scene = dict(self.prev_scene) if isinstance(self.prev_scene, dict) else self.get_scene_dict(advance_state=False)
        self.runtime_env_snapshot = self._capture_runtime_env_snapshot(scene)
        info = dict(info or {})
        info.update(
            {
                "traffic_density": str(self.cfg.env.traffic_density),
                "traffic_vehicles_count": int(self.current_traffic_count),
                "goal_x": float(self.current_goal_x),
                "goal_y": float(self.current_goal_y),
                "goal_lane_id": int(self.current_goal_lane_id),
                "goal_distance": float(self.cfg.env.goal_distance),
                "absolute_observation": bool(self.cfg.env.absolute_obs),
            }
        )
        return state, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        env_action = self.clip_action(action)
        if self.legacy_api:
            try:
                step_out = self.env.step(env_action, self.env.unwrapped)
            except TypeError:
                step_out = self.env.step(env_action, self.env)
            if not isinstance(step_out, tuple) or len(step_out) != 4:
                raise RuntimeError(f"Legacy gym step output not understood: {type(step_out)}")
            next_obs, env_reward, done, info = step_out
            terminated = bool(done)
            truncated = False
        else:
            step_out = self.env.step(env_action)
            if not isinstance(step_out, tuple):
                raise RuntimeError(f"Gymnasium step output not understood: {type(step_out)}")
            if len(step_out) == 5:
                next_obs, env_reward, terminated, truncated, info = step_out
            elif len(step_out) == 4:
                next_obs, env_reward, done, info = step_out
                terminated = bool(done)
                truncated = False
            else:
                raise RuntimeError(f"Gymnasium step output length not supported: {len(step_out)}")
        self.last_obs = next_obs
        self.last_action = env_action.copy()
        return self.build_low_state(next_obs), float(env_reward), bool(terminated), bool(truncated), info

    def _apply_episode_traffic_count(self, rng: np.random.Generator) -> None:
        low, high = self.cfg.env.traffic_vehicle_range()
        count = int(rng.integers(int(low), int(high) + 1))
        self.current_traffic_count = count
        self.cfg.env.vehicles_count = count
        try:
            self.env.unwrapped.config["vehicles_count"] = count
        except Exception:
            pass

    def _observe_current_state(self, fallback_obs=None):
        base_env = self.env.unwrapped
        for expr in (
            lambda: base_env.observation_type.observe(),
            lambda: base_env._observation_type.observe(),
        ):
            try:
                obs = expr()
                if obs is not None:
                    return obs
            except Exception:
                continue
        return fallback_obs

    def _apply_runtime_speed_limit(self) -> None:
        base_env = self.env.unwrapped
        road = getattr(base_env, "road", None)
        network = getattr(road, "network", None)
        limit = float(self.cfg.env.speed_limit)
        graph = getattr(network, "graph", None)
        if isinstance(graph, dict):
            for _src, dsts in graph.items():
                if not isinstance(dsts, dict):
                    continue
                for _dst, lanes in dsts.items():
                    if not isinstance(lanes, (list, tuple)):
                        continue
                    for lane in lanes:
                        try:
                            setattr(lane, "speed_limit", limit)
                        except Exception:
                            pass

    def _lane_position_heading(self, vehicle, lane_id: int, longitudinal: float) -> Tuple[np.ndarray, float]:
        lane_id = int(np.clip(int(lane_id), 0, int(self.cfg.env.lanes_count) - 1))
        lane = self._runtime_lane(vehicle, lane_id=lane_id)
        if lane is not None:
            try:
                pos = np.asarray(lane.position(float(longitudinal), 0.0), dtype=np.float32)
            except Exception:
                pos = np.asarray([float(longitudinal), float(lane_id) * float(self.cfg.env.lane_width)], dtype=np.float32)
            try:
                heading = float(lane.heading_at(float(longitudinal)))
            except Exception:
                heading = 0.0
            return pos, heading
        return np.asarray([float(longitudinal), float(lane_id) * float(self.cfg.env.lane_width)], dtype=np.float32), 0.0

    def _set_vehicle_state(self, vehicle, lane_id: int, longitudinal: float, speed: float) -> None:
        lane_id = int(np.clip(int(lane_id), 0, int(self.cfg.env.lanes_count) - 1))
        pos, heading = self._lane_position_heading(vehicle, lane_id=lane_id, longitudinal=float(longitudinal))
        try:
            vehicle.position = pos.astype(np.float32)
        except Exception:
            pass
        try:
            vehicle.heading = float(heading)
        except Exception:
            pass
        for attr in ("speed", "target_speed"):
            try:
                setattr(vehicle, attr, float(speed))
            except Exception:
                pass
        lane_index = getattr(vehicle, "lane_index", None)
        if isinstance(lane_index, tuple) and len(lane_index) >= 3:
            new_index = (lane_index[0], lane_index[1], int(lane_id))
            for attr in ("lane_index", "target_lane_index"):
                try:
                    setattr(vehicle, attr, new_index)
                except Exception:
                    pass

    def _sample_npc_longitudinal(self, rng: np.random.Generator, lane_id: int, used_by_lane: Dict[int, List[float]], ego_x: float) -> float:
        lane_id = int(lane_id)
        for _ in range(200):
            x = float(ego_x + rng.uniform(-float(self.cfg.env.npc_spawn_rear), float(self.cfg.env.npc_spawn_front)))
            min_gap = float(self.cfg.env.npc_min_gap_same_lane if lane_id in used_by_lane else self.cfg.env.npc_min_gap_other_lane)
            if all(abs(x - float(prev_x)) >= min_gap for prev_x in used_by_lane.get(lane_id, [])):
                return x
        # Deterministic fallback keeps the episode valid even in dense profiles.
        existing = sorted(float(v) for v in used_by_lane.get(lane_id, []))
        if not existing:
            return float(ego_x + self.cfg.env.npc_min_gap_same_lane)
        return float(existing[-1] + max(float(self.cfg.env.npc_min_gap_same_lane), 1.0))

    def _randomize_episode_initial_state(self, rng: np.random.Generator) -> None:
        base_env = self.env.unwrapped
        self._apply_runtime_speed_limit()
        ego = getattr(base_env, "vehicle", None)
        road = getattr(base_env, "road", None)
        if ego is None or road is None:
            return

        ego_lane = int(rng.integers(0, int(self.cfg.env.lanes_count)))
        ego_x = float(rng.uniform(float(self.cfg.env.ego_init_x_min), float(self.cfg.env.ego_init_x_max)))
        ego_speed = float(rng.uniform(float(self.cfg.env.ego_speed_min), float(self.cfg.env.ego_speed_max)))
        self._set_vehicle_state(ego, lane_id=ego_lane, longitudinal=ego_x, speed=ego_speed)

        self.current_goal_lane_id = int(self.cfg.env.goal_lane_id)
        self.current_goal_x = float(ego_x + float(self.cfg.env.goal_distance))
        goal_pos, _ = self._lane_position_heading(ego, lane_id=self.current_goal_lane_id, longitudinal=self.current_goal_x)
        self.current_goal_y = float(goal_pos[1]) if np.asarray(goal_pos).shape[0] >= 2 else float(self.current_goal_lane_id * self.cfg.env.lane_width)

        used_by_lane: Dict[int, List[float]] = {int(ego_lane): [float(ego_x)]}
        vehicles = list(getattr(road, "vehicles", []) or [])
        npc_vehicles = [v for v in vehicles if v is not ego]
        for other in npc_vehicles:
            lane_id = int(rng.integers(0, int(self.cfg.env.lanes_count)))
            x = self._sample_npc_longitudinal(rng, lane_id=lane_id, used_by_lane=used_by_lane, ego_x=ego_x)
            speed = float(rng.uniform(float(self.cfg.env.vehicle_speed_min), float(self.cfg.env.vehicle_speed_max)))
            self._set_vehicle_state(other, lane_id=lane_id, longitudinal=x, speed=speed)
            used_by_lane.setdefault(lane_id, []).append(float(x))

    def action_dim(self) -> int:
        shape = getattr(self.env.action_space, "shape", None)
        if shape is None:
            raise TypeError("This script requires a continuous Box action space.")
        return int(np.prod(shape))

    def sample_random_action(self) -> np.ndarray:
        try:
            return self.clip_action(self.env.action_space.sample())
        except Exception:
            return np.random.uniform(self.action_low, self.action_high).astype(np.float32)

    def clip_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def build_low_state(self, obs) -> np.ndarray:
        scene = self.get_scene_dict(advance_state=True)
        obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        lane_width = max(float(scene.get("lane_width_runtime", self.cfg.env.lane_width)), 1e-6)
        scene_feat = np.array(
            [
                float(scene["ego_speed"]) / max(float(self.cfg.env.max_speed), 1e-6),
                np.clip(float(scene["front_distance"]) / 100.0, 0.0, 1.0),
                np.clip(float(scene["front_rel_speed"]) / 20.0, -1.0, 1.0),
                np.clip(float(scene["lane_id"]) / max(float(self.cfg.env.lanes_count - 1), 1.0), 0.0, 1.0),
                np.clip(float(scene["ttc"]) / 10.0, 0.0, 1.0),
                np.clip(float(scene.get("rear_distance", 1e6)) / 100.0, 0.0, 1.0),
                np.clip(float(scene.get("rear_rel_speed", 0.0)) / 20.0, -1.0, 1.0),
                np.clip(float(scene.get("signed_lane_offset_norm", 0.0)), -1.5, 1.5),
                np.clip(float(scene.get("abs_lane_offset_norm", 0.0)), 0.0, 1.5),
                np.clip(float(scene.get("heading_error", 0.0)) / 1.2, -1.5, 1.5),
                np.clip(float(scene.get("lateral_speed", 0.0)) / max(float(self.cfg.env.max_speed), 1e-6), -1.0, 1.0),
                np.clip(float(scene.get("road_left_margin", 0.0)) / max(2.0 * lane_width, 1e-6), 0.0, 1.5),
                np.clip(float(scene.get("road_right_margin", 0.0)) / max(2.0 * lane_width, 1e-6), 0.0, 1.5),
                np.clip(float(scene.get("road_boundary_margin", 0.0)) / max(float(self.cfg.cost.road_boundary_soft_margin), 1e-6), 0.0, 1.5),
                np.clip(float(scene.get("left_front_distance", 1e6)) / 100.0, 0.0, 1.0),
                np.clip(float(scene.get("right_front_distance", 1e6)) / 100.0, 0.0, 1.0),
                np.clip(float(scene.get("left_rear_distance", 1e6)) / 100.0, 0.0, 1.0),
                np.clip(float(scene.get("right_rear_distance", 1e6)) / 100.0, 0.0, 1.0),
                np.clip(float(scene.get("delta_ttc", 0.0)) / 5.0, -1.0, 1.0),
                np.clip(float(scene.get("delta_front_distance", 0.0)) / 30.0, -1.0, 1.0),
                np.clip(float(scene.get("delta_front_rel_speed", 0.0)) / 10.0, -1.0, 1.0),
                np.clip(float(scene.get("front_acc_proxy", 0.0)) / 6.0, -1.0, 1.0),
                np.clip(float(scene.get("recent_lane_change_flag", 0.0)), 0.0, 1.0),
                np.clip(float(scene.get("goal_distance_remaining", self.cfg.env.goal_distance)) / max(float(self.cfg.env.goal_distance), 1e-6), 0.0, 1.2),
                np.clip((float(scene.get("goal_lane_id", self.cfg.env.goal_lane_id)) - float(scene.get("lane_id", 0))) / max(float(self.cfg.env.lanes_count - 1), 1.0), -1.0, 1.0),
                np.clip(float(scene.get("goal_reached", 0.0)), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs_flat, scene_feat], axis=0).astype(np.float32)

    def get_scene_dict(self, advance_state: bool = False) -> dict:
        base_env = self.env.unwrapped
        ego = base_env.vehicle
        ego_x = float(ego.position[0])
        ego_y = float(ego.position[1])
        ego_speed = float(getattr(ego, "speed", 0.0))
        heading = float(getattr(ego, "heading", 0.0))
        lane_id = self._lane_id(ego)
        geometry = self._runtime_lane_geometry(ego, lane_id)
        current_lane = self._runtime_lane(ego, lane_id=lane_id)
        runtime_speed_limit = float(
            getattr(current_lane, "speed_limit", self.cfg.cost.speed_limit)
        ) if current_lane is not None else float(self.cfg.cost.speed_limit)
        runtime_speed_limit = max(runtime_speed_limit, 1e-6)

        front_same = self._nearest_vehicle(ego, lane_id)
        rear_same = self._nearest_rear_vehicle(ego, lane_id)
        left_lane = lane_id - 1 if lane_id > 0 else None
        right_lane = lane_id + 1 if lane_id < self.cfg.env.lanes_count - 1 else None
        front_left = self._nearest_vehicle(ego, left_lane) if left_lane is not None else None
        front_right = self._nearest_vehicle(ego, right_lane) if right_lane is not None else None
        rear_left = self._nearest_rear_vehicle(ego, left_lane) if left_lane is not None else None
        rear_right = self._nearest_rear_vehicle(ego, right_lane) if right_lane is not None else None

        front_distance, front_rel_speed = self._vehicle_relation(ego, front_same)
        rear_distance, rear_rel_speed = self._rear_vehicle_relation(ego, rear_same)
        left_front_distance, left_front_rel_speed = self._vehicle_relation(ego, front_left)
        right_front_distance, right_front_rel_speed = self._vehicle_relation(ego, front_right)
        left_rear_distance, left_rear_rel_speed = self._rear_vehicle_relation(ego, rear_left)
        right_rear_distance, right_rear_rel_speed = self._rear_vehicle_relation(ego, rear_right)
        front_closing_speed, ttc = self._front_closing_metrics(front_distance, front_rel_speed)
        rear_closing_speed, rear_ttc = self._rear_closing_metrics(rear_distance, rear_rel_speed)
        left_front_closing, left_front_ttc = self._front_closing_metrics(left_front_distance, left_front_rel_speed)
        right_front_closing, right_front_ttc = self._front_closing_metrics(right_front_distance, right_front_rel_speed)
        left_rear_closing, left_rear_ttc = self._rear_closing_metrics(left_rear_distance, left_rear_rel_speed)
        right_rear_closing, right_rear_ttc = self._rear_closing_metrics(right_rear_distance, right_rear_rel_speed)

        lane_width = max(float(geometry["lane_width"]), 1e-6)
        lane_center_y = float(geometry["lane_center_y"])
        signed_lane_offset = float(ego_y - lane_center_y)
        signed_lane_offset_norm = float(signed_lane_offset / lane_width)
        abs_lane_offset_norm = float(abs(signed_lane_offset_norm))
        left_lane_boundary_margin = max(0.0, 0.5 * lane_width + signed_lane_offset)
        right_lane_boundary_margin = max(0.0, 0.5 * lane_width - signed_lane_offset)
        lane_boundary_margin = min(left_lane_boundary_margin, right_lane_boundary_margin)
        road_min_y = float(geometry["road_min_y"])
        road_max_y = float(geometry["road_max_y"])
        road_left_margin = float(ego_y - road_min_y)
        road_right_margin = float(road_max_y - ego_y)
        road_boundary_margin = float(min(road_left_margin, road_right_margin))
        road_boundary_proximity = float(
            np.clip(
                (float(self.cfg.cost.road_boundary_soft_margin) - road_boundary_margin)
                / max(float(self.cfg.cost.road_boundary_soft_margin), 1e-6),
                0.0,
                1.0,
            )
        )
        lane_boundary_proximity = float(
            np.clip(
                (abs_lane_offset_norm - float(self.cfg.cost.lane_center_hard_ratio))
                / max(1.0 - float(self.cfg.cost.lane_center_hard_ratio), 1e-6),
                0.0,
                1.0,
            )
        )
        lane_dir_angle = float(geometry["lane_dir_angle"])
        try:
            velocity = np.asarray(getattr(ego, "velocity", np.zeros((2,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            if velocity.shape[0] >= 2:
                lateral_speed = float(-float(velocity[0]) * np.sin(lane_dir_angle) + float(velocity[1]) * np.cos(lane_dir_angle))
            else:
                lateral_speed = float(ego_speed * np.sin(heading - lane_dir_angle))
        except Exception:
            lateral_speed = float(ego_speed * np.sin(heading - lane_dir_angle))

        prev_scene = self.prev_scene or {}
        prev_lane = int(prev_scene.get("lane_id", lane_id)) if prev_scene else lane_id
        lane_changed = int(lane_id != prev_lane)
        recent_countdown = int(self.recent_lane_change_countdown)
        if lane_changed:
            recent_countdown = max(recent_countdown, 6)
        recent_lane_change_flag = 1.0 if recent_countdown > 0 else 0.0
        prev_ttc = float(prev_scene.get("ttc", ttc)) if prev_scene else float(ttc)
        prev_front_distance = float(prev_scene.get("front_distance", front_distance)) if prev_scene else float(front_distance)
        prev_front_rel_speed = float(prev_scene.get("front_rel_speed", front_rel_speed)) if prev_scene else float(front_rel_speed)
        prev_front_vehicle_speed = float(prev_scene.get("front_vehicle_speed", 0.0)) if prev_scene else 0.0
        front_vehicle_speed = float(ego_speed + front_rel_speed) if front_distance < 1e5 else 0.0
        delta_ttc = float(ttc - prev_ttc)
        delta_front_distance = float(front_distance - prev_front_distance)
        delta_front_rel_speed = float(front_rel_speed - prev_front_rel_speed)
        front_acc_proxy = float(front_vehicle_speed - prev_front_vehicle_speed)

        scene_out = {
            "ego_x": float(ego_x),
            "ego_y": float(ego_y),
            "ego_speed": float(ego_speed),
            "heading": float(heading),
            "heading_error": float(_wrap_to_pi(heading - lane_dir_angle)),
            "lane_dir_angle": float(lane_dir_angle),
            "lateral_speed": float(lateral_speed),
            "lane_id": int(lane_id),
            "lane_center_y": float(lane_center_y),
            "lane_width_runtime": float(lane_width),
            "speed_limit_runtime": float(runtime_speed_limit),
            "signed_lane_offset": float(signed_lane_offset),
            "signed_lane_offset_norm": float(signed_lane_offset_norm),
            "abs_lane_offset_norm": float(abs_lane_offset_norm),
            "left_lane_boundary_margin": float(left_lane_boundary_margin),
            "right_lane_boundary_margin": float(right_lane_boundary_margin),
            "lane_boundary_margin": float(lane_boundary_margin),
            "lane_boundary_proximity": float(lane_boundary_proximity),
            "road_left_margin": float(road_left_margin),
            "road_right_margin": float(road_right_margin),
            "road_boundary_margin": float(road_boundary_margin),
            "road_boundary_proximity": float(road_boundary_proximity),
            "left_lane_id": left_lane,
            "right_lane_id": right_lane,
            "front_distance": float(front_distance),
            "front_rel_speed": float(front_rel_speed),
            "front_closing_speed": float(front_closing_speed),
            "ttc": float(ttc),
            "rear_distance": float(rear_distance),
            "rear_rel_speed": float(rear_rel_speed),
            "rear_closing_speed": float(rear_closing_speed),
            "rear_ttc": float(rear_ttc),
            "left_front_distance": float(left_front_distance),
            "left_front_rel_speed": float(left_front_rel_speed),
            "left_front_closing_speed": float(left_front_closing),
            "left_front_ttc": float(left_front_ttc),
            "right_front_distance": float(right_front_distance),
            "right_front_rel_speed": float(right_front_rel_speed),
            "right_front_closing_speed": float(right_front_closing),
            "right_front_ttc": float(right_front_ttc),
            "left_rear_distance": float(left_rear_distance),
            "left_rear_rel_speed": float(left_rear_rel_speed),
            "left_rear_closing_speed": float(left_rear_closing),
            "left_rear_ttc": float(left_rear_ttc),
            "right_rear_distance": float(right_rear_distance),
            "right_rear_rel_speed": float(right_rear_rel_speed),
            "right_rear_closing_speed": float(right_rear_closing),
            "right_rear_ttc": float(right_rear_ttc),
            "collision": bool(getattr(ego, "crashed", False)),
            "last_action": self.last_action.copy(),
            "lane_changed": int(lane_changed),
            "recent_lane_change_flag": float(recent_lane_change_flag),
            "delta_ttc": float(delta_ttc),
            "delta_front_distance": float(delta_front_distance),
            "delta_front_rel_speed": float(delta_front_rel_speed),
            "front_acc_proxy": float(front_acc_proxy),
            "front_vehicle_speed": float(front_vehicle_speed),
            "goal_x": float(self.current_goal_x),
            "goal_y": float(self.current_goal_y),
            "goal_lane_id": int(self.current_goal_lane_id),
            "goal_distance_remaining": float(max(0.0, self.current_goal_x - ego_x)),
            "goal_longitudinal_progress": float(ego_x - (self.current_goal_x - float(self.cfg.env.goal_distance))),
            "goal_reached": bool(
                ego_x >= float(self.current_goal_x)
                and int(lane_id) == int(self.current_goal_lane_id)
                and abs_lane_offset_norm <= float(self.cfg.env.goal_lane_tolerance)
            ),
            "traffic_density": str(self.cfg.env.traffic_density),
            "traffic_vehicles_count": int(self.current_traffic_count),
        }
        if advance_state:
            self.prev_scene = dict(scene_out)
            self.recent_lane_change_countdown = max(0, recent_countdown - 1)
        return scene_out

    def render_frame(self) -> Optional[np.ndarray]:
        try:
            if self.legacy_api:
                frame = self.env.render(mode="rgb_array")
            else:
                frame = self.env.render()
        except Exception:
            return None
        if frame is None:
            return None
        frame = np.asarray(frame)
        if frame.ndim != 3:
            return None
        return frame.astype(np.uint8)

    def close(self) -> None:
        self.env.close()

    def get_runtime_env_snapshot(self) -> dict:
        return dict(self.runtime_env_snapshot)

    def _read_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        low = getattr(self.env.action_space, "low", None)
        high = getattr(self.env.action_space, "high", None)
        if low is None or high is None:
            raise TypeError("This script requires a finite continuous Box action space.")
        low_arr = np.asarray(low, dtype=np.float32).reshape(-1)
        high_arr = np.asarray(high, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(low_arr)) or not np.all(np.isfinite(high_arr)):
            low_arr = -np.ones(self.action_dim(), dtype=np.float32)
            high_arr = np.ones(self.action_dim(), dtype=np.float32)
        return low_arr.astype(np.float32), high_arr.astype(np.float32)

    def _make_env(self, env_id: str, env_config: Dict[str, Any], render_mode: Optional[str]):
        make_errors: List[str] = []
        if gym is not None:
            try:
                env = gym.make(env_id, config=env_config, render_mode=render_mode)
                self.api_backend = "gymnasium"
                self.legacy_api = False
                return env
            except Exception as exc:
                make_errors.append(f"gymnasium.make failed: {repr(exc)}")
        if legacy_gym is not None:
            try:
                env = legacy_gym.make(env_id, config=env_config)
                self.api_backend = "gym"
                self.legacy_api = True
                return env
            except Exception as exc:
                make_errors.append(f"gym.make failed: {repr(exc)}")
        detail = "; ".join(make_errors) if make_errors else "no gym backend is importable"
        raise RuntimeError(f"Unable to create env '{env_id}'. Details: {detail}")

    def _capture_runtime_env_snapshot(self, scene: Optional[dict] = None) -> dict:
        base_env = self.env.unwrapped
        if scene is None:
            try:
                scene = self.get_scene_dict()
            except Exception:
                scene = {}
        return {
            "env_id": str(self.cfg.env.env_id),
            "runtime_env_config": _jsonable_plain(getattr(base_env, "config", {}) or {}),
            "observation_space_shape": list(getattr(self.env.observation_space, "shape", ()) or ()),
            "action_space_shape": list(getattr(self.env.action_space, "shape", ()) or ()),
            "action_space_low": self.action_low.tolist(),
            "action_space_high": self.action_high.tolist(),
            "runtime_scene_probe": {
                "lane_width_runtime": float(scene.get("lane_width_runtime", self.cfg.env.lane_width)) if scene else float(self.cfg.env.lane_width),
                "lane_center_y_runtime": float(scene.get("lane_center_y", 0.0)) if scene else 0.0,
                "lane_dir_angle_runtime": float(scene.get("lane_dir_angle", 0.0)) if scene else 0.0,
                "speed_limit_runtime": float(scene.get("speed_limit_runtime", self.cfg.cost.speed_limit)) if scene else float(self.cfg.cost.speed_limit),
                "traffic_density": str(self.cfg.env.traffic_density),
                "traffic_vehicles_count": int(self.current_traffic_count),
                "goal_x": float(self.current_goal_x),
                "goal_y": float(self.current_goal_y),
                "goal_lane_id": int(self.current_goal_lane_id),
                "absolute_observation": bool(self.cfg.env.absolute_obs),
            },
        }

    def _runtime_lane_index(self, vehicle, lane_id: Optional[int] = None):
        lane_index = getattr(vehicle, "lane_index", None)
        if not isinstance(lane_index, tuple) or len(lane_index) < 3:
            return lane_index
        if lane_id is None:
            return lane_index
        return (lane_index[0], lane_index[1], int(lane_id))

    def _runtime_lane(self, vehicle, lane_id: Optional[int] = None):
        lane_index = self._runtime_lane_index(vehicle, lane_id=lane_id)
        if lane_index is None:
            return None
        try:
            return self.env.unwrapped.road.network.get_lane(lane_index)
        except Exception:
            return None

    def _runtime_lane_geometry(self, vehicle, lane_id: int) -> dict:
        fallback_center_y = float(lane_id * self.cfg.env.lane_width)
        fallback_lane_width = float(self.cfg.env.lane_width)
        current_lane = self._runtime_lane(vehicle, lane_id=lane_id)
        if current_lane is None:
            return {
                "lane_center_y": fallback_center_y,
                "lane_width": fallback_lane_width,
                "lane_dir_angle": 0.0,
                "road_min_y": -0.5 * fallback_lane_width,
                "road_max_y": (float(self.cfg.env.lanes_count) - 0.5) * fallback_lane_width,
            }

        try:
            longitudinal, _ = current_lane.local_coordinates(vehicle.position)
        except Exception:
            longitudinal = 0.0
        try:
            lane_center_y = float(current_lane.position(longitudinal, 0.0)[1])
        except Exception:
            lane_center_y = fallback_center_y
        try:
            lane_width = float(current_lane.width_at(longitudinal))
        except Exception:
            lane_width = fallback_lane_width
        try:
            lane_dir_angle = float(current_lane.heading_at(longitudinal))
        except Exception:
            lane_dir_angle = 0.0

        road_edges: List[Tuple[float, float]] = []
        for candidate_lane_id in range(max(int(self.cfg.env.lanes_count), 0)):
            lane = self._runtime_lane(vehicle, lane_id=candidate_lane_id)
            if lane is None:
                continue
            try:
                center_y = float(lane.position(longitudinal, 0.0)[1])
            except Exception:
                center_y = float(candidate_lane_id * self.cfg.env.lane_width)
            try:
                width = float(lane.width_at(longitudinal))
            except Exception:
                width = fallback_lane_width
            road_edges.append((center_y - 0.5 * width, center_y + 0.5 * width))
        if not road_edges:
            road_min_y = -0.5 * lane_width
            road_max_y = (float(self.cfg.env.lanes_count) - 0.5) * lane_width
        else:
            road_min_y = float(min(edge[0] for edge in road_edges))
            road_max_y = float(max(edge[1] for edge in road_edges))

        return {
            "lane_center_y": float(lane_center_y),
            "lane_width": float(max(lane_width, 1e-6)),
            "lane_dir_angle": float(lane_dir_angle),
            "road_min_y": float(road_min_y),
            "road_max_y": float(road_max_y),
        }

    def _lane_id(self, vehicle) -> int:
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None:
            return 0
        if isinstance(lane_index, tuple):
            return int(lane_index[-1])
        return int(lane_index)

    def _nearest_vehicle(self, ego, target_lane_id: Optional[int]):
        if target_lane_id is None:
            return None
        candidates = []
        for other in self.env.unwrapped.road.vehicles:
            if other is ego or self._lane_id(other) != target_lane_id:
                continue
            dx = float(other.position[0] - ego.position[0])
            if dx > 0.0:
                candidates.append((dx, other))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _nearest_rear_vehicle(self, ego, target_lane_id: Optional[int]):
        if target_lane_id is None:
            return None
        candidates = []
        for other in self.env.unwrapped.road.vehicles:
            if other is ego or self._lane_id(other) != target_lane_id:
                continue
            dx = float(other.position[0] - ego.position[0])
            if dx < 0.0:
                candidates.append((dx, other))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _vehicle_relation(self, ego, other) -> Tuple[float, float]:
        if other is None:
            return 1e6, 0.0
        distance = float(other.position[0] - ego.position[0])
        rel_speed = float(getattr(other, "speed", 0.0) - getattr(ego, "speed", 0.0))
        return max(distance, 0.0), rel_speed

    def _rear_vehicle_relation(self, ego, other) -> Tuple[float, float]:
        if other is None:
            return 1e6, 0.0
        distance = float(ego.position[0] - other.position[0])
        rel_speed = float(getattr(ego, "speed", 0.0) - getattr(other, "speed", 0.0))
        return max(distance, 0.0), rel_speed

    def _front_closing_metrics(self, distance: float, rel_speed: float) -> Tuple[float, float]:
        closing_speed = float(max(0.0, -float(rel_speed)))
        if float(distance) < 1e5 and closing_speed > 0.0:
            ttc = float(distance) / max(closing_speed, 1e-3)
        else:
            ttc = 99.0
        return float(closing_speed), float(ttc)

    def _rear_closing_metrics(self, distance: float, rear_rel_speed: float) -> Tuple[float, float]:
        closing_speed = float(max(0.0, -float(rear_rel_speed)))
        if float(distance) < 1e5 and closing_speed > 0.0:
            ttc = float(distance) / max(closing_speed, 1e-3)
        else:
            ttc = 99.0
        return float(closing_speed), float(ttc)


def is_offroad_scene(cfg: Config, scene: dict) -> bool:
    return bool(
        float(scene.get("road_boundary_margin", 1e6)) < float(cfg.cost.road_boundary_off_margin)
        or float(scene.get("abs_lane_offset_norm", 0.0)) > 1.10
    )


def terminal_failure(cfg: Config, scene: dict) -> bool:
    return bool(scene.get("collision", False) or is_offroad_scene(cfg, scene))


def goal_reached_scene(cfg: Config, scene: dict) -> bool:
    return bool(
        bool(scene.get("goal_reached", False))
        and not bool(scene.get("collision", False))
        and not is_offroad_scene(cfg, scene)
    )


def terminal_success(cfg: Config, scene: dict) -> bool:
    return goal_reached_scene(cfg, scene)


def lane_boundary_terms(cfg: Config, scene: dict) -> dict:
    center_soft = float(cfg.cost.lane_center_soft_ratio)
    center_hard = float(cfg.cost.lane_center_hard_ratio)
    road_soft = float(cfg.cost.road_boundary_soft_margin)
    road_hard = float(cfg.cost.road_boundary_hard_margin)
    abs_offset = float(scene.get("abs_lane_offset_norm", 0.0))
    heading_abs = abs(float(scene.get("heading_error", 0.0)))
    lateral_speed = abs(float(scene.get("lateral_speed", 0.0)))
    road_margin = float(scene.get("road_boundary_margin", 1e6))
    center_soft_term = _upper_quadratic_cost(abs_offset, center_soft, center_hard)
    center_hard_term = _upper_quadratic_cost(abs_offset, center_hard, 1.0)
    road_soft_term = _squared_barrier_cost(road_margin, road_soft)
    road_hard_term = _squared_barrier_cost(road_margin, road_hard)
    near_edge = _clip01((road_soft - road_margin) / max(road_soft, 1e-6))
    heading_term = _clip01(heading_abs / 1.05) * max(near_edge, 0.35 * center_hard_term)
    lateral_term = _clip01(lateral_speed / max(0.45 * float(cfg.env.max_speed), 1e-6)) * max(near_edge, 0.35 * center_hard_term)
    lane_cost = _clip01(max(center_hard_term, road_soft_term, road_hard_term, 0.60 * heading_term, 0.60 * lateral_term))
    return {
        "center_soft_term": float(center_soft_term),
        "center_hard_term": float(center_hard_term),
        "road_soft_term": float(road_soft_term),
        "road_hard_term": float(road_hard_term),
        "heading_term": float(heading_term),
        "lateral_term": float(lateral_term),
        "lane_cost": float(lane_cost),
    }


def compute_costs(cfg: Config, scene: dict, next_scene: dict, action) -> dict:
    """Compute clear CMDP constraint costs from simulator variables.

    Let v be ego speed, d_f the nearest front distance, TTC the time-to-collision,
    e_l the normalized lane-center offset, and v_lim the lane speed limit.

      C_col      = 1[collision]
      C_headway  = max(0, (d_safe - d_f) / d_safe), d_safe=max(d_min, tau_h*v)
      C_ttc      = max(0, (TTC_safe - TTC) / TTC_safe)
      C_safety   = max(C_col, C_headway, C_ttc)
      C_boundary = max(1[offroad], max(0, (|e_l|-e_soft)/(e_hard-e_soft)))
      C_speed    = max(0, (v-v_lim)/v_lim)
      C_comfort  = mean normalized action magnitude and action change
      C_total    = weighted sum of the above.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    last_action = np.asarray(scene.get("last_action", np.zeros_like(action)), dtype=np.float32).reshape(-1)

    collision_cost = 1.0 if bool(next_scene.get("collision", False)) else 0.0

    front_distance = float(next_scene.get("front_distance", 1e6))
    ego_speed = max(0.0, float(next_scene.get("ego_speed", 0.0)))
    d_safe = max(float(cfg.cost.min_headway), float(cfg.cost.time_headway_sec) * ego_speed)
    headway_cost = _clip01((d_safe - front_distance) / max(d_safe, 1e-6))

    ttc = float(next_scene.get("ttc", 99.0))
    ttc_cost = _clip01((float(cfg.cost.ttc_safe) - ttc) / max(float(cfg.cost.ttc_safe), 1e-6))
    front_collision = max(collision_cost, ttc_cost, headway_cost)

    rear_distance = float(next_scene.get("rear_distance", 1e6))
    rear_ttc = float(next_scene.get("rear_ttc", 99.0))
    rear_gap_cost = _clip01((float(cfg.cost.rear_gap_distance) - rear_distance) / max(float(cfg.cost.rear_gap_distance), 1e-6))
    rear_ttc_cost = _clip01((float(cfg.cost.rear_ttc) - rear_ttc) / max(float(cfg.cost.rear_ttc), 1e-6))
    rear_collision = max(rear_gap_cost, rear_ttc_cost)

    lane_changed = int(next_scene.get("lane_id", scene.get("lane_id", 0)) != scene.get("lane_id", 0))
    side_front = min(float(next_scene.get("left_front_distance", 1e6)), float(next_scene.get("right_front_distance", 1e6)))
    side_rear = min(float(next_scene.get("left_rear_distance", 1e6)), float(next_scene.get("right_rear_distance", 1e6)))
    side_front_cost = _clip01((float(cfg.cost.side_gap_front) - side_front) / max(float(cfg.cost.side_gap_front), 1e-6))
    side_rear_cost = _clip01((float(cfg.cost.side_gap_rear) - side_rear) / max(float(cfg.cost.side_gap_rear), 1e-6))
    side_cost = float(lane_changed) * max(side_front_cost, side_rear_cost)

    safety_cost = _clip01(max(collision_cost, headway_cost, ttc_cost, 0.65 * rear_collision, side_cost))

    speed_limit = max(float(next_scene.get("speed_limit_runtime", cfg.cost.speed_limit)), 1e-6)
    overspeed_cost = _clip01((ego_speed - speed_limit) / speed_limit)

    abs_lane_offset = float(next_scene.get("abs_lane_offset_norm", 0.0))
    lane_soft = float(cfg.cost.lane_center_soft_ratio)
    lane_hard = max(float(cfg.cost.lane_center_hard_ratio), lane_soft + 1e-6)
    lane_center_cost = _clip01((abs_lane_offset - lane_soft) / max(lane_hard - lane_soft, 1e-6))
    offroad_cost = 1.0 if is_offroad_scene(cfg, next_scene) else 0.0
    lane_cost = _clip01(max(lane_center_cost, offroad_cost))

    acc_cmd = abs(float(action[0])) * float(cfg.cost.action_acc_scale) if action.shape[0] > 0 else 0.0
    steer_cmd = abs(float(action[1])) * float(cfg.cost.action_steer_scale) if action.shape[0] > 1 else 0.0
    acc_delta = abs(float(action[0] - last_action[0])) * float(cfg.cost.action_acc_scale) if action.shape[0] > 0 else 0.0
    steer_delta = abs(float(action[1] - last_action[1])) * float(cfg.cost.action_steer_scale) if action.shape[0] > 1 else 0.0
    comfort_cost = _clip01(
        0.25 * _clip01(acc_cmd / max(float(cfg.cost.acc_limit), 1e-6))
        + 0.25 * _clip01(steer_cmd / max(float(cfg.cost.steer_limit), 1e-6))
        + 0.25 * _clip01(acc_delta / max(float(cfg.cost.action_delta_limit), 1e-6))
        + 0.25 * _clip01(steer_delta / max(float(cfg.cost.steer_limit), 1e-6))
    )

    total_cost = (
        float(cfg.cost.collision_weight) * safety_cost
        + float(cfg.cost.overspeed_weight) * overspeed_cost
        + float(cfg.cost.comfort_weight) * comfort_cost
        + float(cfg.cost.lane_weight) * lane_cost
    )
    return {
        "collision_cost": float(collision_cost),
        "collision_cost_front": float(front_collision),
        "collision_cost_rear": float(rear_collision),
        "collision_cost_lateral": float(side_cost),
        "headway_cost": float(max(headway_cost, ttc_cost)),
        "overspeed_cost": float(overspeed_cost),
        "comfort_cost": float(comfort_cost),
        "lane_cost": float(lane_cost),
        "safety_cost": float(safety_cost),
        "boundary_cost": float(lane_cost),
        "offroad_cost": float(offroad_cost),
        "total_cost": float(total_cost),
    }


def classify_transition_danger(cfg: Config, scene: dict, next_scene: dict, cost_dict: dict) -> dict:
    collision = bool(next_scene.get("collision", False)) or float(cost_dict.get("collision_cost", 0.0)) >= 0.99
    headway_violation = float(next_scene.get("front_distance", 1e6)) < float(cfg.cost.min_headway)
    low_ttc = float(next_scene.get("ttc", 99.0)) < float(cfg.cost.ttc_safe)
    lane_changed = bool(next_scene.get("lane_id", scene.get("lane_id", 0)) != scene.get("lane_id", 0))
    front_prev = float(scene.get("front_distance", 1e6))
    front_next = float(next_scene.get("front_distance", 1e6))
    front_shrink = float(front_prev - front_next)
    lane_change_shrink = bool(
        lane_changed
        and front_shrink > 4.0
        and front_next < (2.0 * float(cfg.cost.min_headway))
    )
    near_danger = bool(headway_violation or low_ttc or lane_change_shrink)
    label = 2 if collision else (1 if near_danger else 0)
    return {
        "danger_label": int(label),
        "is_collision": bool(collision),
        "is_near_danger": bool(near_danger),
    }


def _mission_reward(cfg: Config, next_scene: dict, timeout_failure: bool = False) -> Tuple[float, bool, bool]:
    success = terminal_success(cfg, next_scene)
    failure = bool(terminal_failure(cfg, next_scene) or timeout_failure)
    if success:
        return float(cfg.reward.mission_success_reward), bool(success), bool(failure)
    if failure:
        return float(cfg.reward.mission_failure_penalty), bool(success), bool(failure)
    return 0.0, bool(success), bool(failure)


def compute_goal_shaped_reward(
    cfg: Config,
    scene: dict,
    next_scene: dict,
    action,
    env_reward: float,
    timeout_failure: bool = False,
) -> Tuple[float, dict]:
    """Dense reward used by the refactored SAC/CSAC baselines.

      R = R_ms + w_p * Delta_progress + w_v * v/v_lim + w_g * lane_goal_score
          - w_c * e_lane^2 - w_h * |heading_error| - w_a * |a| - w_s * |Delta a|.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    last_action = np.asarray(scene.get("last_action", np.zeros_like(action)), dtype=np.float32).reshape(-1)
    mission_term, success, failure = _mission_reward(cfg, next_scene, timeout_failure=timeout_failure)

    dt = 1.0 / max(float(cfg.env.policy_frequency), 1e-6)
    progress_delta = float(next_scene.get("goal_longitudinal_progress", 0.0)) - float(scene.get("goal_longitudinal_progress", 0.0))
    progress_term = float(np.clip(progress_delta / max(float(cfg.env.max_speed) * dt, 1e-6), -1.0, 1.0))

    speed_norm = float(np.clip(float(next_scene.get("ego_speed", 0.0)) / max(float(cfg.env.max_speed), 1e-6), 0.0, 1.2))
    lane_gap = abs(float(cfg.env.goal_lane_id) - float(next_scene.get("lane_id", 0)))
    lane_goal_score = 1.0 - float(np.clip(lane_gap / max(float(cfg.env.lanes_count - 1), 1.0), 0.0, 1.0))

    lane_center_penalty = float(np.clip(float(next_scene.get("abs_lane_offset_norm", 0.0)) ** 2, 0.0, cfg.reward.max_lane_center_penalty))
    heading_penalty = float(np.clip(abs(float(next_scene.get("heading_error", 0.0))) / 1.2, 0.0, cfg.reward.max_heading_penalty))
    action_penalty = float(np.clip(np.mean(np.abs(action)), 0.0, cfg.reward.max_action_penalty))
    smooth_penalty = float(np.clip(np.mean(np.abs(action - last_action)), 0.0, cfg.reward.max_smooth_penalty))
    env_term = float(cfg.reward.env_reward_scale) * float(env_reward) if bool(cfg.reward.use_env_reward) else 0.0

    reward = (
        mission_term
        + float(cfg.reward.goal_progress_weight) * progress_term
        + 0.25 * speed_norm
        + float(cfg.reward.goal_lane_weight) * lane_goal_score
        + env_term
        - float(cfg.reward.lane_center_weight) * lane_center_penalty
        - float(cfg.reward.heading_weight) * heading_penalty
        - float(cfg.reward.action_weight) * action_penalty
        - float(cfg.reward.smooth_weight) * smooth_penalty
    )
    terms = {
        "mission_term": float(mission_term),
        "success_term": float(int(success)),
        "failure_term": float(int(failure)),
        "goal_progress_reward": float(progress_term),
        "speed_reward": float(speed_norm),
        "goal_lane_reward": float(lane_goal_score),
        "env_reward_term": float(env_term),
        "lane_center_penalty": float(lane_center_penalty),
        "heading_penalty": float(heading_penalty),
        "action_penalty": float(action_penalty),
        "smooth_penalty": float(smooth_penalty),
        "reward": float(reward),
        "source": "goal_shaped",
    }
    return float(reward), terms


def compute_paper_formula_reward(
    cfg: Config,
    scene: dict,
    next_scene: dict,
    action,
    env_reward: float,
    timeout_failure: bool = False,
) -> Tuple[float, dict]:
    """Paper-style reward shown in the prompt.

      R = R_ms + R_lc + R_e + R_s
      R_ms = 10 if mission success, -10 if mission failure, 0 otherwise.
      R_lc = 0.4 if the heading points toward the goal lane, -0.5 if away from it, 0 otherwise.
      R_e  = 0.3*(v-v_min)/(v_max-v_min), if v in [v_min, v_max], otherwise 0.
      R_s  = -1/(TTC+0.1), if TTC < 3, otherwise 0.
    """
    mission_term, success, failure = _mission_reward(cfg, next_scene, timeout_failure=timeout_failure)

    lane_delta = float(cfg.env.goal_lane_id) - float(next_scene.get("lane_id", 0))
    desired_sign = float(np.sign(lane_delta))
    heading_error = float(next_scene.get("heading_error", 0.0))
    phi_toward_goal = desired_sign * heading_error if desired_sign != 0.0 else 0.0
    eps = float(cfg.reward.paper_heading_epsilon)
    if desired_sign == 0.0:
        lane_change_term = 0.0
    elif phi_toward_goal > eps:
        lane_change_term = float(cfg.reward.paper_lane_change_reward)
    elif phi_toward_goal < -eps:
        lane_change_term = float(cfg.reward.paper_wrong_lane_change_penalty)
    else:
        lane_change_term = 0.0

    v = float(next_scene.get("ego_speed", 0.0))
    v_min = float(cfg.reward.paper_speed_min)
    v_max = max(float(next_scene.get("speed_limit_runtime", cfg.cost.speed_limit)), v_min + 1e-6)
    if v_min <= v <= v_max:
        speed_term = float(cfg.reward.paper_speed_reward_scale) * (v - v_min) / max(v_max - v_min, 1e-6)
    else:
        speed_term = 0.0

    ttc = float(next_scene.get("ttc", 99.0))
    if ttc < float(cfg.reward.paper_ttc_threshold):
        safety_term = -1.0 / (max(ttc, 0.0) + float(cfg.reward.paper_ttc_eps))
    else:
        safety_term = 0.0

    reward = float(mission_term + lane_change_term + speed_term + safety_term)
    terms = {
        "R_ms": float(mission_term),
        "R_lc": float(lane_change_term),
        "R_e": float(speed_term),
        "R_s": float(safety_term),
        "success_term": float(int(success)),
        "failure_term": float(int(failure)),
        "reward": float(reward),
        "source": "paper_formula_Rms_Rlc_Re_Rs",
    }
    return float(reward), terms


def compute_reward(
    cfg: Config,
    scene: dict,
    next_scene: dict,
    action,
    env_reward: float,
    timeout_failure: bool = False,
) -> Tuple[float, dict]:
    reward_type = str(getattr(cfg.reward, "reward_type", "goal_shaped")).lower().strip()
    if reward_type == "paper_formula":
        return compute_paper_formula_reward(cfg, scene, next_scene, action, env_reward, timeout_failure=timeout_failure)
    return compute_goal_shaped_reward(cfg, scene, next_scene, action, env_reward, timeout_failure=timeout_failure)


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.state_dim = int(state_dim)
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_collision = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_headway = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_overspeed = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_comfort = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_lane = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_total = np.zeros((capacity, 1), dtype=np.float32)
        self.cost_safety = np.zeros((capacity, 1), dtype=np.float32)
        self.episode_id = np.full((capacity,), -1, dtype=np.int64)
        self.step_in_episode = np.full((capacity,), -1, dtype=np.int32)
        self.danger_label = np.zeros((capacity,), dtype=np.int8)  # 0 normal, 1 near-danger, 2 danger/collision
        self.success_flag = np.full((capacity,), -1, dtype=np.int8)
        self.collision_flag = np.zeros((capacity,), dtype=np.int8)
        self.active_indices: set[int] = set()
        self.label_index_sets: Dict[int, set[int]] = {0: set(), 1: set(), 2: set()}
        self.episode_step_to_index: Dict[Tuple[int, int], int] = {}
        self.index_to_episode_step: Dict[int, Tuple[int, int]] = {}
        self.episode_to_indices: Dict[int, set[int]] = defaultdict(set)

    def _remove_index(self, index: int) -> None:
        if index in self.active_indices:
            self.active_indices.remove(index)
        old_label = int(self.danger_label[index])
        if old_label in self.label_index_sets:
            self.label_index_sets[old_label].discard(index)
        key = self.index_to_episode_step.pop(index, None)
        if key is not None:
            self.episode_step_to_index.pop(key, None)
            ep = int(key[0])
            if ep in self.episode_to_indices:
                self.episode_to_indices[ep].discard(index)
                if not self.episode_to_indices[ep]:
                    self.episode_to_indices.pop(ep, None)

    def _set_label(self, index: int, label: int) -> None:
        label = int(np.clip(label, 0, 2))
        old = int(self.danger_label[index])
        if old == label:
            return
        self.label_index_sets[old].discard(index)
        self.danger_label[index] = label
        self.label_index_sets[label].add(index)

    def mark_precollision_danger(self, episode_id: int, current_step: int, back_steps: int) -> None:
        if back_steps <= 0:
            return
        start = max(0, int(current_step) - int(back_steps))
        end = max(start, int(current_step))
        for s in range(start, end):
            key = (int(episode_id), int(s))
            idx = self.episode_step_to_index.get(key)
            if idx is not None:
                self._set_label(int(idx), 2)

    def mark_episode_success(self, episode_id: int, success: bool) -> None:
        indices = list(self.episode_to_indices.get(int(episode_id), set()))
        if not indices:
            return
        mark = np.int8(1 if bool(success) else 0)
        for idx in indices:
            self.success_flag[int(idx)] = mark

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        cost_dict: Optional[dict] = None,
        transition_meta: Optional[dict] = None,
    ) -> int:
        cost_dict = cost_dict or {}
        transition_meta = transition_meta or {}
        if self.size >= self.capacity:
            self._remove_index(self.ptr)
        self.state[self.ptr] = np.asarray(state, dtype=np.float32)
        self.action[self.ptr] = np.asarray(action, dtype=np.float32)
        self.reward[self.ptr] = float(reward)
        self.next_state[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.done[self.ptr] = float(done)
        self.cost_collision[self.ptr] = float(cost_dict.get("collision_cost", 0.0))
        self.cost_headway[self.ptr] = float(cost_dict.get("headway_cost", 0.0))
        self.cost_overspeed[self.ptr] = float(cost_dict.get("overspeed_cost", 0.0))
        self.cost_comfort[self.ptr] = float(cost_dict.get("comfort_cost", 0.0))
        self.cost_lane[self.ptr] = float(cost_dict.get("lane_cost", 0.0))
        self.cost_total[self.ptr] = float(cost_dict.get("total_cost", 0.0))
        self.cost_safety[self.ptr] = float(cost_dict.get("safety_cost", 0.0))
        ep_id = int(transition_meta.get("episode_id", -1))
        step_id = int(transition_meta.get("step_in_episode", -1))
        self.episode_id[self.ptr] = ep_id
        self.step_in_episode[self.ptr] = step_id
        is_collision = bool(transition_meta.get("is_collision", False))
        is_near_danger = bool(transition_meta.get("is_near_danger", False))
        label = 2 if is_collision else (1 if is_near_danger else int(transition_meta.get("danger_label", 0)))
        self.danger_label[self.ptr] = np.int8(int(np.clip(label, 0, 2)))
        self.success_flag[self.ptr] = np.int8(-1)
        self.collision_flag[self.ptr] = np.int8(1 if is_collision else 0)
        self.active_indices.add(self.ptr)
        self.label_index_sets[int(self.danger_label[self.ptr])].add(self.ptr)
        key = (ep_id, step_id)
        self.episode_step_to_index[key] = int(self.ptr)
        self.index_to_episode_step[int(self.ptr)] = key
        self.episode_to_indices[ep_id].add(int(self.ptr))
        added_index = int(self.ptr)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return added_index

    def _draw_from_pool(self, pool: set[int], n: int) -> List[int]:
        if n <= 0:
            return []
        if not pool:
            return []
        arr = np.asarray(list(pool), dtype=np.int64)
        idx = np.random.choice(arr, size=int(n), replace=bool(len(arr) < int(n)))
        return [int(v) for v in idx.tolist()]

    def _sample_indices(self, batch_size: int, priority_cfg: Optional[dict]) -> Tuple[np.ndarray, Dict[str, float]]:
        valid = np.asarray(list(self.active_indices), dtype=np.int64)
        if valid.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        near_pool = self.label_index_sets[1]
        danger_pool = self.label_index_sets[2]
        if not priority_cfg or not bool(priority_cfg.get("enabled", False)):
            idx = np.random.choice(valid, size=int(batch_size), replace=bool(valid.size < int(batch_size)))
            stats = {
                "sampled_near_danger_count": float(np.sum(self.danger_label[idx] == 1)),
                "sampled_collision_count": float(np.sum(self.collision_flag[idx] > 0)),
                "sampled_success_count": float(np.sum(self.success_flag[idx] == 1)),
                "sampled_danger_ratio_actual": float(np.mean(self.danger_label[idx] == 2)),
                "sampled_near_ratio_actual": float(np.mean(self.danger_label[idx] == 1)),
                "replay_danger_count": float(len(danger_pool)),
                "replay_near_danger_count": float(len(near_pool)),
            }
            return idx.astype(np.int64), stats

        danger_ratio = float(np.clip(priority_cfg.get("danger_ratio", 0.0), 0.0, 1.0))
        near_ratio = float(np.clip(priority_cfg.get("near_ratio", 0.0), 0.0, 1.0 - danger_ratio))
        n_danger = int(round(int(batch_size) * danger_ratio))
        n_near = int(round(int(batch_size) * near_ratio))
        n_normal = int(batch_size) - n_danger - n_near
        sampled: List[int] = []
        sampled += self._draw_from_pool(danger_pool, n_danger)
        sampled += self._draw_from_pool(near_pool, n_near)
        normal_pool = self.active_indices.difference(danger_pool).difference(near_pool)
        sampled += self._draw_from_pool(normal_pool, n_normal)
        short = int(batch_size) - len(sampled)
        if short > 0:
            sampled += self._draw_from_pool(self.active_indices, short)
        if len(sampled) <= 0:
            sampled = [int(v) for v in np.random.choice(valid, size=int(batch_size), replace=True).tolist()]
        idx = np.asarray(sampled, dtype=np.int64)
        if idx.size > int(batch_size):
            idx = idx[: int(batch_size)]
        np.random.shuffle(idx)
        stats = {
            "sampled_near_danger_count": float(np.sum(self.danger_label[idx] == 1)),
            "sampled_collision_count": float(np.sum(self.collision_flag[idx] > 0)),
            "sampled_success_count": float(np.sum(self.success_flag[idx] == 1)),
            "sampled_danger_ratio_actual": float(np.mean(self.danger_label[idx] == 2)),
            "sampled_near_ratio_actual": float(np.mean(self.danger_label[idx] == 1)),
            "replay_danger_count": float(len(danger_pool)),
            "replay_near_danger_count": float(len(near_pool)),
        }
        return idx, stats

    def _build_sequence(self, index: int, seq_len: int, for_next: bool) -> Tuple[np.ndarray, np.ndarray]:
        seq = np.zeros((int(seq_len), self.state_dim), dtype=np.float32)
        mask = np.zeros((int(seq_len),), dtype=np.float32)
        ep = int(self.episode_id[index])
        step = int(self.step_in_episode[index])
        for t in range(int(seq_len)):
            target_step = int(step - (int(seq_len) - 1 - t) + (1 if for_next else 0))
            key = (ep, target_step)
            src_idx = self.episode_step_to_index.get(key)
            if src_idx is not None and src_idx in self.active_indices:
                seq[t] = self.state[int(src_idx)]
                mask[t] = 1.0
            elif for_next and target_step == step + 1:
                seq[t] = self.next_state[index]
                mask[t] = 1.0
        if float(np.sum(mask)) <= 0.0:
            seq[-1] = self.next_state[index] if for_next else self.state[index]
            mask[-1] = 1.0
        return seq, mask

    def _nstep_collision_headway(self, index: int, n_step: int, gamma: float) -> Tuple[float, float, int, float, float]:
        ep = int(self.episode_id[index])
        step = int(self.step_in_episode[index])
        collision_total = 0.0
        headway_total = 0.0
        disc = 1.0
        bootstrap_index: Optional[int] = None
        done_n = 1.0
        for k in range(max(1, int(n_step))):
            key = (ep, step + k)
            idx = self.episode_step_to_index.get(key)
            if idx is None or idx not in self.active_indices:
                break
            idx_i = int(idx)
            collision_total += disc * float(self.cost_collision[idx_i, 0])
            headway_total += disc * float(self.cost_headway[idx_i, 0])
            bootstrap_index = idx_i
            if float(self.done[idx_i, 0]) >= 0.5:
                done_n = 1.0
                break
            disc *= float(gamma)
            done_n = 0.0
        if bootstrap_index is None:
            bootstrap_index = int(index)
            done_n = 1.0
            disc = 0.0
        bootstrap_discount = 0.0 if done_n >= 0.5 else float(disc)
        return float(collision_total), float(headway_total), int(bootstrap_index), float(done_n), float(bootstrap_discount)

    def sample(self, batch_size: int, sample_cfg: Optional[dict] = None) -> dict:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        sample_cfg = sample_cfg or {}
        idx, stats = self._sample_indices(int(batch_size), priority_cfg=sample_cfg)
        seq_len = max(1, int(sample_cfg.get("seq_len", 1)))
        n_step = max(1, int(sample_cfg.get("safety_n_step", 1)))
        gamma = float(sample_cfg.get("gamma", 0.99))
        safety_w_collision = float(sample_cfg.get("safety_w_collision", 1.0))
        safety_w_headway = float(sample_cfg.get("safety_w_headway", 1.0))
        safety_weight_sum = max(safety_w_collision + safety_w_headway, 1e-6)
        state_seq = np.zeros((idx.shape[0], seq_len, self.state_dim), dtype=np.float32)
        next_state_seq = np.zeros((idx.shape[0], seq_len, self.state_dim), dtype=np.float32)
        state_mask = np.zeros((idx.shape[0], seq_len), dtype=np.float32)
        next_state_mask = np.zeros((idx.shape[0], seq_len), dtype=np.float32)
        bootstrap_state_seq = np.zeros((idx.shape[0], seq_len, self.state_dim), dtype=np.float32)
        bootstrap_state_mask = np.zeros((idx.shape[0], seq_len), dtype=np.float32)
        bootstrap_done = np.zeros((idx.shape[0], 1), dtype=np.float32)
        bootstrap_discount = np.zeros((idx.shape[0], 1), dtype=np.float32)
        nstep_collision = np.zeros((idx.shape[0], 1), dtype=np.float32)
        nstep_headway = np.zeros((idx.shape[0], 1), dtype=np.float32)
        nstep_safety = np.zeros((idx.shape[0], 1), dtype=np.float32)
        for i, j in enumerate(idx.tolist()):
            s_seq, s_m = self._build_sequence(int(j), seq_len=seq_len, for_next=False)
            ns_seq, ns_m = self._build_sequence(int(j), seq_len=seq_len, for_next=True)
            state_seq[i] = s_seq
            next_state_seq[i] = ns_seq
            state_mask[i] = s_m
            next_state_mask[i] = ns_m
            (
                nstep_collision[i, 0],
                nstep_headway[i, 0],
                bootstrap_idx,
                bootstrap_done[i, 0],
                bootstrap_discount[i, 0],
            ) = self._nstep_collision_headway(int(j), n_step=n_step, gamma=gamma)
            bs_seq, bs_m = self._build_sequence(int(bootstrap_idx), seq_len=seq_len, for_next=True)
            bootstrap_state_seq[i] = bs_seq
            bootstrap_state_mask[i] = bs_m
            nstep_safety[i, 0] = (
                safety_w_collision * nstep_collision[i, 0]
                + safety_w_headway * nstep_headway[i, 0]
            ) / safety_weight_sum
        return {
            "state": torch.as_tensor(self.state[idx], device=self.device),
            "action": torch.as_tensor(self.action[idx], device=self.device),
            "reward": torch.as_tensor(self.reward[idx], device=self.device),
            "next_state": torch.as_tensor(self.next_state[idx], device=self.device),
            "done": torch.as_tensor(self.done[idx], device=self.device),
            "cost_collision": torch.as_tensor(self.cost_collision[idx], device=self.device),
            "cost_headway": torch.as_tensor(self.cost_headway[idx], device=self.device),
            "cost_overspeed": torch.as_tensor(self.cost_overspeed[idx], device=self.device),
            "cost_comfort": torch.as_tensor(self.cost_comfort[idx], device=self.device),
            "cost_lane": torch.as_tensor(self.cost_lane[idx], device=self.device),
            "cost_total": torch.as_tensor(self.cost_total[idx], device=self.device),
            "cost_safety": torch.as_tensor(self.cost_safety[idx], device=self.device),
            "cost_collision_nstep": torch.as_tensor(nstep_collision, device=self.device),
            "cost_headway_nstep": torch.as_tensor(nstep_headway, device=self.device),
            "cost_safety_nstep": torch.as_tensor(nstep_safety, device=self.device),
            "state_seq": torch.as_tensor(state_seq, device=self.device),
            "next_state_seq": torch.as_tensor(next_state_seq, device=self.device),
            "state_seq_mask": torch.as_tensor(state_mask, device=self.device),
            "next_state_seq_mask": torch.as_tensor(next_state_mask, device=self.device),
            "constraint_bootstrap_state_seq": torch.as_tensor(bootstrap_state_seq, device=self.device),
            "constraint_bootstrap_state_mask": torch.as_tensor(bootstrap_state_mask, device=self.device),
            "constraint_bootstrap_done": torch.as_tensor(bootstrap_done, device=self.device),
            "constraint_bootstrap_discount": torch.as_tensor(bootstrap_discount, device=self.device),
            "sampled_success_count": float(stats["sampled_success_count"]),
            "sampled_near_danger_count": float(stats["sampled_near_danger_count"]),
            "sampled_collision_count": float(stats["sampled_collision_count"]),
            "sampled_danger_ratio_actual": float(stats["sampled_danger_ratio_actual"]),
            "sampled_near_ratio_actual": float(stats["sampled_near_ratio_actual"]),
            "replay_danger_count": float(stats["replay_danger_count"]),
            "replay_near_danger_count": float(stats["replay_near_danger_count"]),
        }


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=-1))


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_low, action_high):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32).view(1, -1)
        high = torch.as_tensor(action_high, dtype=torch.float32).view(1, -1)
        scale = torch.clamp((high - low) * 0.5, min=1e-6)
        bias = (high + low) * 0.5
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def forward(self, state):
        h = self.trunk(state)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        unit_action = torch.tanh(z)
        action = self.action_bias + self.action_scale * unit_action
        log_prob = dist.log_prob(z) - torch.log(1.0 - unit_action.pow(2) + 1e-6)
        log_prob = log_prob - torch.log(self.action_scale + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mu_action = self.action_bias + self.action_scale * torch.tanh(mu)
        return action, log_prob, mu_action


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, recurrent: bool):
        super().__init__()
        self.recurrent = bool(recurrent)
        self.hidden_dim = int(hidden_dim)
        if self.recurrent:
            self.gru = nn.GRU(input_size=int(input_dim), hidden_size=int(hidden_dim), num_layers=1, batch_first=True)
        else:
            self.ff = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
            )

    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if seq.dim() == 2:
            seq = seq.unsqueeze(1)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(1)
        if self.recurrent:
            out, _ = self.gru(seq)
            if mask is None:
                return out[:, -1, :]
            lengths = torch.clamp(mask.sum(dim=1).long(), min=1)
            gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
            return out.gather(1, gather_idx).squeeze(1)
        last = seq[:, -1, :]
        return self.ff(last)


class SequenceCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, recurrent: bool):
        super().__init__()
        self.encoder = SequenceEncoder(state_dim, hidden_dim, recurrent=recurrent)
        self.q = MLP(hidden_dim + action_dim, 1, hidden_dim)

    def forward(self, state_seq: torch.Tensor, action: torch.Tensor, state_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.encoder(state_seq, state_mask)
        return self.q(torch.cat([z, action], dim=-1))


class Stage4Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_low, action_high, recurrent: bool, decoupled: bool):
        super().__init__()
        self.action_dim = int(action_dim)
        self.decoupled = bool(decoupled and action_dim >= 2)
        self.encoder = SequenceEncoder(state_dim, hidden_dim, recurrent=recurrent)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if self.decoupled:
            self.long_mu = nn.Linear(hidden_dim, 1)
            self.long_log_std = nn.Linear(hidden_dim, 1)
            self.lat_mu = nn.Linear(hidden_dim, 1)
            self.lat_log_std = nn.Linear(hidden_dim, 1)
            extra_dim = max(0, int(action_dim) - 2)
            if extra_dim > 0:
                self.extra_mu = nn.Linear(hidden_dim, extra_dim)
                self.extra_log_std = nn.Linear(hidden_dim, extra_dim)
            else:
                self.extra_mu = None
                self.extra_log_std = None
        else:
            self.mu_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32).view(1, -1)
        high = torch.as_tensor(action_high, dtype=torch.float32).view(1, -1)
        scale = torch.clamp((high - low) * 0.5, min=1e-6)
        bias = (high + low) * 0.5
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def forward(self, state_seq: torch.Tensor, state_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(state_seq, state_mask)
        h = self.trunk(h)
        if not self.decoupled:
            mu = self.mu_head(h)
            log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
            return mu, log_std
        mu_parts = [self.long_mu(h), self.lat_mu(h)]
        log_std_parts = [
            torch.clamp(self.long_log_std(h), LOG_STD_MIN, LOG_STD_MAX),
            torch.clamp(self.lat_log_std(h), LOG_STD_MIN, LOG_STD_MAX),
        ]
        if self.extra_mu is not None and self.extra_log_std is not None:
            mu_parts.append(self.extra_mu(h))
            log_std_parts.append(torch.clamp(self.extra_log_std(h), LOG_STD_MIN, LOG_STD_MAX))
        mu = torch.cat(mu_parts, dim=-1)
        log_std = torch.cat(log_std_parts, dim=-1)
        return mu, log_std

    def sample(self, state_seq: torch.Tensor, state_mask: Optional[torch.Tensor] = None):
        mu, log_std = self(state_seq, state_mask=state_mask)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        unit_action = torch.tanh(z)
        action = self.action_bias + self.action_scale * unit_action
        log_prob = dist.log_prob(z) - torch.log(1.0 - unit_action.pow(2) + 1e-6)
        log_prob = log_prob - torch.log(self.action_scale + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mu_action = self.action_bias + self.action_scale * torch.tanh(mu)
        return action, log_prob, mu_action


class QuantileSafetyCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, recurrent: bool, n_quantiles: int):
        super().__init__()
        self.n_quantiles = int(max(2, n_quantiles))
        self.encoder = SequenceEncoder(state_dim, hidden_dim, recurrent=recurrent)
        self.body = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_quantiles),
        )

    def forward(self, state_seq: torch.Tensor, action: torch.Tensor, state_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.encoder(state_seq, state_mask)
        return self.body(torch.cat([z, action], dim=-1))

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config, action_low, action_high):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)
        self.gamma = float(cfg.sac.gamma)
        self.tau = float(cfg.sac.tau)
        self.batch_size = int(cfg.sac.batch_size)
        self.target_entropy = float(cfg.sac.target_entropy)
        self.actor = Actor(state_dim, action_dim, cfg.sac.hidden_dim, action_low, action_high).to(self.device)
        self.q1 = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.q2 = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.q1_target = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.q2_target = Critic(state_dim, action_dim, cfg.sac.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.sac.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.sac.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.sac.critic_lr)
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.sac.alpha_lr)
        self.replay = ReplayBuffer(state_dim, action_dim, cfg.sac.buffer_size, self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def reset_episode_context(self) -> None:
        return None

    def select_action(self, state, evaluate: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state_t)
            else:
                action, _, _ = self.actor.sample(state_t)
        return action.cpu().numpy()[0].astype(np.float32)

    def update(self) -> dict:
        if self.replay.size < self.batch_size:
            return {}
        batch = self.replay.sample(self.batch_size)
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            q_next = torch.min(self.q1_target(next_state, next_action), self.q2_target(next_state, next_action))
            target_q = reward + (1.0 - done) * self.gamma * (q_next - self.alpha.detach() * next_log_prob)

        q1_loss = F.mse_loss(self.q1(state, action), target_q)
        q2_loss = F.mse_loss(self.q2(state, action), target_q)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.sac.critic_grad_clip_norm)
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.sac.critic_grad_clip_norm)
        self.q2_opt.step()

        new_action, log_prob, _ = self.actor.sample(state)
        q_new = torch.min(self.q1(state, new_action), self.q2(state, new_action))
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.sac.actor_grad_clip_norm)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
        }

    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


class LagrangianConstrainedSACAgent(SACAgent):
    """Compact constrained SAC for the new goal-reaching CMDP.

    This agent is intentionally simpler than the legacy Stage4 agent. It follows a
    standard Lagrangian CMDP design: learn reward critics Q_r and cost critics Q_c,
    then optimize the actor with Q_r penalized by adaptive multipliers lambda_i Q_c_i.
    No rule shield, no recovery policy, no priority danger sampling, no recurrent/n-step nesting.
    """

    def __init__(self, state_dim: int, action_dim: int, cfg: Config, action_low, action_high):
        super().__init__(state_dim, action_dim, cfg, action_low, action_high)
        self.cost_critics = nn.ModuleDict({
            "safety": Critic(state_dim, action_dim, cfg.sac.hidden_dim),
            "boundary": Critic(state_dim, action_dim, cfg.sac.hidden_dim),
            "speed": Critic(state_dim, action_dim, cfg.sac.hidden_dim),
        }).to(self.device)
        self.cost_targets = nn.ModuleDict({
            name: Critic(state_dim, action_dim, cfg.sac.hidden_dim)
            for name in self.cost_critics.keys()
        }).to(self.device)
        for name in self.cost_critics.keys():
            self.cost_targets[name].load_state_dict(self.cost_critics[name].state_dict())
        self.cost_opts = {
            name: torch.optim.Adam(self.cost_critics[name].parameters(), lr=cfg.sac.critic_lr)
            for name in self.cost_critics.keys()
        }
        self.lambdas = {
            "safety": torch.tensor(float(cfg.sac.lambda_init), dtype=torch.float32, device=self.device),
            "boundary": torch.tensor(float(cfg.sac.lambda_init), dtype=torch.float32, device=self.device),
            "speed": torch.tensor(float(cfg.sac.lambda_init), dtype=torch.float32, device=self.device),
        }
        self.budgets = {
            "safety": float(cfg.sac.constraint_budget_safety),
            "boundary": float(cfg.sac.constraint_budget_boundary),
            "speed": float(cfg.sac.constraint_budget_speed),
        }
        self.lambda_lr = float(cfg.sac.constraint_lambda_lr)
        self.lambda_max = float(cfg.sac.constraint_lambda_max)

    def _cost_upper_bound(self) -> float:
        return float(1.0 / max(1.0 - self.gamma, 1e-6))

    def _average_cost_q(self, value: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.gamma) * torch.clamp(value, min=0.0, max=self._cost_upper_bound())

    def _update_cost_critic(self, name: str, state, action, done, next_state, next_action, immediate_cost) -> torch.Tensor:
        critic = self.cost_critics[name]
        target = self.cost_targets[name]
        opt = self.cost_opts[name]
        with torch.no_grad():
            next_cost_q = target(next_state, next_action)
            target_cost_q = immediate_cost + (1.0 - done) * self.gamma * next_cost_q
            target_cost_q = torch.clamp(target_cost_q, min=0.0, max=self._cost_upper_bound())
        pred = critic(state, action)
        loss = F.mse_loss(pred, target_cost_q)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.cfg.sac.critic_grad_clip_norm)
        opt.step()
        return loss

    def update(self) -> dict:
        if self.replay.size < self.batch_size:
            return {}
        batch = self.replay.sample(self.batch_size)
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            q_next = torch.min(self.q1_target(next_state, next_action), self.q2_target(next_state, next_action))
            target_q = reward + (1.0 - done) * self.gamma * (q_next - self.alpha.detach() * next_log_prob)

        q1_loss = F.mse_loss(self.q1(state, action), target_q)
        q2_loss = F.mse_loss(self.q2(state, action), target_q)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.sac.critic_grad_clip_norm)
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.sac.critic_grad_clip_norm)
        self.q2_opt.step()

        cost_losses = {
            "safety": self._update_cost_critic("safety", state, action, done, next_state, next_action, batch["cost_safety"]),
            "boundary": self._update_cost_critic("boundary", state, action, done, next_state, next_action, batch["cost_lane"]),
            "speed": self._update_cost_critic("speed", state, action, done, next_state, next_action, batch["cost_overspeed"]),
        }

        new_action, log_prob, _ = self.actor.sample(state)
        q_new = torch.min(self.q1(state, new_action), self.q2(state, new_action))
        cost_q = {
            name: self._average_cost_q(self.cost_critics[name](state, new_action))
            for name in self.cost_critics.keys()
        }
        lagrangian_penalty = sum(self.lambdas[name].detach() * cost_q[name] for name in self.cost_critics.keys())
        actor_loss = (self.alpha.detach() * log_prob - q_new + lagrangian_penalty).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.sac.actor_grad_clip_norm)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for name in self.cost_critics.keys():
                violation = cost_q[name].mean().detach() - float(self.budgets[name])
                self.lambdas[name] = torch.clamp(
                    self.lambdas[name] + self.lambda_lr * violation,
                    min=0.0,
                    max=self.lambda_max,
                )

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        for name in self.cost_critics.keys():
            self._soft_update(self.cost_critics[name], self.cost_targets[name])

        out = {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
            "fixed_actor_penalty": float(lagrangian_penalty.mean().item()),
            "lambda_safety": float(self.lambdas["safety"].item()),
            "lambda_boundary": float(self.lambdas["boundary"].item()),
            "lambda_speed": float(self.lambdas["speed"].item()),
            "constraint_q_safety": float(cost_q["safety"].mean().item()),
            "constraint_q_boundary": float(cost_q["boundary"].mean().item()),
            "constraint_q_speed": float(cost_q["speed"].mean().item()),
            "mean_cost_safety": float(batch["cost_safety"].mean().item()),
            "batch_cost_safety": float(batch["cost_safety"].mean().item()),
            "batch_cost_boundary": float(batch["cost_lane"].mean().item()),
            "batch_cost_speed": float(batch["cost_overspeed"].mean().item()),
        }
        for name, loss in cost_losses.items():
            out[f"cost_{name}_loss"] = float(loss.item())
            out[f"cost_{name}_q"] = float(cost_q[name].mean().item())
        return out


class LegacyPenaltySACAgent(SACAgent):
    def __init__(self, state_dim: int, action_dim: int, cfg: Config, action_low, action_high):
        super().__init__(state_dim, action_dim, cfg, action_low, action_high)
        self.cost_critics = nn.ModuleDict({
            "collision": Critic(state_dim, action_dim, cfg.sac.hidden_dim),
            "headway": Critic(state_dim, action_dim, cfg.sac.hidden_dim),
            "overspeed": Critic(state_dim, action_dim, cfg.sac.hidden_dim),
            "comfort": Critic(state_dim, action_dim, cfg.sac.hidden_dim),
            "lane": Critic(state_dim, action_dim, cfg.sac.hidden_dim),
        }).to(self.device)
        self.cost_targets = nn.ModuleDict({
            name: Critic(state_dim, action_dim, cfg.sac.hidden_dim) for name in self.cost_critics.keys()
        }).to(self.device)
        for name in self.cost_critics.keys():
            self.cost_targets[name].load_state_dict(self.cost_critics[name].state_dict())
        self.cost_opts = {
            name: torch.optim.Adam(self.cost_critics[name].parameters(), lr=cfg.sac.critic_lr)
            for name in self.cost_critics.keys()
        }
        self.fixed_penalties = {
            "collision": float(cfg.sac.penalty_collision),
            "headway": float(cfg.sac.penalty_headway),
            "overspeed": float(cfg.sac.penalty_overspeed),
            "comfort": float(cfg.sac.penalty_comfort),
            "lane": float(cfg.sac.penalty_lane),
        }

    def _cost_upper_bound(self) -> float:
        return float(1.0 / max(1.0 - self.gamma, 1e-6))

    def _average_cost_q(self, value: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.gamma) * torch.clamp(value, min=0.0, max=self._cost_upper_bound())

    def _update_cost_critic(self, name: str, state, action, done, next_state, next_action, immediate_cost):
        critic = self.cost_critics[name]
        target = self.cost_targets[name]
        opt = self.cost_opts[name]
        with torch.no_grad():
            next_cost = target(next_state, next_action)
            target_cost = immediate_cost + (1.0 - done) * self.gamma * next_cost
            target_cost = torch.clamp(target_cost, min=0.0, max=self._cost_upper_bound())
        loss = F.mse_loss(critic(state, action), target_cost)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.cfg.sac.critic_grad_clip_norm)
        opt.step()
        return loss

    def update(self) -> dict:
        if self.replay.size < self.batch_size:
            return {}
        batch = self.replay.sample(self.batch_size)
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            q_next = torch.min(self.q1_target(next_state, next_action), self.q2_target(next_state, next_action))
            target_q = reward + (1.0 - done) * self.gamma * (q_next - self.alpha.detach() * next_log_prob)

        q1_loss = F.mse_loss(self.q1(state, action), target_q)
        q2_loss = F.mse_loss(self.q2(state, action), target_q)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.sac.critic_grad_clip_norm)
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.sac.critic_grad_clip_norm)
        self.q2_opt.step()

        cost_losses = {
            "collision": self._update_cost_critic("collision", state, action, done, next_state, next_action, batch["cost_collision"]),
            "headway": self._update_cost_critic("headway", state, action, done, next_state, next_action, batch["cost_headway"]),
            "overspeed": self._update_cost_critic("overspeed", state, action, done, next_state, next_action, batch["cost_overspeed"]),
            "comfort": self._update_cost_critic("comfort", state, action, done, next_state, next_action, batch["cost_comfort"]),
            "lane": self._update_cost_critic("lane", state, action, done, next_state, next_action, batch["cost_lane"]),
        }

        new_action, log_prob, _ = self.actor.sample(state)
        q_new = torch.min(self.q1(state, new_action), self.q2(state, new_action))
        cost_estimates = {
            name: self._average_cost_q(self.cost_critics[name](state, new_action))
            for name in self.cost_critics.keys()
        }
        fixed_actor_penalty = sum(self.fixed_penalties[name] * cost_estimates[name] for name in self.cost_critics.keys())
        actor_loss = (self.alpha.detach() * log_prob - q_new + fixed_actor_penalty).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.sac.actor_grad_clip_norm)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        for name in self.cost_critics.keys():
            self._soft_update(self.cost_critics[name], self.cost_targets[name])

        out = {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
            "fixed_actor_penalty": float(fixed_actor_penalty.mean().item()),
        }
        for name, loss in cost_losses.items():
            out[f"cost_{name}_loss"] = float(loss.item())
            out[f"cost_{name}_q"] = float(cost_estimates[name].mean().item())
            out[f"batch_cost_{name}"] = float(batch[f"cost_{name}"].mean().item())
        return out


class Stage4PenaltySACAgent(SACAgent):
    def __init__(self, state_dim: int, action_dim: int, cfg: Config, action_low, action_high):
        super().__init__(state_dim, action_dim, cfg, action_low, action_high)
        self.recurrent_enabled = bool(cfg.sac.enable_recurrent_encoder)
        self.decoupled_actor_enabled = bool(cfg.sac.enable_decoupled_actor)
        self.seq_len = max(1, int(cfg.sac.seq_len))
        self.actor_update_interval = max(1, int(cfg.sac.actor_update_interval))
        self.safety_budget_collision = float(cfg.sac.safety_budget_collision)
        self.safety_budget_headway = float(cfg.sac.safety_budget_headway)
        self.lambda_lr = float(cfg.sac.lambda_lr)
        self.lambda_max = float(cfg.sac.lambda_max)
        self.lambda_collision = torch.tensor(float(cfg.sac.lambda_init), dtype=torch.float32, device=self.device)
        self.lambda_headway = torch.tensor(float(cfg.sac.lambda_init), dtype=torch.float32, device=self.device)
        self.enable_lagrangian = bool(cfg.sac.enable_lagrangian_safety)
        self.enable_tail_risk_cvar = bool(cfg.sac.enable_tail_risk_cvar)
        self.cvar_alpha = float(np.clip(float(cfg.sac.cvar_alpha), 1e-3, 0.99))
        self.update_counter = 0
        self.alpha_min = max(0.0, float(cfg.sac.alpha_min))
        self.actor_decouple_mode = str(cfg.sac.actor_decouple_mode)
        self.fixed_collision_penalty = float(cfg.sac.penalty_collision)
        self.fixed_headway_penalty = float(cfg.sac.penalty_headway)
        self.soft_penalties = {
            "overspeed": float(cfg.sac.penalty_overspeed),
            "comfort": float(cfg.sac.penalty_comfort),
            "lane": float(cfg.sac.penalty_lane),
        }
        self.soft_long_weight = 1.0
        self.soft_lat_weight = 0.60
        self.priority_enabled = bool(cfg.sac.enable_priority_safety_replay)
        self.safety_n_step = max(1, int(cfg.sac.safety_n_step))
        self.danger_ratio = float(np.clip(cfg.sac.danger_ratio, 0.0, 1.0))
        self.near_danger_ratio = float(np.clip(cfg.sac.near_danger_ratio, 0.0, 1.0 - self.danger_ratio))
        self._act_state_buffer: deque[np.ndarray] = deque(maxlen=self.seq_len)
        model_hidden = int(cfg.sac.gru_hidden_dim if self.recurrent_enabled else cfg.sac.hidden_dim)

        self.actor = Stage4Actor(
            state_dim,
            action_dim,
            model_hidden,
            action_low,
            action_high,
            recurrent=self.recurrent_enabled,
            decoupled=self.decoupled_actor_enabled,
        ).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.sac.actor_lr)

        self.q1 = SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled).to(self.device)
        self.q2 = SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled).to(self.device)
        self.q1_target = SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled).to(self.device)
        self.q2_target = SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.sac.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.sac.critic_lr)

        self.cost_critics = nn.ModuleDict(
            {
                "collision": SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled),
                "headway": SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled),
                "overspeed": SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled),
                "comfort": SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled),
                "lane": SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled),
            }
        ).to(self.device)
        self.cost_targets = nn.ModuleDict(
            {
                name: SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled)
                for name in self.cost_critics.keys()
            }
        ).to(self.device)
        for name in self.cost_critics.keys():
            self.cost_targets[name].load_state_dict(self.cost_critics[name].state_dict())
        self.cost_opts = {
            name: torch.optim.Adam(self.cost_critics[name].parameters(), lr=cfg.sac.critic_lr)
            for name in self.cost_critics.keys()
        }

        self.constraint_q_collision = SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled).to(self.device)
        self.constraint_q_headway = SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled).to(self.device)
        self.constraint_q_collision_target = SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled).to(self.device)
        self.constraint_q_headway_target = SequenceCritic(state_dim, action_dim, model_hidden, recurrent=self.recurrent_enabled).to(self.device)
        self.constraint_q_collision_target.load_state_dict(self.constraint_q_collision.state_dict())
        self.constraint_q_headway_target.load_state_dict(self.constraint_q_headway.state_dict())
        self.constraint_q_collision_opt = torch.optim.Adam(self.constraint_q_collision.parameters(), lr=cfg.sac.critic_lr)
        self.constraint_q_headway_opt = torch.optim.Adam(self.constraint_q_headway.parameters(), lr=cfg.sac.critic_lr)

        self.safety_quantiles = None
        self.safety_quantiles_target = None
        self.safety_quantiles_opt = None
        if self.enable_tail_risk_cvar:
            self.safety_quantiles = QuantileSafetyCritic(
                state_dim,
                action_dim,
                model_hidden,
                recurrent=self.recurrent_enabled,
                n_quantiles=max(2, int(cfg.sac.cvar_quantiles)),
            ).to(self.device)
            self.safety_quantiles_target = QuantileSafetyCritic(
                state_dim,
                action_dim,
                model_hidden,
                recurrent=self.recurrent_enabled,
                n_quantiles=max(2, int(cfg.sac.cvar_quantiles)),
            ).to(self.device)
            self.safety_quantiles_target.load_state_dict(self.safety_quantiles.state_dict())
            self.safety_quantiles_opt = torch.optim.Adam(self.safety_quantiles.parameters(), lr=cfg.sac.critic_lr)
            self.quantile_taus = (torch.arange(max(2, int(cfg.sac.cvar_quantiles)), device=self.device, dtype=torch.float32) + 0.5) / float(
                max(2, int(cfg.sac.cvar_quantiles))
            )

    def reset_episode_context(self) -> None:
        self._act_state_buffer.clear()

    def _cost_upper_bound(self) -> float:
        return float(1.0 / max(1.0 - self.gamma, 1e-6))

    def _average_cost_q(self, value: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.gamma) * torch.clamp(value, min=0.0, max=self._cost_upper_bound())

    def _build_action_sequence(self, state: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        vec = np.asarray(state, dtype=np.float32).reshape(-1)
        self._act_state_buffer.append(vec.copy())
        seq = np.zeros((self.seq_len, vec.shape[0]), dtype=np.float32)
        mask = np.zeros((self.seq_len,), dtype=np.float32)
        offset = self.seq_len - len(self._act_state_buffer)
        for i, s in enumerate(self._act_state_buffer):
            seq[offset + i] = s
            mask[offset + i] = 1.0
        seq_t = torch.as_tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        return seq_t, mask_t

    def select_action(self, state, evaluate: bool = False) -> np.ndarray:
        seq_t, mask_t = self._build_action_sequence(np.asarray(state, dtype=np.float32))
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(seq_t, mask_t)
            else:
                action, _, _ = self.actor.sample(seq_t, mask_t)
        return action.cpu().numpy()[0].astype(np.float32)

    def _update_cost_critic(self, name: str, state_seq, state_mask, action, done, next_state_seq, next_state_mask, next_action, immediate_cost):
        critic = self.cost_critics[name]
        target = self.cost_targets[name]
        opt = self.cost_opts[name]
        with torch.no_grad():
            next_cost = target(next_state_seq, next_action, next_state_mask)
            target_cost = immediate_cost + (1.0 - done) * self.gamma * next_cost
            target_cost = torch.clamp(target_cost, min=0.0, max=self._cost_upper_bound())
        loss = F.mse_loss(critic(state_seq, action, state_mask), target_cost)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.cfg.sac.critic_grad_clip_norm)
        opt.step()
        return loss

    def _update_constraint_critic(
        self,
        name: str,
        state_seq,
        state_mask,
        action,
        bootstrap_state_seq,
        bootstrap_state_mask,
        bootstrap_done,
        bootstrap_discount,
        nstep_cost,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if name == "collision":
            critic = self.constraint_q_collision
            target = self.constraint_q_collision_target
            opt = self.constraint_q_collision_opt
        elif name == "headway":
            critic = self.constraint_q_headway
            target = self.constraint_q_headway_target
            opt = self.constraint_q_headway_opt
        else:
            raise ValueError(f"Unsupported constraint critic name: {name}")

        with torch.no_grad():
            bootstrap_action, _, _ = self.actor.sample(bootstrap_state_seq, bootstrap_state_mask)
            bootstrap_q = target(bootstrap_state_seq, bootstrap_action, bootstrap_state_mask)
            target_cost = nstep_cost + (1.0 - bootstrap_done) * bootstrap_discount * bootstrap_q
            target_cost = torch.clamp(target_cost, min=0.0, max=self._cost_upper_bound())
        pred = critic(state_seq, action, state_mask)
        loss = F.mse_loss(pred, target_cost)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.cfg.sac.critic_grad_clip_norm)
        opt.step()
        return loss, target_cost.mean()

    def _quantile_huber_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        taus = self.quantile_taus.view(1, -1, 1)
        diff = target.unsqueeze(1) - pred.unsqueeze(2)
        abs_diff = torch.abs(diff)
        huber = torch.where(abs_diff <= 1.0, 0.5 * diff.pow(2), abs_diff - 0.5)
        weight = torch.abs(taus - (diff.detach() < 0.0).float())
        return (weight * huber).mean()

    def _tail_cvar(self, quantiles: torch.Tensor) -> torch.Tensor:
        n_q = int(quantiles.shape[-1])
        k = max(1, int(math.ceil(float(self.cvar_alpha) * float(n_q))))
        sorted_vals, _ = torch.sort(quantiles, dim=-1, descending=True)
        tail = sorted_vals[:, :k]
        tail = torch.clamp(tail, min=0.0, max=self._cost_upper_bound())
        return (1.0 - self.gamma) * tail.mean(dim=-1, keepdim=True)

    def _update_tail_risk(self, state_seq, state_mask, action, done, next_state_seq, next_state_mask, next_action, immediate_safety):
        if self.safety_quantiles is None or self.safety_quantiles_target is None or self.safety_quantiles_opt is None:
            return torch.tensor(0.0, device=self.device)
        pred = self.safety_quantiles(state_seq, action, state_mask)
        with torch.no_grad():
            next_q = self.safety_quantiles_target(next_state_seq, next_action, next_state_mask)
            target = immediate_safety + (1.0 - done) * self.gamma * next_q
            target = torch.clamp(target, min=0.0, max=self._cost_upper_bound())
        loss = self._quantile_huber_loss(pred, target)
        self.safety_quantiles_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.safety_quantiles.parameters(), self.cfg.sac.critic_grad_clip_norm)
        self.safety_quantiles_opt.step()
        return loss

    def update(self) -> dict:
        if self.replay.size < self.batch_size:
            return {}
        batch = self.replay.sample(
            self.batch_size,
            sample_cfg={
                "enabled": bool(self.priority_enabled),
                "danger_ratio": float(self.danger_ratio),
                "near_ratio": float(self.near_danger_ratio),
                "seq_len": int(self.seq_len),
                "safety_n_step": int(self.safety_n_step),
                "gamma": float(self.gamma),
                "safety_w_collision": float(self.cfg.sac.safety_w_collision),
                "safety_w_headway": float(self.cfg.sac.safety_w_headway),
            },
        )
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]
        state_seq = batch["state_seq"]
        next_state_seq = batch["next_state_seq"]
        state_mask = batch["state_seq_mask"]
        next_state_mask = batch["next_state_seq_mask"]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state_seq, next_state_mask)
            q_next = torch.min(
                self.q1_target(next_state_seq, next_action, next_state_mask),
                self.q2_target(next_state_seq, next_action, next_state_mask),
            )
            target_q = reward + (1.0 - done) * self.gamma * (q_next - self.alpha.detach() * next_log_prob)
        q1_loss = F.mse_loss(self.q1(state_seq, action, state_mask), target_q)
        q2_loss = F.mse_loss(self.q2(state_seq, action, state_mask), target_q)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.sac.critic_grad_clip_norm)
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.sac.critic_grad_clip_norm)
        self.q2_opt.step()

        cost_losses = {
            "collision": self._update_cost_critic(
                "collision", state_seq, state_mask, action, done, next_state_seq, next_state_mask, next_action, batch["cost_collision"]
            ),
            "headway": self._update_cost_critic(
                "headway", state_seq, state_mask, action, done, next_state_seq, next_state_mask, next_action, batch["cost_headway"]
            ),
            "overspeed": self._update_cost_critic(
                "overspeed", state_seq, state_mask, action, done, next_state_seq, next_state_mask, next_action, batch["cost_overspeed"]
            ),
            "comfort": self._update_cost_critic(
                "comfort", state_seq, state_mask, action, done, next_state_seq, next_state_mask, next_action, batch["cost_comfort"]
            ),
            "lane": self._update_cost_critic(
                "lane", state_seq, state_mask, action, done, next_state_seq, next_state_mask, next_action, batch["cost_lane"]
            ),
        }
        constraint_collision_loss, target_collision_bootstrap_mean = self._update_constraint_critic(
            "collision",
            state_seq,
            state_mask,
            action,
            batch["constraint_bootstrap_state_seq"],
            batch["constraint_bootstrap_state_mask"],
            batch["constraint_bootstrap_done"],
            batch["constraint_bootstrap_discount"],
            batch["cost_collision_nstep"],
        )
        constraint_headway_loss, target_headway_bootstrap_mean = self._update_constraint_critic(
            "headway",
            state_seq,
            state_mask,
            action,
            batch["constraint_bootstrap_state_seq"],
            batch["constraint_bootstrap_state_mask"],
            batch["constraint_bootstrap_done"],
            batch["constraint_bootstrap_discount"],
            batch["cost_headway_nstep"],
        )
        tail_loss = self._update_tail_risk(
            state_seq,
            state_mask,
            action,
            done,
            next_state_seq,
            next_state_mask,
            next_action,
            batch["cost_safety"],
        )

        self.update_counter += 1
        actor_loss = torch.tensor(float("nan"), device=self.device)
        alpha_loss = torch.tensor(float("nan"), device=self.device)
        dual_loss = torch.tensor(float("nan"), device=self.device)
        safety_q_avg_mean = torch.tensor(0.0, device=self.device)
        constraint_q_collision_mean = torch.tensor(0.0, device=self.device)
        constraint_q_headway_mean = torch.tensor(0.0, device=self.device)
        safety_q_cvar_mean = torch.tensor(float("nan"), device=self.device)
        fixed_actor_penalty = torch.tensor(0.0, device=self.device)

        if self.update_counter % self.actor_update_interval == 0:
            new_action, log_prob, _ = self.actor.sample(state_seq, state_mask)
            actor_eval_action = new_action
            q_new = torch.min(
                self.q1(state_seq, new_action, state_mask),
                self.q2(state_seq, new_action, state_mask),
            )
            cost_estimates = {
                name: self._average_cost_q(self.cost_critics[name](state_seq, new_action, state_mask))
                for name in self.cost_critics.keys()
            }
            overspeed_q = cost_estimates["overspeed"]
            comfort_q = cost_estimates["comfort"]
            lane_q = cost_estimates["lane"]
            if self.decoupled_actor_enabled and self.actor_decouple_mode == "weighted":
                long_soft = overspeed_q + 0.5 * comfort_q
                lat_soft = lane_q + 0.25 * comfort_q
                soft_penalty = self.soft_long_weight * long_soft + self.soft_lat_weight * lat_soft
            elif self.decoupled_actor_enabled and self.actor_decouple_mode == "alternating_stopgrad":
                if self.update_counter % 2 == 0:
                    action_long = torch.cat([new_action[:, 0:1], new_action[:, 1:2].detach(), new_action[:, 2:].detach()], dim=-1)
                    actor_eval_action = action_long
                    overspeed_long = self._average_cost_q(self.cost_critics["overspeed"](state_seq, action_long, state_mask))
                    comfort_long = self._average_cost_q(self.cost_critics["comfort"](state_seq, action_long, state_mask))
                    q_new = torch.min(self.q1(state_seq, action_long, state_mask), self.q2(state_seq, action_long, state_mask))
                    soft_penalty = self.soft_long_weight * (overspeed_long + 0.5 * comfort_long)
                else:
                    action_lat = torch.cat([new_action[:, 0:1].detach(), new_action[:, 1:2], new_action[:, 2:].detach()], dim=-1)
                    actor_eval_action = action_lat
                    lane_lat = self._average_cost_q(self.cost_critics["lane"](state_seq, action_lat, state_mask))
                    comfort_lat = self._average_cost_q(self.cost_critics["comfort"](state_seq, action_lat, state_mask))
                    q_new = torch.min(self.q1(state_seq, action_lat, state_mask), self.q2(state_seq, action_lat, state_mask))
                    soft_penalty = self.soft_lat_weight * (lane_lat + 0.25 * comfort_lat)
            else:
                soft_penalty = (
                    self.soft_penalties["overspeed"] * overspeed_q
                    + self.soft_penalties["comfort"] * comfort_q
                    + self.soft_penalties["lane"] * lane_q
                )

            constraint_q_collision = self._average_cost_q(self.constraint_q_collision(state_seq, actor_eval_action, state_mask))
            constraint_q_headway = self._average_cost_q(self.constraint_q_headway(state_seq, actor_eval_action, state_mask))
            constraint_q_collision_mean = constraint_q_collision.mean()
            constraint_q_headway_mean = constraint_q_headway.mean()
            safety_q_avg_mean = 0.5 * (constraint_q_collision_mean + constraint_q_headway_mean)

            if self.enable_lagrangian:
                lambda_collision_used = self.lambda_collision.detach()
                lambda_headway_used = self.lambda_headway.detach()
            else:
                lambda_collision_used = torch.tensor(float(self.fixed_collision_penalty), dtype=torch.float32, device=self.device)
                lambda_headway_used = torch.tensor(float(self.fixed_headway_penalty), dtype=torch.float32, device=self.device)

            fixed_actor_penalty = (
                lambda_collision_used * constraint_q_collision
                + lambda_headway_used * constraint_q_headway
                + soft_penalty
            )
            if self.enable_tail_risk_cvar and self.safety_quantiles is not None:
                tail_pred = self.safety_quantiles(state_seq, actor_eval_action, state_mask)
                tail_cvar = self._tail_cvar(tail_pred)
                safety_q_cvar_mean = tail_cvar.mean()
                lambda_tail = 0.5 * (lambda_collision_used + lambda_headway_used)
                fixed_actor_penalty = fixed_actor_penalty + lambda_tail * tail_cvar

            actor_loss = (self.alpha.detach() * log_prob - q_new + fixed_actor_penalty).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.sac.actor_grad_clip_norm)
            self.actor_opt.step()

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                alpha_clamped = torch.clamp(self.log_alpha.exp(), min=float(self.alpha_min))
                self.log_alpha.copy_(torch.log(alpha_clamped))

            if self.enable_lagrangian:
                violation_collision = constraint_q_collision_mean.detach() - float(self.safety_budget_collision)
                violation_headway = constraint_q_headway_mean.detach() - float(self.safety_budget_headway)
                dual_loss = self.lambda_collision * violation_collision + self.lambda_headway * violation_headway
                self.lambda_collision = torch.clamp(
                    self.lambda_collision + float(self.lambda_lr) * violation_collision.detach(),
                    min=0.0,
                    max=float(self.lambda_max),
                )
                self.lambda_headway = torch.clamp(
                    self.lambda_headway + float(self.lambda_lr) * violation_headway.detach(),
                    min=0.0,
                    max=float(self.lambda_max),
                )

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self._soft_update(self.constraint_q_collision, self.constraint_q_collision_target)
        self._soft_update(self.constraint_q_headway, self.constraint_q_headway_target)
        if self.enable_tail_risk_cvar and self.safety_quantiles is not None and self.safety_quantiles_target is not None:
            self._soft_update(self.safety_quantiles, self.safety_quantiles_target)
        for name in self.cost_critics.keys():
            self._soft_update(self.cost_critics[name], self.cost_targets[name])

        lambda_collision_value = float(self.lambda_collision.item()) if self.enable_lagrangian else float(self.fixed_collision_penalty)
        lambda_headway_value = float(self.lambda_headway.item()) if self.enable_lagrangian else float(self.fixed_headway_penalty)
        out = {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()) if torch.isfinite(actor_loss) else float("nan"),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()) if torch.isfinite(alpha_loss) else float("nan"),
            "fixed_actor_penalty": float(fixed_actor_penalty.mean().item()),
            "lambda_safety": float(0.5 * (lambda_collision_value + lambda_headway_value)),
            "lambda_collision": float(lambda_collision_value),
            "lambda_headway": float(lambda_headway_value),
            "dual_loss": float(dual_loss.item()) if torch.isfinite(dual_loss) else float("nan"),
            "safety_q_avg": float(safety_q_avg_mean.item()),
            "safety_q1": float(constraint_q_collision_mean.item()),
            "safety_q2": float(constraint_q_headway_mean.item()),
            "constraint_q_collision": float(constraint_q_collision_mean.item()),
            "constraint_q_headway": float(constraint_q_headway_mean.item()),
            "safety_q_cvar": float(safety_q_cvar_mean.item()) if torch.isfinite(safety_q_cvar_mean) else float("nan"),
            "safety_q1_loss": float(constraint_collision_loss.item()),
            "safety_q2_loss": float(constraint_headway_loss.item()),
            "constraint_q_collision_loss": float(constraint_collision_loss.item()),
            "constraint_q_headway_loss": float(constraint_headway_loss.item()),
            "target_collision_nstep_bootstrap": float(target_collision_bootstrap_mean.item()),
            "target_headway_nstep_bootstrap": float(target_headway_bootstrap_mean.item()),
            "safety_tail_loss": float(tail_loss.item()) if torch.isfinite(tail_loss) else 0.0,
            "mean_cost_safety": float(batch["cost_safety"].mean().item()),
            "sampled_success_count": float(batch["sampled_success_count"]),
            "sampled_near_danger_count": float(batch["sampled_near_danger_count"]),
            "sampled_collision_count": float(batch["sampled_collision_count"]),
            "sampled_danger_ratio_actual": float(batch["sampled_danger_ratio_actual"]),
            "sampled_near_ratio_actual": float(batch["sampled_near_ratio_actual"]),
            "replay_danger_count": float(batch["replay_danger_count"]),
            "replay_near_danger_count": float(batch["replay_near_danger_count"]),
        }
        for name, loss in cost_losses.items():
            out[f"cost_{name}_loss"] = float(loss.item())
        out["cost_collision_q"] = float(self._average_cost_q(self.cost_critics["collision"](state_seq, action, state_mask)).mean().item())
        out["cost_headway_q"] = float(self._average_cost_q(self.cost_critics["headway"](state_seq, action, state_mask)).mean().item())
        out["cost_overspeed_q"] = float(self._average_cost_q(self.cost_critics["overspeed"](state_seq, action, state_mask)).mean().item())
        out["cost_comfort_q"] = float(self._average_cost_q(self.cost_critics["comfort"](state_seq, action, state_mask)).mean().item())
        out["cost_lane_q"] = float(self._average_cost_q(self.cost_critics["lane"](state_seq, action, state_mask)).mean().item())
        out["batch_cost_collision"] = float(batch["cost_collision"].mean().item())
        out["batch_cost_headway"] = float(batch["cost_headway"].mean().item())
        out["batch_cost_overspeed"] = float(batch["cost_overspeed"].mean().item())
        out["batch_cost_comfort"] = float(batch["cost_comfort"].mean().item())
        out["batch_cost_lane"] = float(batch["cost_lane"].mean().item())
        out["batch_cost_safety"] = float(batch["cost_safety"].mean().item())
        return out


def build_agent(state_dim: int, action_dim: int, cfg: Config, action_low, action_high):
    configure_mode_behavior(cfg, cfg.train.mode)
    if cfg.train.mode in {"goal_csac_lagrangian", "paper_csac_lagrangian"}:
        return LagrangianConstrainedSACAgent(state_dim, action_dim, cfg, action_low, action_high)
    if cfg.train.mode in {"goal_sac_pure", "paper_sac_pure"}:
        return SACAgent(state_dim, action_dim, cfg, action_low, action_high)
    raise ValueError(f"Unsupported mode: {cfg.train.mode}")


class EpisodeMetrics:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rewards: List[float] = []
        self.env_rewards: List[float] = []
        self.speeds: List[float] = []
        self.front_distances: List[float] = []
        self.ttcs: List[float] = []
        self.abs_acc: List[float] = []
        self.abs_steer: List[float] = []
        self.smooth_acc: List[float] = []
        self.smooth_steer: List[float] = []
        self.lane_offsets: List[float] = []
        self.heading_errors: List[float] = []
        self.road_margins: List[float] = []
        self.lane_changes = 0
        self.collided = False
        self.failed_offroad = False
        self.goal_reached = False
        self.goal_distance_remaining: List[float] = []
        self.goal_progress: List[float] = []
        self.steps = 0
        self.first_collision_step: Optional[int] = None
        self.unsafe_headway_count = 0
        self.overspeed_count = 0
        self.low_ttc_count = 0
        self.offroad_count = 0
        self.long_actions: List[float] = []
        self.lat_actions: List[float] = []
        self.cost_collision: List[float] = []
        self.cost_headway: List[float] = []
        self.cost_overspeed: List[float] = []
        self.cost_comfort: List[float] = []
        self.cost_lane: List[float] = []
        self.cost_total: List[float] = []
        self.cost_safety: List[float] = []
        self.cost_collision_front: List[float] = []
        self.cost_collision_rear: List[float] = []
        self.cost_collision_lateral: List[float] = []

    def update(self, scene: dict, next_scene: dict, action, reward: float, env_reward: float, cost_dict: dict) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        last_action = np.asarray(scene.get("last_action", np.zeros_like(action)), dtype=np.float32).reshape(-1)
        self.rewards.append(float(reward))
        self.env_rewards.append(float(env_reward))
        self.speeds.append(float(next_scene.get("ego_speed", 0.0)))
        self.front_distances.append(float(min(next_scene.get("front_distance", 1e6), 1000.0)))
        self.ttcs.append(float(min(next_scene.get("ttc", 99.0), 100.0)))
        self.abs_acc.append(float(abs(action[0])) if action.shape[0] > 0 else 0.0)
        self.abs_steer.append(float(abs(action[1])) if action.shape[0] > 1 else 0.0)
        self.long_actions.append(float(action[0]) if action.shape[0] > 0 else 0.0)
        self.lat_actions.append(float(action[1]) if action.shape[0] > 1 else 0.0)
        self.smooth_acc.append(float(abs(action[0] - last_action[0])) if action.shape[0] > 0 else 0.0)
        self.smooth_steer.append(float(abs(action[1] - last_action[1])) if action.shape[0] > 1 else 0.0)
        self.lane_offsets.append(float(next_scene.get("abs_lane_offset_norm", 0.0)))
        self.heading_errors.append(abs(float(next_scene.get("heading_error", 0.0))))
        self.road_margins.append(float(next_scene.get("road_boundary_margin", 0.0)))
        self.goal_distance_remaining.append(float(next_scene.get("goal_distance_remaining", float(self.cfg.env.goal_distance))))
        self.goal_progress.append(float(next_scene.get("goal_longitudinal_progress", 0.0)))
        if goal_reached_scene(self.cfg, next_scene):
            self.goal_reached = True
        if next_scene.get("lane_id", 0) != scene.get("lane_id", 0):
            self.lane_changes += 1
        if bool(next_scene.get("collision", False)):
            self.collided = True
            if self.first_collision_step is None:
                self.first_collision_step = int(self.steps)
        offroad_now = is_offroad_scene(self.cfg, next_scene)
        if offroad_now:
            self.failed_offroad = True
            self.offroad_count += 1
        self.unsafe_headway_count += int(float(next_scene.get("front_distance", 1e6)) < float(self.cfg.cost.min_headway))
        speed_limit_runtime = float(next_scene.get("speed_limit_runtime", self.cfg.cost.speed_limit))
        self.overspeed_count += int(float(next_scene.get("ego_speed", 0.0)) > speed_limit_runtime)
        self.low_ttc_count += int(float(next_scene.get("ttc", 99.0)) < float(self.cfg.cost.ttc_safe))
        self.cost_collision.append(float(cost_dict.get("collision_cost", 0.0)))
        self.cost_headway.append(float(cost_dict.get("headway_cost", 0.0)))
        self.cost_overspeed.append(float(cost_dict.get("overspeed_cost", 0.0)))
        self.cost_comfort.append(float(cost_dict.get("comfort_cost", 0.0)))
        self.cost_lane.append(float(cost_dict.get("lane_cost", 0.0)))
        self.cost_total.append(float(cost_dict.get("total_cost", 0.0)))
        self.cost_safety.append(float(cost_dict.get("safety_cost", 0.0)))
        self.cost_collision_front.append(float(cost_dict.get("collision_cost_front", 0.0)))
        self.cost_collision_rear.append(float(cost_dict.get("collision_cost_rear", 0.0)))
        self.cost_collision_lateral.append(float(cost_dict.get("collision_cost_lateral", 0.0)))
        self.steps += 1

    def summary(self) -> dict:
        ttc_arr = np.asarray(self.ttcs, dtype=np.float32) if self.ttcs else np.asarray([99.0], dtype=np.float32)
        ttc_p10 = float(np.percentile(ttc_arr, 10))
        ttc_p25 = float(np.percentile(ttc_arr, 25))
        ttc_p50 = float(np.percentile(ttc_arr, 50))
        ttc_p75 = float(np.percentile(ttc_arr, 75))
        ttc_p90 = float(np.percentile(ttc_arr, 90))
        profile_len = max(1, int(self.cfg.sac.danger_precollision_steps))
        pre_long = [0.0 for _ in range(profile_len)]
        pre_speed = [0.0 for _ in range(profile_len)]
        if self.first_collision_step is not None and self.first_collision_step > 0:
            end_idx = int(self.first_collision_step)
            start_idx = max(0, end_idx - profile_len)
            long_seg = self.long_actions[start_idx:end_idx]
            speed_seg = self.speeds[start_idx:end_idx]
            long_pad = [0.0 for _ in range(profile_len - len(long_seg))]
            speed_pad = [0.0 for _ in range(profile_len - len(speed_seg))]
            pre_long = long_pad + [float(v) for v in long_seg]
            pre_speed = speed_pad + [float(v) for v in speed_seg]
        success = bool(self.goal_reached and not self.collided and not self.failed_offroad)
        return {
            "episode_return": float(np.sum(self.rewards)),
            "episode_env_return": float(np.sum(self.env_rewards)),
            "steps": int(self.steps),
            "collision": int(self.collided),
            "goal_reached": int(self.goal_reached),
            "success": int(success),
            "collision_rate": float(int(self.collided)),
            "goal_reached_rate": float(int(self.goal_reached)),
            "success_rate": float(int(success)),
            "final_goal_distance_remaining": float(self.goal_distance_remaining[-1]) if self.goal_distance_remaining else float(self.cfg.env.goal_distance),
            "max_goal_progress": _safe_max(self.goal_progress),
            "mean_env_reward": _safe_mean(self.env_rewards),
            "mean_env_reward_per_step": _safe_mean(self.env_rewards),
            "mean_speed": _safe_mean(self.speeds),
            "mean_front_distance": _safe_mean(self.front_distances),
            "min_front_distance": _safe_min(self.front_distances),
            "mean_ttc": _safe_mean(self.ttcs),
            "min_ttc": _safe_min(self.ttcs),
            "ttc_p10": float(ttc_p10),
            "ttc_p25": float(ttc_p25),
            "ttc_p50": float(ttc_p50),
            "ttc_p75": float(ttc_p75),
            "ttc_p90": float(ttc_p90),
            "mean_abs_acc": _safe_mean(self.abs_acc),
            "mean_abs_steer": _safe_mean(self.abs_steer),
            "std_long_action": float(np.std(np.asarray(self.long_actions, dtype=np.float32))) if self.long_actions else 0.0,
            "std_lat_action": float(np.std(np.asarray(self.lat_actions, dtype=np.float32))) if self.lat_actions else 0.0,
            "mean_smooth_acc": _safe_mean(self.smooth_acc),
            "mean_smooth_steer": _safe_mean(self.smooth_steer),
            "mean_abs_lane_offset": _safe_mean(self.lane_offsets),
            "max_abs_lane_offset": _safe_max(self.lane_offsets),
            "mean_abs_heading_error": _safe_mean(self.heading_errors),
            "min_road_boundary_margin": _safe_min(self.road_margins),
            "lane_changes": int(self.lane_changes),
            "unsafe_headway_rate": float(self.unsafe_headway_count / max(self.steps, 1)),
            "overspeed_rate": float(self.overspeed_count / max(self.steps, 1)),
            "low_ttc_rate": float(self.low_ttc_count / max(self.steps, 1)),
            "offroad_rate": float(self.offroad_count / max(self.steps, 1)),
            "mean_cost_collision": _safe_mean(self.cost_collision),
            "collision_cost_front": _safe_mean(self.cost_collision_front),
            "collision_cost_rear": _safe_mean(self.cost_collision_rear),
            "collision_cost_lateral": _safe_mean(self.cost_collision_lateral),
            "mean_cost_headway": _safe_mean(self.cost_headway),
            "mean_cost_overspeed": _safe_mean(self.cost_overspeed),
            "mean_cost_comfort": _safe_mean(self.cost_comfort),
            "mean_cost_lane": _safe_mean(self.cost_lane),
            "mean_cost_total": _safe_mean(self.cost_total),
            "mean_cost_safety": _safe_mean(self.cost_safety),
            "precollision_long_action_mean_profile": pre_long,
            "precollision_speed_profile": pre_speed,
        }


TRAIN_DIAGNOSTIC_FIELDS = [
    "episode",
    "episode_return",
    "success",
    "collision",
    "goal_reached",
    "success_rate",
    "goal_reached_rate",
    "collision_rate",
    "final_goal_distance_remaining",
    "max_goal_progress",
    "mean_env_reward_per_step",
    "offroad_rate",
    "unsafe_headway_rate",
    "overspeed_rate",
    "low_ttc_rate",
    "mean_speed",
    "min_front_distance",
    "ttc_p10",
    "ttc_p25",
    "ttc_p50",
    "ttc_p75",
    "ttc_p90",
    "mean_cost_collision",
    "collision_cost_front",
    "collision_cost_rear",
    "collision_cost_lateral",
    "mean_cost_headway",
    "mean_cost_overspeed",
    "mean_cost_comfort",
    "mean_cost_lane",
    "mean_cost_total",
    "mean_cost_safety",
    "std_long_action",
    "std_lat_action",
    "q1_loss",
    "q2_loss",
    "actor_loss",
    "alpha",
    "alpha_loss",
    "fixed_actor_penalty",
    "lambda_safety",
    "lambda_boundary",
    "lambda_speed",
    "lambda_collision",
    "lambda_headway",
    "constraint_q_safety",
    "constraint_q_boundary",
    "constraint_q_speed",
    "safety_q_avg",
    "dual_loss",
    "safety_q1",
    "safety_q2",
    "constraint_q_collision",
    "constraint_q_headway",
    "safety_q_cvar",
    "safety_q1_loss",
    "safety_q2_loss",
    "constraint_q_collision_loss",
    "constraint_q_headway_loss",
    "target_collision_nstep_bootstrap",
    "target_headway_nstep_bootstrap",
    "safety_tail_loss",
    "cost_collision_loss",
    "cost_headway_loss",
    "cost_safety_loss",
    "cost_boundary_loss",
    "cost_speed_loss",
    "cost_overspeed_loss",
    "cost_comfort_loss",
    "cost_lane_loss",
    "cost_collision_q",
    "cost_headway_q",
    "cost_overspeed_q",
    "cost_comfort_q",
    "cost_lane_q",
    "batch_cost_collision",
    "batch_cost_headway",
    "batch_cost_safety",
    "batch_cost_boundary",
    "batch_cost_speed",
    "sampled_success_count",
    "sampled_near_danger_count",
    "sampled_collision_count",
    "sampled_danger_ratio_actual",
    "sampled_near_ratio_actual",
    "replay_danger_count",
    "replay_near_danger_count",
]


def build_train_diagnostic_row(ep: int, ep_summary: dict, last_update_info: dict) -> dict:
    row = {
        "episode": int(ep),
        "episode_return": float(ep_summary.get("episode_return", 0.0)),
        "success": float(ep_summary.get("success", 0.0)),
        "collision": float(ep_summary.get("collision", 0.0)),
        "goal_reached": float(ep_summary.get("goal_reached", 0.0)),
        "success_rate": float(ep_summary.get("success_rate", ep_summary.get("success", 0.0))),
        "goal_reached_rate": float(ep_summary.get("goal_reached_rate", ep_summary.get("goal_reached", 0.0))),
        "collision_rate": float(ep_summary.get("collision_rate", ep_summary.get("collision", 0.0))),
        "final_goal_distance_remaining": float(ep_summary.get("final_goal_distance_remaining", 0.0)),
        "max_goal_progress": float(ep_summary.get("max_goal_progress", 0.0)),
        "mean_env_reward_per_step": float(
            ep_summary.get("mean_env_reward_per_step", ep_summary.get("mean_env_reward", 0.0))
        ),
        "offroad_rate": float(ep_summary.get("offroad_rate", 0.0)),
        "unsafe_headway_rate": float(ep_summary.get("unsafe_headway_rate", 0.0)),
        "overspeed_rate": float(ep_summary.get("overspeed_rate", 0.0)),
        "low_ttc_rate": float(ep_summary.get("low_ttc_rate", 0.0)),
        "mean_speed": float(ep_summary.get("mean_speed", 0.0)),
        "min_front_distance": float(ep_summary.get("min_front_distance", 0.0)),
        "ttc_p10": float(ep_summary.get("ttc_p10", 0.0)),
        "ttc_p25": float(ep_summary.get("ttc_p25", 0.0)),
        "ttc_p50": float(ep_summary.get("ttc_p50", 0.0)),
        "ttc_p75": float(ep_summary.get("ttc_p75", 0.0)),
        "ttc_p90": float(ep_summary.get("ttc_p90", 0.0)),
        "mean_cost_collision": float(ep_summary.get("mean_cost_collision", 0.0)),
        "collision_cost_front": float(ep_summary.get("collision_cost_front", 0.0)),
        "collision_cost_rear": float(ep_summary.get("collision_cost_rear", 0.0)),
        "collision_cost_lateral": float(ep_summary.get("collision_cost_lateral", 0.0)),
        "mean_cost_headway": float(ep_summary.get("mean_cost_headway", 0.0)),
        "mean_cost_overspeed": float(ep_summary.get("mean_cost_overspeed", 0.0)),
        "mean_cost_comfort": float(ep_summary.get("mean_cost_comfort", 0.0)),
        "mean_cost_lane": float(ep_summary.get("mean_cost_lane", 0.0)),
        "mean_cost_total": float(ep_summary.get("mean_cost_total", 0.0)),
        "mean_cost_safety": float(ep_summary.get("mean_cost_safety", 0.0)),
        "std_long_action": float(ep_summary.get("std_long_action", 0.0)),
        "std_lat_action": float(ep_summary.get("std_lat_action", 0.0)),
    }
    for key in TRAIN_DIAGNOSTIC_FIELDS:
        row.setdefault(key, float(last_update_info.get(key, float("nan"))))
    return row


def summarize_episode_rows(rows: List[dict]) -> dict:
    if not rows:
        return {}
    keys = [
        "episode_return",
        "success",
        "collision",
        "goal_reached",
        "success_rate",
        "goal_reached_rate",
        "collision_rate",
        "final_goal_distance_remaining",
        "max_goal_progress",
        "mean_env_reward_per_step",
        "mean_speed",
        "min_front_distance",
        "ttc_p10",
        "ttc_p25",
        "ttc_p50",
        "ttc_p75",
        "ttc_p90",
        "unsafe_headway_rate",
        "overspeed_rate",
        "low_ttc_rate",
        "offroad_rate",
        "mean_cost_total",
        "mean_cost_safety",
        "std_long_action",
        "std_lat_action",
    ]
    out = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        out[key] = float(np.mean(values)) if values else 0.0
    return out


def create_tensorboard_writer(cfg: Config):
    path = str(cfg.diagnostics.tensorboard_log_dir).strip()
    if not path or SummaryWriter is None:
        return None, ""
    run_dir = os.path.join(path, cfg.train.mode, f"seed_{cfg.train.seed}_{_timestamp()}")
    os.makedirs(run_dir, exist_ok=True)
    return SummaryWriter(log_dir=run_dir, flush_secs=max(1, int(cfg.diagnostics.tensorboard_flush_secs))), run_dir


def write_tensorboard_row(writer, row: dict, mode: str) -> None:
    if writer is None:
        return
    ep = int(row.get("episode", 0))
    for key, value in row.items():
        if key == "episode":
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            writer.add_scalar(f"{mode}/{key}", numeric, ep)


def export_csv(rows: List[dict], save_path: str, fieldnames: List[str]) -> str:
    if not save_path:
        return ""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return os.path.abspath(save_path)


def export_json(payload: dict, save_path: str) -> str:
    if not save_path:
        return ""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable_plain(payload), f, indent=2, ensure_ascii=False)
    return os.path.abspath(save_path)


def default_result_path(cfg: Config) -> str:
    return os.path.join("results", f"{_timestamp()}_{cfg.train.mode}_seed{cfg.train.seed}_run_result.json")


def parse_mode_sequence(mode_expr: str, fallback_mode: str) -> List[str]:
    text = str(mode_expr).strip()
    if not text:
        return [str(fallback_mode).lower().strip()]
    modes: List[str] = []
    for raw_token in re.split(r"[,\s]+", text):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token not in ALL_MODES:
            raise ValueError(f"Invalid mode '{token}'. Choose from {ALL_MODES}.")
        if token not in modes:
            modes.append(token)
    return modes or [str(fallback_mode).lower().strip()]


def parse_seed_sequence(seed_expr: str, fallback_seed: int) -> List[int]:
    text = str(seed_expr).strip()
    if not text:
        return [int(fallback_seed)]
    seeds: List[int] = []
    for raw_token in re.split(r"[,\s]+", text):
        token = raw_token.strip()
        if not token:
            continue
        if re.fullmatch(r"-?\d+", token):
            seeds.append(int(token))
            continue
        range_match = re.fullmatch(r"(-?\d+)-(-?\d+)(?::(\d+))?", token)
        if range_match is None:
            raise ValueError(f"Invalid seed token '{token}'. Use ints or ranges like 40-44 or 40-50:2.")
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        step_mag = int(range_match.group(3) or "1")
        if step_mag <= 0:
            raise ValueError(f"Invalid seed range step in token '{token}'. Step must be positive.")
        step = step_mag if end >= start else -step_mag
        current = start
        if step > 0:
            while current <= end:
                seeds.append(int(current))
                current += step
        else:
            while current >= end:
                seeds.append(int(current))
                current += step
    if not seeds:
        return [int(fallback_seed)]
    deduped: List[int] = []
    seen = set()
    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        deduped.append(int(seed))
    return deduped


def _seeded_file_path(path: str, seed: int) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    if "{seed}" in raw:
        return raw.replace("{seed}", str(seed))
    stem, ext = os.path.splitext(raw)
    seed_tag = f"_seed{seed}"
    if re.search(r"_seed-?\d+$", stem):
        stem = re.sub(r"_seed-?\d+$", seed_tag, stem)
    else:
        stem = f"{stem}{seed_tag}"
    return f"{stem}{ext}"


def _seeded_dir_path(path: str, seed: int) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    if "{seed}" in raw:
        return raw.replace("{seed}", str(seed))
    if re.search(r"(?:^|[\\/])seed[_-]?-?\d+$", raw):
        return re.sub(r"seed[_-]?-?\d+$", f"seed_{seed}", raw)
    return os.path.join(raw, f"seed_{seed}")


def _apply_seed_specific_paths(cfg: Config, seed: int) -> None:
    cfg.train.seed = int(seed)
    if cfg.diagnostics.export_train_log:
        cfg.diagnostics.train_log_path = _seeded_file_path(cfg.diagnostics.train_log_path, seed)
    if cfg.diagnostics.save_train_json:
        cfg.diagnostics.train_json_path = _seeded_file_path(cfg.diagnostics.train_json_path, seed)
    if str(cfg.diagnostics.run_result_json_path).strip():
        cfg.diagnostics.run_result_json_path = _seeded_file_path(cfg.diagnostics.run_result_json_path, seed)
    if str(cfg.diagnostics.tensorboard_log_dir).strip():
        cfg.diagnostics.tensorboard_log_dir = _seeded_dir_path(cfg.diagnostics.tensorboard_log_dir, seed)
    if str(cfg.eval.video_dir).strip():
        cfg.eval.video_dir = _seeded_dir_path(cfg.eval.video_dir, seed)


def _mode_tagged_file_path(path: str, mode: str) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    if "{mode}" in raw:
        return raw.replace("{mode}", str(mode))
    stem, ext = os.path.splitext(raw)
    if re.search(r"_(goal_sac_pure|goal_csac_lagrangian|paper_sac_pure|paper_csac_lagrangian)$", stem):
        stem = re.sub(r"_(goal_sac_pure|goal_csac_lagrangian|paper_sac_pure|paper_csac_lagrangian)$", f"_{mode}", stem)
    else:
        stem = f"{stem}_{mode}"
    return f"{stem}{ext}"


def _mode_tagged_dir_path(path: str, mode: str) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    if "{mode}" in raw:
        return raw.replace("{mode}", str(mode))
    return os.path.join(raw, str(mode))


def _apply_mode_specific_paths(cfg: Config, mode: str) -> None:
    if cfg.diagnostics.export_train_log:
        cfg.diagnostics.train_log_path = _mode_tagged_file_path(cfg.diagnostics.train_log_path, mode)
    if cfg.diagnostics.save_train_json:
        cfg.diagnostics.train_json_path = _mode_tagged_file_path(cfg.diagnostics.train_json_path, mode)
    if str(cfg.diagnostics.run_result_json_path).strip():
        cfg.diagnostics.run_result_json_path = _mode_tagged_file_path(cfg.diagnostics.run_result_json_path, mode)
    if str(cfg.diagnostics.tensorboard_log_dir).strip():
        cfg.diagnostics.tensorboard_log_dir = _mode_tagged_dir_path(cfg.diagnostics.tensorboard_log_dir, mode)
    if str(cfg.eval.video_dir).strip():
        cfg.eval.video_dir = _mode_tagged_dir_path(cfg.eval.video_dir, mode)


def _safe_stats(values: List[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _default_multi_seed_summary_path(cfg: Config) -> str:
    base = str(cfg.diagnostics.run_result_json_path).strip()
    if base:
        if "{seed}" in base:
            base = base.replace("{seed}", "all")
        stem, ext = os.path.splitext(base)
        final_ext = ext or ".json"
        return f"{stem}_multi_seed_summary{final_ext}"
    return os.path.join("results", f"{_timestamp()}_{cfg.train.mode}_multi_seed_summary.json")


def _build_multi_seed_summary(base_cfg: Config, seeds: List[int], run_results: List[dict]) -> dict:
    per_seed: List[dict] = []
    for result in run_results:
        train_summary = dict(result.get("train_summary") or {})
        eval_primary = dict(((result.get("eval") or {}).get("primary")) or {})
        config_payload = dict(result.get("config") or {})
        train_cfg_payload = dict(config_payload.get("train") or {})
        seed_value = int(train_cfg_payload.get("seed", 0))
        row = {
            "seed": seed_value,
            "run_result_json_path": str(result.get("run_result_json_path", "")),
            "train_return": float(train_summary.get("episode_return", 0.0)),
            "train_success": float(train_summary.get("success", 0.0)),
            "train_collision": float(train_summary.get("collision", 0.0)),
            "train_offroad_rate": float(train_summary.get("offroad_rate", 0.0)),
            "train_mean_cost_total": float(train_summary.get("mean_cost_total", 0.0)),
            "eval_return": float(eval_primary.get("episode_return", 0.0)),
            "eval_success": float(eval_primary.get("success", 0.0)),
            "eval_collision": float(eval_primary.get("collision", 0.0)),
            "eval_offroad_rate": float(eval_primary.get("offroad_rate", 0.0)),
            "eval_mean_cost_total": float(eval_primary.get("mean_cost_total", 0.0)),
            "eval_mean_speed": float(eval_primary.get("mean_speed", 0.0)),
        }
        per_seed.append(row)
    keys = [
        "train_return",
        "train_success",
        "train_collision",
        "train_offroad_rate",
        "train_mean_cost_total",
        "eval_return",
        "eval_success",
        "eval_collision",
        "eval_offroad_rate",
        "eval_mean_cost_total",
        "eval_mean_speed",
    ]
    aggregate = {key: _safe_stats([float(row.get(key, 0.0)) for row in per_seed]) for key in keys}
    return {
        "mode": str(base_cfg.train.mode),
        "penalty_impl": str(base_cfg.sac.penalty_impl),
        "upgrade_stage": str(base_cfg.sac.upgrade_stage),
        "seeds_requested": [int(s) for s in seeds],
        "seed_count": int(len(seeds)),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "timestamp_utc": _utc_now_iso(),
    }


def run_multi_seed(base_cfg: Config, seeds: List[int], summary_path: str = "") -> dict:
    run_results: List[dict] = []
    total = len(seeds)
    for idx, seed in enumerate(seeds, start=1):
        seed_cfg = copy.deepcopy(base_cfg)
        _apply_seed_specific_paths(seed_cfg, int(seed))
        print(f"[MULTI-SEED] ({idx}/{total}) seed={seed_cfg.train.seed}")
        run_results.append(run(seed_cfg))
    summary = _build_multi_seed_summary(base_cfg, seeds, run_results)
    output_path = str(summary_path).strip() or _default_multi_seed_summary_path(base_cfg)
    summary["multi_seed_summary_path"] = export_json(summary, output_path)
    print(f"[MULTI-SEED] summary saved: {summary['multi_seed_summary_path']}")
    return summary


def _build_multi_mode_summary(modes: List[str], seeds: List[int], run_results: List[dict]) -> dict:
    rows: List[dict] = []
    for result in run_results:
        train_summary = dict(result.get("train_summary") or {})
        eval_primary = dict(((result.get("eval") or {}).get("primary")) or {})
        config_payload = dict(result.get("config") or {})
        train_payload = dict(config_payload.get("train") or {})
        env_payload = dict(config_payload.get("env") or {})
        reward_payload = dict(config_payload.get("reward") or {})
        rows.append({
            "mode": str(train_payload.get("mode", result.get("mode", ""))),
            "reward_type": str(reward_payload.get("reward_type", "")),
            "traffic_density": str(env_payload.get("traffic_density", "")),
            "seed": int(train_payload.get("seed", 0)),
            "run_result_json_path": str(result.get("run_result_json_path", "")),
            "train_success": float(train_summary.get("success", 0.0)),
            "train_collision": float(train_summary.get("collision", 0.0)),
            "train_goal_reached": float(train_summary.get("goal_reached", 0.0)),
            "train_return": float(train_summary.get("episode_return", 0.0)),
            "train_mean_cost_total": float(train_summary.get("mean_cost_total", 0.0)),
            "eval_success": float(eval_primary.get("success", 0.0)),
            "eval_collision": float(eval_primary.get("collision", 0.0)),
            "eval_goal_reached": float(eval_primary.get("goal_reached", 0.0)),
            "eval_return": float(eval_primary.get("episode_return", 0.0)),
            "eval_mean_cost_total": float(eval_primary.get("mean_cost_total", 0.0)),
            "eval_mean_speed": float(eval_primary.get("mean_speed", 0.0)),
        })
    aggregate: Dict[str, dict] = {}
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        aggregate[mode] = {
            key: _safe_stats([float(row.get(key, 0.0)) for row in mode_rows])
            for key in [
                "train_success", "train_collision", "train_goal_reached", "train_return", "train_mean_cost_total",
                "eval_success", "eval_collision", "eval_goal_reached", "eval_return", "eval_mean_cost_total", "eval_mean_speed",
            ]
        }
    return {
        "modes_requested": [str(m) for m in modes],
        "seeds_requested": [int(s) for s in seeds],
        "runs": rows,
        "aggregate_by_mode": aggregate,
        "timestamp_utc": _utc_now_iso(),
    }


def run_multi_mode_seed(base_cfg: Config, modes: List[str], seeds: List[int], summary_path: str = "") -> dict:
    run_results: List[dict] = []
    total = len(modes) * len(seeds)
    counter = 0
    for mode in modes:
        for seed in seeds:
            counter += 1
            run_cfg = copy.deepcopy(base_cfg)
            configure_mode_behavior(run_cfg, mode)
            _apply_mode_specific_paths(run_cfg, mode)
            _apply_seed_specific_paths(run_cfg, int(seed))
            print(f"[MULTI-MODE] ({counter}/{total}) mode={mode} seed={seed}")
            run_results.append(run(run_cfg))
    summary = _build_multi_mode_summary(modes, seeds, run_results)
    output_path = str(summary_path).strip() or os.path.join(
        "results", f"{_timestamp()}_four_mode_comparison_summary.json"
    )
    summary["multi_mode_summary_path"] = export_json(summary, output_path)
    print(f"[MULTI-MODE] summary saved: {summary['multi_mode_summary_path']}")
    return summary


def maybe_print_update(update_info: dict) -> None:
    if not update_info:
        return
    msg = (
        f"[UPDATE] q1={update_info.get('q1_loss', 0.0):.4f} "
        f"q2={update_info.get('q2_loss', 0.0):.4f} "
        f"actor={update_info.get('actor_loss', 0.0):.4f} "
        f"alpha={update_info.get('alpha', 0.0):.4f}"
    )
    if "fixed_actor_penalty" in update_info:
        msg += f" fixed_penalty={update_info.get('fixed_actor_penalty', 0.0):.4f}"
    if "lambda_safety" in update_info:
        msg += f" lambda_safety={update_info.get('lambda_safety', 0.0):.4f}"
    if "lambda_collision" in update_info and "lambda_headway" in update_info:
        msg += (
            f" lambda_collision={update_info.get('lambda_collision', 0.0):.4f}"
            f" lambda_headway={update_info.get('lambda_headway', 0.0):.4f}"
        )
    if "safety_q_avg" in update_info:
        msg += f" safety_q_avg={update_info.get('safety_q_avg', 0.0):.4f}"
    if "constraint_q_collision" in update_info and "constraint_q_headway" in update_info:
        msg += (
            f" constraint_q_collision={update_info.get('constraint_q_collision', 0.0):.4f}"
            f" constraint_q_headway={update_info.get('constraint_q_headway', 0.0):.4f}"
        )
    print(msg)


def _video_output_path(cfg: Config, ep: int, eval_seed: int) -> str:
    base_dir = str(cfg.eval.video_dir).strip() or os.path.join("results", "eval_videos")
    os.makedirs(base_dir, exist_ok=True)
    filename = f"{cfg.train.mode}_seed{cfg.train.seed}_evalep{ep}_evalseed{eval_seed}_{_timestamp()}.mp4"
    return os.path.abspath(os.path.join(base_dir, filename))


def _write_video(frames: List[np.ndarray], save_path: str, fps: int) -> Tuple[str, str]:
    if not frames:
        return "", "no_frames"
    if imageio is None:
        return "", "imageio_missing"
    try:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        imageio.mimsave(save_path, frames, fps=max(1, int(fps)))
        return os.path.abspath(save_path), "saved"
    except Exception as exc:
        return "", f"write_failed: {exc}"


def evaluate_policy(cfg: Config, agent) -> dict:
    print("\n================ EVAL RAW POLICY START ================\n")
    eval_results = []
    saved_videos: List[str] = []
    video_events: List[dict] = []
    for ep in range(cfg.eval.episodes):
        eval_seed = int(cfg.train.seed + cfg.eval.seed_offset + ep)
        env = HighwayEnvWrapper(cfg, render=cfg.eval.render, record_video=cfg.eval.save_video)
        frames: List[np.ndarray] = []
        try:
            state, _ = env.reset(seed=eval_seed)
            agent.reset_episode_context()
            if cfg.eval.save_video:
                first_frame = env.render_frame()
                if first_frame is not None:
                    frames.append(first_frame)
            metrics = EpisodeMetrics(cfg)
            for t in range(cfg.train.max_steps_per_episode):
                scene = env.get_scene_dict()
                action = env.clip_action(agent.select_action(state, evaluate=True))
                next_state, env_reward, terminated, truncated, _ = env.step(action)
                if cfg.eval.save_video:
                    frame = env.render_frame()
                    if frame is not None:
                        frames.append(frame)
                next_scene = env.get_scene_dict()
                next_is_success = terminal_success(cfg, next_scene)
                next_is_failure = terminal_failure(cfg, next_scene)
                timeout_failure = bool((truncated or (t + 1 >= cfg.train.max_steps_per_episode)) and not next_is_success and not next_is_failure)
                reward, _ = compute_reward(cfg, scene, next_scene, action, env_reward, timeout_failure=timeout_failure)
                cost_dict = compute_costs(cfg, scene, next_scene, action)
                done = bool(terminated or truncated or next_is_failure or next_is_success or timeout_failure)
                metrics.update(scene, next_scene, action, reward, env_reward, cost_dict)
                if cfg.eval.print_step:
                    print(
                        f"[EVAL-STEP][ep={ep}][t={t}] "
                        f"speed={next_scene['ego_speed']:.2f} "
                        f"front_d={next_scene['front_distance']:.2f} "
                        f"ttc={next_scene['ttc']:.2f} "
                        f"action=[{action[0]:.2f},{action[1]:.2f}] "
                        f"reward={reward:.3f} "
                        f"collision={int(next_scene['collision'])} "
                        f"cost={cost_dict['total_cost']:.3f}"
                    )
                state = next_state
                if done:
                    break
            ep_summary = metrics.summary()
            if cfg.eval.save_video:
                video_path, video_status = _write_video(frames, _video_output_path(cfg, ep, eval_seed), fps=cfg.env.policy_frequency)
                ep_summary["video_path"] = video_path
                ep_summary["video_status"] = video_status
                if video_path:
                    saved_videos.append(video_path)
                    print(f"[EVAL-VIDEO][ep={ep}] saved={video_path}")
                else:
                    print(f"[EVAL-VIDEO][ep={ep}][WARN] status={video_status}")
                video_events.append({
                    "episode": int(ep),
                    "eval_seed": int(eval_seed),
                    "video_path": video_path,
                    "video_status": video_status,
                    "frame_count": int(len(frames)),
                })
            eval_results.append(ep_summary)
            print(
                f"[EVAL][ep={ep}] return={ep_summary['episode_return']:.3f} "
                f"steps={ep_summary['steps']} success={ep_summary['success']} "
                f"collision={ep_summary['collision']} mean_speed={ep_summary['mean_speed']:.3f} "
                f"cost={ep_summary['mean_cost_total']:.3f}"
            )
        finally:
            env.close()
    summary = summarize_episode_rows(eval_results)
    print("\n================ EVAL RAW POLICY SUMMARY ================\n")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    return {
        "raw": {"episodes": eval_results, "summary": summary},
        "primary": summary,
        "primary_report": "raw",
        "saved_videos": saved_videos,
        "video_events": video_events,
    }


def run(cfg: Config) -> dict:
    if cfg.train.mode not in ALL_MODES:
        raise ValueError(f"Unsupported mode: {cfg.train.mode}")
    set_seed(cfg.train.seed)
    print(
        f"[RUN-CONFIG] mode={cfg.train.mode} traffic={cfg.env.traffic_density} "
        f"seed={cfg.train.seed} device={describe_runtime_device(cfg.train.device)} "
        f"scenario=4lanes_30mps_20s_absobs_goal500 "
        f"penalty_impl={cfg.sac.penalty_impl} stage={cfg.sac.upgrade_stage}"
    )
    if cfg.train.mode == "penalty_sac_pure":
        print(
            "[SAFETY-BUDGET-NOTE] safety_budget_collision/headway are round-1 initial "
            "implementation values for multi-seed tuning, not theoretical constants."
        )
    env = HighwayEnvWrapper(cfg, render=cfg.train.render)
    state, _ = env.reset(seed=cfg.train.seed)
    state_dim = int(state.shape[0])
    action_dim = env.action_dim()
    agent = build_agent(state_dim, action_dim, cfg, env.action_low, env.action_high)
    tb_writer, tb_run_dir = create_tensorboard_writer(cfg)
    if tb_run_dir:
        print(f"[DIAG] tensorboard run dir: {tb_run_dir}")

    total_steps = 0
    train_episode_results: List[dict] = []
    train_diagnostic_rows: List[dict] = []
    start_time = time.time()
    try:
        for ep in range(cfg.train.episodes):
            state, _ = env.reset(seed=cfg.train.seed + ep)
            agent.reset_episode_context()
            metrics = EpisodeMetrics(cfg)
            last_update_info: Dict[str, float] = {}
            for t in range(cfg.train.max_steps_per_episode):
                scene = env.get_scene_dict()
                if total_steps < cfg.sac.start_steps:
                    action = env.sample_random_action()
                else:
                    action = env.clip_action(agent.select_action(state, evaluate=False))

                next_state, env_reward, terminated, truncated, _ = env.step(action)
                next_scene = env.get_scene_dict()
                next_is_success = terminal_success(cfg, next_scene)
                next_is_failure = terminal_failure(cfg, next_scene)
                timeout_failure = bool((truncated or (t + 1 >= cfg.train.max_steps_per_episode)) and not next_is_success and not next_is_failure)
                reward, reward_terms = compute_reward(cfg, scene, next_scene, action, env_reward, timeout_failure=timeout_failure)
                cost_dict = compute_costs(cfg, scene, next_scene, action)
                done = bool(terminated or truncated or next_is_failure or next_is_success or timeout_failure)
                # Danger tags are training-data annotations only. They do not alter online actions.
                danger_info = classify_transition_danger(cfg, scene, next_scene, cost_dict)
                agent.replay.add(
                    state,
                    action,
                    reward,
                    next_state,
                    float(done),
                    cost_dict=cost_dict,
                    transition_meta={
                        "episode_id": int(ep),
                        "step_in_episode": int(t),
                        "danger_label": int(danger_info["danger_label"]),
                        "is_collision": bool(danger_info["is_collision"]),
                        "is_near_danger": bool(danger_info["is_near_danger"]),
                    },
                )
                if bool(danger_info["is_collision"]):
                    agent.replay.mark_precollision_danger(
                        episode_id=int(ep),
                        current_step=int(t),
                        back_steps=int(cfg.sac.danger_precollision_steps),
                    )
                metrics.update(scene, next_scene, action, reward, env_reward, cost_dict)

                update_info = {}
                if total_steps >= cfg.sac.update_after and total_steps % cfg.sac.update_every == 0:
                    update_info = agent.update()
                    if update_info:
                        last_update_info = dict(update_info)

                if cfg.train.print_every_step:
                    print(
                        f"[STEP][ep={ep}][t={t}] speed={next_scene['ego_speed']:.2f} "
                        f"front_d={next_scene['front_distance']:.2f} ttc={next_scene['ttc']:.2f} "
                        f"action=[{action[0]:.2f},{action[1]:.2f}] reward={reward:.3f} "
                        f"source={reward_terms['source']} collision={int(next_scene['collision'])} "
                        f"cost={cost_dict['total_cost']:.3f}"
                    )
                    maybe_print_update(update_info)

                state = next_state
                total_steps += 1
                if done:
                    break

            ep_summary = metrics.summary()
            agent.replay.mark_episode_success(int(ep), bool(ep_summary.get("success", 0)))
            train_episode_results.append(ep_summary)
            train_diag_row = build_train_diagnostic_row(ep, ep_summary, last_update_info)
            train_diagnostic_rows.append(train_diag_row)
            write_tensorboard_row(tb_writer, train_diag_row, cfg.train.mode)
            print(
                f"[EPISODE END] ep={ep} return={ep_summary['episode_return']:.3f} "
                f"steps={ep_summary['steps']} success={ep_summary['success']} "
                f"collision={ep_summary['collision']} offroad_rate={ep_summary['offroad_rate']:.3f} "
                f"overspeed_rate={ep_summary['overspeed_rate']:.3f} "
                f"mean_cost_total={ep_summary['mean_cost_total']:.3f} replay_size={agent.replay.size}"
            )
            print(
                f"[TRAIN-DIAG] ep={ep} success={train_diag_row['success']:.0f} "
                f"collision={train_diag_row['collision']:.0f} offroad_rate={train_diag_row['offroad_rate']:.3f} "
                f"unsafe_headway_rate={train_diag_row['unsafe_headway_rate']:.3f} "
                f"overspeed_rate={train_diag_row['overspeed_rate']:.3f} "
                f"mean_cost_total={train_diag_row['mean_cost_total']:.3f} "
                f"return={train_diag_row['episode_return']:.3f}"
            )
    finally:
        env.close()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    train_summary = summarize_episode_rows(train_episode_results)
    train_log_path = export_csv(train_diagnostic_rows, cfg.diagnostics.train_log_path, TRAIN_DIAGNOSTIC_FIELDS) if cfg.diagnostics.export_train_log else ""
    train_json_path = export_json(train_diagnostic_rows, cfg.diagnostics.train_json_path) if cfg.diagnostics.save_train_json else ""
    effective_features = _effective_feature_flags(cfg)
    config_payload = asdict(cfg)
    config_payload["effective_features"] = dict(effective_features)
    result = {
        "mode": cfg.train.mode,
        "train_episodes": train_episode_results,
        "train_summary": train_summary,
        "eval": None,
        "config": config_payload,
        "metadata": {
            "timestamp_utc": _utc_now_iso(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "elapsed_sec": float(time.time() - start_time),
            "runtime_env": env.get_runtime_env_snapshot(),
            "penalty_impl": str(cfg.sac.penalty_impl),
            "upgrade_stage": str(cfg.sac.upgrade_stage),
            "effective_features": dict(effective_features),
        },
        "train_diagnostics": train_diagnostic_rows,
        "train_diagnostics_csv_path": train_log_path,
        "train_diagnostics_json_path": train_json_path,
        "train_tensorboard_dir": tb_run_dir,
        "run_result_json_path": "",
    }
    if cfg.eval.enabled:
        result["eval"] = evaluate_policy(cfg, agent)
    result_path = str(cfg.diagnostics.run_result_json_path).strip() or default_result_path(cfg)
    result["run_result_json_path"] = export_json(result, result_path)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Four-mode highway SAC / constrained SAC comparison experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Windows launch examples:\n"
            "  Single line:\n"
            "    python fix10D_four_sac_reward_constraints.py --modes goal_sac_pure,goal_csac_lagrangian,paper_sac_pure,paper_csac_lagrangian --traffic-density sparse --episodes 500 --max-steps 400\n"
            "  cmd.exe multiline (caret ^):\n"
            "    python fix10C_diag_instrumented_stage4.py ^\n"
            "      --modes goal_sac_pure,goal_csac_lagrangian,paper_sac_pure,paper_csac_lagrangian ^\n"
            "      --penalty-impl stage4_constrained_recurrent_v1 ^\n"
            "      --upgrade-stage m1m2 ^\n"
            "      --episodes 500 ^\n"
            "      --max-steps 200\n"
            "  PowerShell multiline (backtick `):\n"
            "    python fix10C_diag_instrumented_stage4.py `\n"
            "      --modes goal_sac_pure,goal_csac_lagrangian,paper_sac_pure,paper_csac_lagrangian `\n"
            "      --penalty-impl stage4_constrained_recurrent_v1 `\n"
            "      --upgrade-stage m1m2 `\n"
            "      --episodes 500 `\n"
            "      --max-steps 200"
        ),
    )
    parser.add_argument(
        "--scenario-profile",
        type=str,
        default="highway",
        choices=["highway"],
        help="Highway-only experiment profile.",
    )
    parser.add_argument("--traffic-density", type=str, default="sparse", choices=["sparse", "dense"],
                        help="Traffic profile: sparse samples 10~15 vehicles; dense samples 20~25 vehicles per episode.")
    parser.add_argument("--mode", type=str, default="goal_csac_lagrangian", choices=ALL_MODES)
    parser.add_argument("--modes", type=str, default="", help="Comma/space separated modes to run for comparison. Overrides --mode when provided.")
    parser.add_argument("--penalty-impl", type=str, default="stage4_constrained_recurrent_v1", choices=PENALTY_IMPLS)
    parser.add_argument("--upgrade-stage", type=str, default="m1m2", choices=UPGRADE_STAGES)
    parser.add_argument("--experiment-round", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--disable-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma/space separated seeds or ranges, e.g. '40,41,42' or '40-44' or '40-50:2'.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--eval-render", action="store_true")
    parser.add_argument("--eval-save-video", action="store_true")
    parser.add_argument("--eval-video-dir", type=str, default="")
    parser.add_argument("--print-every-step", action="store_true")
    parser.add_argument("--eval-print-step", action="store_true")
    parser.add_argument("--train-log-path", type=str, default="")
    parser.add_argument("--train-json-path", type=str, default="")
    parser.add_argument("--run-result-json-path", type=str, default="")
    parser.add_argument("--multi-seed-summary-path", type=str, default="")
    parser.add_argument("--tensorboard-dir", type=str, default="")
    parser.add_argument("--tensorboard-flush-secs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--start-steps", type=int, default=None)
    parser.add_argument("--update-after", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--penalty-collision", type=float, default=None)
    parser.add_argument("--penalty-headway", type=float, default=None)
    parser.add_argument("--penalty-overspeed", type=float, default=None)
    parser.add_argument("--penalty-comfort", type=float, default=None)
    parser.add_argument("--penalty-lane", type=float, default=None)
    parser.add_argument("--actor-decouple-mode", type=str, default=None, choices=ACTOR_DECOUPLE_MODES)
    parser.add_argument("--safety-w-collision", type=float, default=None)
    parser.add_argument("--safety-w-headway", type=float, default=None)
    parser.add_argument("--safety-budget", type=float, default=None)
    parser.add_argument("--safety-budget-collision", type=float, default=None)
    parser.add_argument("--safety-budget-headway", type=float, default=None)
    parser.add_argument("--lambda-init", type=float, default=None)
    parser.add_argument("--lambda-lr", type=float, default=None)
    parser.add_argument("--lambda-max", type=float, default=None)
    parser.add_argument("--constraint-budget-safety", type=float, default=None)
    parser.add_argument("--constraint-budget-boundary", type=float, default=None)
    parser.add_argument("--constraint-budget-speed", type=float, default=None)
    parser.add_argument("--constraint-lambda-lr", type=float, default=None)
    parser.add_argument("--constraint-lambda-max", type=float, default=None)
    parser.add_argument("--actor-update-interval", type=int, default=None)
    parser.add_argument("--alpha-min", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--gru-hidden-dim", type=int, default=None)
    parser.add_argument("--safety-n-step", type=int, default=None)
    parser.add_argument("--danger-ratio", type=float, default=None)
    parser.add_argument("--near-danger-ratio", type=float, default=None)
    parser.add_argument("--danger-precollision-steps", type=int, default=None)
    parser.add_argument("--cvar-alpha", type=float, default=None)
    parser.add_argument("--cvar-quantiles", type=int, default=None)

    parser.add_argument("--enable-priority-safety-replay", dest="enable_priority_safety_replay", action="store_true", default=None)
    parser.add_argument("--disable-priority-safety-replay", dest="enable_priority_safety_replay", action="store_false")
    parser.add_argument("--enable-lagrangian-safety", dest="enable_lagrangian_safety", action="store_true", default=None)
    parser.add_argument("--disable-lagrangian-safety", dest="enable_lagrangian_safety", action="store_false")
    parser.add_argument("--enable-recurrent-encoder", dest="enable_recurrent_encoder", action="store_true", default=None)
    parser.add_argument("--disable-recurrent-encoder", dest="enable_recurrent_encoder", action="store_false")
    parser.add_argument("--enable-decoupled-actor", dest="enable_decoupled_actor", action="store_true", default=None)
    parser.add_argument("--disable-decoupled-actor", dest="enable_decoupled_actor", action="store_false")
    parser.add_argument("--enable-tail-risk-cvar", dest="enable_tail_risk_cvar", action="store_true", default=None)
    parser.add_argument("--disable-tail-risk-cvar", dest="enable_tail_risk_cvar", action="store_false")
    return parser.parse_args()


def apply_args(cfg: Config, args) -> Config:
    if str(getattr(args, "scenario_profile", "highway")).lower().strip() != "highway":
        raise ValueError("Only 'highway' scenario-profile is supported in this script.")
    cfg.env.traffic_density = str(getattr(args, "traffic_density", "sparse")).lower().strip()
    cfg.env.absolute_obs = True
    cfg.env.speed_limit = 30.0
    cfg.env.max_speed = 30.0
    cfg.env.duration = 20
    cfg.env.simulation_frequency = 20
    cfg.env.policy_frequency = 20
    cfg.env.goal_lane_id = int(cfg.env.lanes_count - 1)
    cfg.cost.speed_limit = 30.0
    cfg.train.mode = str(args.mode).lower()
    cfg.train.episodes = max(0, int(args.episodes))
    cfg.train.max_steps_per_episode = max(1, int(args.max_steps))
    cfg.train.seed = int(args.seed)
    cfg.train.device = resolve_runtime_device(args.device)
    cfg.train.render = bool(args.render)
    cfg.train.print_every_step = bool(args.print_every_step)
    cfg.eval.episodes = max(0, int(args.eval_episodes))
    cfg.eval.enabled = (not bool(args.disable_eval)) and cfg.eval.episodes > 0
    cfg.eval.render = bool(args.eval_render)
    cfg.eval.print_step = bool(args.eval_print_step)
    cfg.eval.save_video = bool(args.eval_save_video)
    cfg.eval.video_dir = str(args.eval_video_dir).strip()
    cfg.diagnostics.export_train_log = bool(str(args.train_log_path).strip())
    cfg.diagnostics.train_log_path = str(args.train_log_path).strip()
    cfg.diagnostics.save_train_json = bool(str(args.train_json_path).strip())
    cfg.diagnostics.train_json_path = str(args.train_json_path).strip()
    cfg.diagnostics.run_result_json_path = str(args.run_result_json_path).strip()
    cfg.diagnostics.tensorboard_log_dir = str(args.tensorboard_dir).strip()
    cfg.diagnostics.tensorboard_flush_secs = max(1, int(args.tensorboard_flush_secs))
    if args.batch_size is not None:
        cfg.sac.batch_size = max(1, int(args.batch_size))
    if args.start_steps is not None:
        cfg.sac.start_steps = max(0, int(args.start_steps))
    if args.update_after is not None:
        cfg.sac.update_after = max(0, int(args.update_after))
    if args.buffer_size is not None:
        cfg.sac.buffer_size = max(1, int(args.buffer_size))
    if args.hidden_dim is not None:
        cfg.sac.hidden_dim = max(16, int(args.hidden_dim))
    if args.penalty_collision is not None:
        cfg.sac.penalty_collision = float(args.penalty_collision)
    if args.penalty_headway is not None:
        cfg.sac.penalty_headway = float(args.penalty_headway)
    if args.penalty_overspeed is not None:
        cfg.sac.penalty_overspeed = float(args.penalty_overspeed)
    if args.penalty_comfort is not None:
        cfg.sac.penalty_comfort = float(args.penalty_comfort)
    if args.penalty_lane is not None:
        cfg.sac.penalty_lane = float(args.penalty_lane)
    cfg.sac.penalty_impl = str(args.penalty_impl).lower()
    cfg.sac.upgrade_stage = str(args.upgrade_stage).lower()
    cfg.sac.experiment_round = int(np.clip(int(args.experiment_round), 1, 3))
    if args.safety_w_collision is not None:
        cfg.sac.safety_w_collision = max(0.0, float(args.safety_w_collision))
    if args.safety_w_headway is not None:
        cfg.sac.safety_w_headway = max(0.0, float(args.safety_w_headway))
    if args.safety_budget is not None:
        shared_budget = max(0.0, float(args.safety_budget))
        cfg.sac.safety_budget = shared_budget
        cfg.sac.safety_budget_collision = shared_budget
        cfg.sac.safety_budget_headway = shared_budget
    if args.safety_budget_collision is not None:
        cfg.sac.safety_budget_collision = max(0.0, float(args.safety_budget_collision))
    if args.safety_budget_headway is not None:
        cfg.sac.safety_budget_headway = max(0.0, float(args.safety_budget_headway))
    if args.lambda_init is not None:
        cfg.sac.lambda_init = max(0.0, float(args.lambda_init))
    if args.lambda_lr is not None:
        cfg.sac.lambda_lr = max(0.0, float(args.lambda_lr))
    if args.lambda_max is not None:
        cfg.sac.lambda_max = max(0.0, float(args.lambda_max))
    if getattr(args, "constraint_budget_safety", None) is not None:
        cfg.sac.constraint_budget_safety = max(0.0, float(args.constraint_budget_safety))
    if getattr(args, "constraint_budget_boundary", None) is not None:
        cfg.sac.constraint_budget_boundary = max(0.0, float(args.constraint_budget_boundary))
    if getattr(args, "constraint_budget_speed", None) is not None:
        cfg.sac.constraint_budget_speed = max(0.0, float(args.constraint_budget_speed))
    if getattr(args, "constraint_lambda_lr", None) is not None:
        cfg.sac.constraint_lambda_lr = max(0.0, float(args.constraint_lambda_lr))
    if getattr(args, "constraint_lambda_max", None) is not None:
        cfg.sac.constraint_lambda_max = max(0.0, float(args.constraint_lambda_max))
    if args.actor_update_interval is not None:
        cfg.sac.actor_update_interval = max(1, int(args.actor_update_interval))
    if args.alpha_min is not None:
        cfg.sac.alpha_min = max(0.0, float(args.alpha_min))
    if args.seq_len is not None:
        cfg.sac.seq_len = max(1, int(args.seq_len))
    if args.gru_hidden_dim is not None:
        cfg.sac.gru_hidden_dim = max(16, int(args.gru_hidden_dim))
    if args.safety_n_step is not None:
        cfg.sac.safety_n_step = max(1, int(args.safety_n_step))
    if args.danger_ratio is not None:
        cfg.sac.danger_ratio = float(np.clip(args.danger_ratio, 0.0, 1.0))
    if args.near_danger_ratio is not None:
        cfg.sac.near_danger_ratio = float(np.clip(args.near_danger_ratio, 0.0, 1.0))
    if args.danger_precollision_steps is not None:
        cfg.sac.danger_precollision_steps = max(0, int(args.danger_precollision_steps))
    if args.actor_decouple_mode is not None:
        cfg.sac.actor_decouple_mode = str(args.actor_decouple_mode).lower()
    if args.cvar_alpha is not None:
        cfg.sac.cvar_alpha = float(np.clip(args.cvar_alpha, 1e-3, 0.99))
    if args.cvar_quantiles is not None:
        cfg.sac.cvar_quantiles = max(2, int(args.cvar_quantiles))

    configure_mode_behavior(cfg, cfg.train.mode)
    cfg.sac.safety_budget = float(0.5 * (cfg.sac.safety_budget_collision + cfg.sac.safety_budget_headway))
    cfg.sac.near_danger_ratio = float(np.clip(cfg.sac.near_danger_ratio, 0.0, 1.0 - float(cfg.sac.danger_ratio)))
    validate_highway_only_config(cfg)
    return cfg


if __name__ == "__main__":
    parsed_args = parse_args()
    config = apply_args(get_config(), parsed_args)
    configure_torch_runtime(config.train.device)
    requested_seeds = parse_seed_sequence(getattr(parsed_args, "seeds", ""), config.train.seed)
    requested_modes = parse_mode_sequence(getattr(parsed_args, "modes", ""), config.train.mode)
    if len(requested_modes) == 1 and len(requested_seeds) == 1:
        configure_mode_behavior(config, requested_modes[0])
        run(config)
    else:
        print(f"[MULTI-MODE] requested modes: {requested_modes}")
        print(f"[MULTI-MODE] requested seeds: {requested_seeds}")
        run_multi_mode_seed(
            base_cfg=config,
            modes=requested_modes,
            seeds=requested_seeds,
            summary_path=str(getattr(parsed_args, "multi_seed_summary_path", "")).strip(),
        )
