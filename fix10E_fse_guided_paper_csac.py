from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
try:
    import highway_env  # noqa: F401  # registers highway-env ids
except Exception:
    highway_env = None
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


FSE_ACTION_RISK_MODES = [
    "paper_csac_fse_z_gated_action_risk",
    "paper_csac_fse_z_gated_action_risk_shuffled_tokens",
]
FSE_RL_MODES = [
    "paper_csac_fse_z_concat",
    "paper_csac_fse_z_gated",
    "paper_csac_fse_z_gated_random",
    "paper_csac_fse_z_gated_shuffled",
    "paper_csac_fse_z_gated_zero",
] + FSE_ACTION_RISK_MODES
ALL_MODES = ["paper_sac_pure", "paper_csac_lagrangian"] + FSE_RL_MODES
FSE_TASKS = ["none", "collect", "build-dataset", "build-z-artifacts", "train", "eval", "full", "smoke"]
FSE_COLLECT_POLICIES = ["online_train", "eval_policy", "noisy_eval", "random"]
FSE_FUSION_MODES = ["concat", "gated"]
FSE_Z_MODES = ["real", "random", "shuffled", "zero"]
FSE_RANDOM_Z_DISTRIBUTIONS = ["global_empirical_matched", "batch_empirical_matched", "per_sample_norm_matched", "standard_normal"]
FSE_RUN_TIERS = ["smoke", "formal"]
FSE_Z_ARTIFACT_TARGETS = ["both", "stats", "pool"]
FSE_Z_ARTIFACT_SOURCE_SPLITS = ["train", "val", "train_val"]
FSE_RL_MODE_CONFIG = {
    "paper_csac_fse_z_concat": ("concat", "real"),
    "paper_csac_fse_z_gated": ("gated", "real"),
    "paper_csac_fse_z_gated_random": ("gated", "random"),
    "paper_csac_fse_z_gated_shuffled": ("gated", "shuffled"),
    "paper_csac_fse_z_gated_zero": ("gated", "zero"),
    "paper_csac_fse_z_gated_action_risk": ("gated", "real"),
    "paper_csac_fse_z_gated_action_risk_shuffled_tokens": ("gated", "real"),
}
FSE_FORBIDDEN_FORMAL_Z_SOURCE_SPLITS = {"test", "mixed_test", "final_mixed_test", "final_test"}
PENALTY_IMPLS = ["legacy_fixed_penalty", "stage4_constrained_recurrent_v1"]
UPGRADE_STAGES = ["none", "m3m6", "m1m2", "m4m5", "final"]
ACTOR_DECOUPLE_MODES = ["weighted", "alternating_stopgrad"]

PUBLIC_OUTPUT_ROOT = os.path.join("results", "fix10E_fse_offline")
PUBLIC_STAGE0_BASELINE_ROOT = os.path.join(PUBLIC_OUTPUT_ROOT, "stage0_baseline")
PUBLIC_STAGE1_RAW_ROOT = os.path.join(PUBLIC_OUTPUT_ROOT, "stage1_raw")
PUBLIC_STAGE2_DATASET_ROOT = os.path.join(PUBLIC_OUTPUT_ROOT, "stage2_dataset")
PUBLIC_STAGE3_TRAIN_ROOT = os.path.join(PUBLIC_OUTPUT_ROOT, "stage3_train")
PUBLIC_STAGE4_EVAL_ROOT = os.path.join(PUBLIC_OUTPUT_ROOT, "stage4_eval")
PUBLIC_BASELINE_CHECKPOINT_PATH = os.path.join(PUBLIC_STAGE0_BASELINE_ROOT, "{mode}", "{seed}", "checkpoint.pt")
PUBLIC_BASELINE_TRAIN_LOG_PATH = os.path.join(PUBLIC_STAGE0_BASELINE_ROOT, "{mode}", "{seed}", "train_log.csv")
PUBLIC_BASELINE_TRAIN_JSON_PATH = os.path.join(PUBLIC_STAGE0_BASELINE_ROOT, "{mode}", "{seed}", "train_rows.json")
PUBLIC_BASELINE_RUN_RESULT_JSON_PATH = os.path.join(PUBLIC_STAGE0_BASELINE_ROOT, "{mode}", "{seed}", "run_result.json")
PUBLIC_BASELINE_TENSORBOARD_DIR = os.path.join(PUBLIC_STAGE0_BASELINE_ROOT, "{mode}", "{seed}", "tensorboard")
LOG_STD_MIN = -20
LOG_STD_MAX = 2

FSE_RISK_NAMES = ("collision", "unsafe_headway", "low_ttc", "offroad", "lane_boundary")
FSE_EXPOSURE_NAMES = ("unsafe_headway", "low_ttc", "offroad", "lane_boundary")
FSE_REGRESSION_NAMES = ("min_ttc", "min_front_distance", "min_boundary_margin", "progress", "safety_cost_sum")

COLLECT_ONLINE_TRAIN = 0
COLLECT_EVAL_POLICY = 1
COLLECT_RANDOM = 2
COLLECT_NOISY_EVAL = 3
FSE_COLLECT_POLICY_TO_ID = {
    "online_train": COLLECT_ONLINE_TRAIN,
    "eval_policy": COLLECT_EVAL_POLICY,
    "random": COLLECT_RANDOM,
    "noisy_eval": COLLECT_NOISY_EVAL,
}
FSE_COLLECT_ID_TO_POLICY = {int(v): str(k) for k, v in FSE_COLLECT_POLICY_TO_ID.items()}

SOURCE_GROUP_MAIN_CSAC = 0
SOURCE_GROUP_AUX_RANDOM_OOD = 1
SOURCE_GROUP_AUX_SAC_RISK = 2
SOURCE_GROUP_MIXED_ABLATION = 3

DONE_NONE = 0
DONE_COLLISION = 1
DONE_OFFROAD = 2
DONE_TIMEOUT = 3
DONE_GOAL = 4
DONE_OTHER = 5
DONE_REASON_ID_TO_NAME = {
    DONE_NONE: "none",
    DONE_COLLISION: "collision",
    DONE_OFFROAD: "offroad",
    DONE_TIMEOUT: "timeout",
    DONE_GOAL: "goal",
    DONE_OTHER: "other",
}

FSE_UNSAFE_HEADWAY_TAU = 1.2
FSE_LOW_TTC_TAU = 4.0
FSE_LANE_BOUNDARY_MARGIN_THRESHOLD = 0.3
FSE_LANE_OFFSET_RATIO_THRESHOLD = 0.45
FSE_COST_SAFETY_DEFINITION = "max(cost_collision,cost_headway,cost_lane)"
FSE_COST_LANE_DEFINITION = "current script cost_lane from compute_costs; verify no task-preference terms before using as safety exposure"
FSE_ACTION_SPACE_POLICY_NORMALIZED = "policy_normalized_tanh"
FSE_ACTION_ORDER = ("acceleration", "steering")
FSE_ACTION_RISK_WEIGHTS = (1.0, 0.7, 0.6, 0.9, 0.4)
FSE_ACTION_RISK_HORIZON_WEIGHTS = (0.50, 0.35, 0.15)


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
    # reward_type is frozen in this script to paper_formula.
    reward_type: str = "paper_formula"
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
class FSEConfig:
    task: str = "none"
    dataset_path: str = ""
    raw_path: str = ""
    output_dir: str = "results/fse"
    checkpoint_path: str = ""
    action_conditioned: bool = False
    action_risk_checkpoint_path: str = ""
    action_risk_acceptance_json: str = ""
    strict_action_gate: bool = False
    action_risk_beta: float = 0.05
    action_risk_uncertainty_kappa: float = 2.0
    action_risk_grad_diag_interval: int = 50
    action_risk_active_threshold: float = 0.05
    action_risk_gate_active_threshold: float = 0.10
    collect_policy: str = "online_train"
    policy_checkpoint_path: str = ""
    policy_checkpoint_path_template: str = ""
    policy_checkpoint_map: str = ""
    allow_untrained_eval_policy: bool = False
    source_ratio: str = "online_train=0.65,eval_policy=0.25,noisy_eval=0.10"
    source_balanced_sampler: bool = False
    noise_std_acc: float = 0.15
    noise_std_steer: float = 0.05
    noise_clip: float = 0.30
    split_stratify_by: str = "collect_policy,traffic_density,mode"
    horizons: Tuple[int, ...] = (10, 20, 40)
    z_dim: int = 64
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    focal_gamma: float = 2.0
    focal_alpha: Tuple[float, ...] = (0.75, 0.40, 0.50, 0.65, 0.40)
    lambda_exposure: float = 0.5
    lambda_reg: float = 0.25
    lambda_mono: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 128
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    risk_window_ratio: float = 0.30
    rare_window_ratio: float = 0.10
    threshold: float = 0.5
    fixed_fpr: float = 0.05
    fixed_recall: float = 0.80
    ece_bins: int = 10
    fusion_mode: str = "concat"
    z_mode: str = "real"
    random_z_distribution: str = "global_empirical_matched"
    random_z_std_floor: float = 1e-4
    run_tier: str = "smoke"
    allow_legacy_normalization: bool = False
    z_global_stats_path: str = ""
    shuffle_pool_path: str = ""
    z_artifacts: str = "both"
    z_artifact_source_split: str = "train_val"
    z_stats_output_path: str = ""
    shuffle_pool_output_path: str = ""
    z_artifact_max_samples: int = 0
    z_artifact_min_samples: int = 32
    z_trace_stride: int = 1
    z_trace_max_rows: int = 200000
    formal_min_z_artifact_samples: int = 32
    raw_state_dim: int = 0
    state_aug_dim: int = 0


@dataclass
class TrainConfig:
    seed: int = 42
    episodes: int = 10
    max_steps_per_episode: int = 400
    device: str = "cuda:0"
    mode: str = "paper_csac_lagrangian"
    render: bool = False
    print_every_step: bool = False
    enable_scenario_frame: bool = True
    scenario_frame_debug: bool = False
    scenario_frame_probe_interval: int = 0


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
    export_train_log: bool = True
    train_log_path: str = PUBLIC_BASELINE_TRAIN_LOG_PATH
    save_train_json: bool = True
    train_json_path: str = PUBLIC_BASELINE_TRAIN_JSON_PATH
    run_result_json_path: str = PUBLIC_BASELINE_RUN_RESULT_JSON_PATH
    checkpoint_save_path: str = PUBLIC_BASELINE_CHECKPOINT_PATH
    tensorboard_log_dir: str = PUBLIC_BASELINE_TENSORBOARD_DIR
    tensorboard_flush_secs: int = 10


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    fse: FSEConfig = field(default_factory=FSEConfig)
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


def is_fse_rl_mode(mode: str) -> bool:
    return str(mode).lower().strip() in FSE_RL_MODES


def is_fse_action_risk_mode(mode: str) -> bool:
    return str(mode).lower().strip() in FSE_ACTION_RISK_MODES


def is_fse_action_risk_shuffled_mode(mode: str) -> bool:
    return str(mode).lower().strip() == "paper_csac_fse_z_gated_action_risk_shuffled_tokens"


def stable_int_hash(*items: object) -> int:
    payload = "|".join(map(str, items)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def sha256_file(path: str) -> str:
    if not str(path or "").strip() or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_jsonable(value: Any) -> str:
    text = json.dumps(_jsonable_plain(value), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_trainable_params(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def count_all_params(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return int(sum(p.numel() for p in module.parameters()))


def count_nonzero_grad_params(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    count = 0
    for p in module.parameters():
        if p.grad is not None:
            count += int(torch.any(p.grad.detach() != 0).item())
    return int(count)


def model_weight_sha256(module: nn.Module) -> str:
    h = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items(), key=lambda kv: kv[0]):
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def env_action_to_policy_norm_tensor(action_env: torch.Tensor, action_scale: torch.Tensor, action_bias: torch.Tensor) -> torch.Tensor:
    return torch.clamp((action_env - action_bias) / torch.clamp(action_scale, min=1e-6), -1.0, 1.0)


def policy_norm_to_env_action_tensor(action_norm: torch.Tensor, action_scale: torch.Tensor, action_bias: torch.Tensor) -> torch.Tensor:
    return action_bias + action_scale * torch.clamp(action_norm, -1.0, 1.0)


def env_action_to_policy_norm_np(action_env: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    action_env = np.asarray(action_env, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    scale = np.maximum((high - low) * 0.5, 1e-6).astype(np.float32)
    bias = ((high + low) * 0.5).astype(np.float32)
    return np.clip((action_env - bias) / scale, -1.0, 1.0).astype(np.float32)


def default_policy_action_bounds(action_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    dim = max(1, int(action_dim))
    return -np.ones((dim,), dtype=np.float32), np.ones((dim,), dtype=np.float32)


def get_config() -> Config:
    cfg = Config()
    cfg.train.mode = str(cfg.train.mode).lower()
    cfg.fse.task = str(cfg.fse.task).lower()
    cfg.fse.collect_policy = str(cfg.fse.collect_policy).lower()
    cfg.sac.penalty_impl = str(cfg.sac.penalty_impl).lower()
    cfg.sac.upgrade_stage = str(cfg.sac.upgrade_stage).lower()
    cfg.sac.actor_decouple_mode = str(cfg.sac.actor_decouple_mode).lower()
    cfg.reward.reward_type = str(cfg.reward.reward_type).lower()
    return cfg


def validate_highway_only_config(cfg: Config) -> None:
    cfg.env.assert_highway_lock()


def configure_mode_behavior(cfg: Config, mode: Optional[str] = None) -> None:
    """Map paper experiment modes to reward type and agent family."""
    selected = str(mode if mode is not None else cfg.train.mode).lower().strip()
    if selected not in ALL_MODES:
        raise ValueError(f"Unsupported mode: {selected}. Choose from {ALL_MODES}.")
    cfg.train.mode = selected
    cfg.reward.reward_type = "paper_formula"
    # The new constrained modes use the compact Lagrangian CMDP agent below; legacy stage guards are disabled.
    cfg.sac.penalty_impl = "legacy_fixed_penalty"
    cfg.sac.upgrade_stage = "none"
    cfg.sac.enable_priority_safety_replay = False
    cfg.sac.enable_lagrangian_safety = selected.endswith("csac_lagrangian") or selected in FSE_RL_MODES
    cfg.sac.enable_recurrent_encoder = False
    cfg.sac.enable_decoupled_actor = False
    cfg.sac.enable_tail_risk_cvar = False
    if selected in FSE_RL_MODES:
        cfg.fse.task = "none"
        cfg.train.enable_scenario_frame = True
        expected_fusion, expected_z = FSE_RL_MODE_CONFIG[selected]
        if bool(getattr(cfg.fse, "_user_explicit_fusion_mode", False)) and str(cfg.fse.fusion_mode).lower().strip() != expected_fusion:
            raise ValueError(
                f"Mode {selected} requires --fse-fusion-mode {expected_fusion}, "
                f"got {cfg.fse.fusion_mode}."
            )
        if bool(getattr(cfg.fse, "_user_explicit_z_mode", False)) and str(cfg.fse.z_mode).lower().strip() != expected_z:
            raise ValueError(
                f"Mode {selected} requires --fse-z-mode {expected_z}, "
                f"got {cfg.fse.z_mode}."
            )
        cfg.fse.fusion_mode = expected_fusion
        cfg.fse.z_mode = expected_z


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
        "enable_scenario_frame": bool(cfg.train.enable_scenario_frame),
        "fse_task": str(cfg.fse.task),
        "fse_action_conditioned": bool(cfg.fse.action_conditioned),
        "fse_action_risk_enabled": bool(is_fse_action_risk_mode(cfg.train.mode)),
        "fse_action_risk_beta": float(cfg.fse.action_risk_beta),
        "fse_action_risk_uncertainty_kappa": float(cfg.fse.action_risk_uncertainty_kappa),
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


SCENARIO_SCHEMA_VERSION = "scenario_frame_v1_12x24"
SCENARIO_TOKEN_COUNT = 12
SCENARIO_TOKEN_DIM = 24
SCENARIO_SCALAR_CONTEXT_MODEL_DIM = 10

TOKEN_TYPE_EGO = 0
TOKEN_TYPE_NEIGHBOR = 1
TOKEN_TYPE_LANE = 2
TOKEN_TYPE_GOAL = 3

ROLE_EGO = 0
ROLE_FRONT_SAME = 1
ROLE_REAR_SAME = 2
ROLE_LEFT_FRONT = 3
ROLE_LEFT_REAR = 4
ROLE_RIGHT_FRONT = 5
ROLE_RIGHT_REAR = 6
ROLE_CURRENT_LANE = 7
ROLE_LEFT_LANE = 8
ROLE_RIGHT_LANE = 9
ROLE_GOAL_LANE = 10
ROLE_GOAL = 11

SCENARIO_NEIGHBOR_ROLES = [
    "front_same",
    "rear_same",
    "left_front",
    "left_rear",
    "right_front",
    "right_rear",
]
SCENARIO_LANE_ROLES = [
    "current_lane",
    "left_lane",
    "right_lane",
    "goal_lane",
]
SCENARIO_TOKEN_ORDER = ["ego"] + SCENARIO_NEIGHBOR_ROLES + SCENARIO_LANE_ROLES + ["goal"]
SCENARIO_TOKEN_ROLE_IDS = np.asarray(
    [
        ROLE_EGO,
        ROLE_FRONT_SAME,
        ROLE_REAR_SAME,
        ROLE_LEFT_FRONT,
        ROLE_LEFT_REAR,
        ROLE_RIGHT_FRONT,
        ROLE_RIGHT_REAR,
        ROLE_CURRENT_LANE,
        ROLE_LEFT_LANE,
        ROLE_RIGHT_LANE,
        ROLE_GOAL_LANE,
        ROLE_GOAL,
    ],
    dtype=np.int64,
)
SCENARIO_TOKEN_TYPE_IDS = np.asarray(
    [
        TOKEN_TYPE_EGO,
        TOKEN_TYPE_NEIGHBOR,
        TOKEN_TYPE_NEIGHBOR,
        TOKEN_TYPE_NEIGHBOR,
        TOKEN_TYPE_NEIGHBOR,
        TOKEN_TYPE_NEIGHBOR,
        TOKEN_TYPE_NEIGHBOR,
        TOKEN_TYPE_LANE,
        TOKEN_TYPE_LANE,
        TOKEN_TYPE_LANE,
        TOKEN_TYPE_LANE,
        TOKEN_TYPE_GOAL,
    ],
    dtype=np.int64,
)


@dataclass
class ScenarioFrame:
    tokens: np.ndarray
    token_mask: np.ndarray
    entity_valid_mask: np.ndarray
    token_type_ids: np.ndarray
    token_role_ids: np.ndarray
    scalar_context_model: np.ndarray
    diagnostic_context: Dict[str, Any]
    meta: Dict[str, Any]


class ScenarioFrameBuilder:
    """Pure scene-to-token adapter for future safety evidence models."""

    _NEIGHBOR_FIELD_MAP = {
        "front_same": (("front_distance",), ("front_rel_speed",), ("front_closing_speed",), ("front_ttc", "ttc"), 0.0, 1.0, 0.0),
        "rear_same": (("rear_distance",), ("rear_rel_speed",), ("rear_closing_speed",), ("rear_ttc",), 0.0, -1.0, 0.0),
        "left_front": (("left_front_distance",), ("left_front_rel_speed",), ("left_front_closing_speed",), ("left_front_ttc",), -1.0, 1.0, -1.0),
        "left_rear": (("left_rear_distance",), ("left_rear_rel_speed",), ("left_rear_closing_speed",), ("left_rear_ttc",), -1.0, -1.0, -1.0),
        "right_front": (("right_front_distance",), ("right_front_rel_speed",), ("right_front_closing_speed",), ("right_front_ttc",), 1.0, 1.0, 1.0),
        "right_rear": (("right_rear_distance",), ("right_rear_rel_speed",), ("right_rear_closing_speed",), ("right_rear_ttc",), 1.0, -1.0, 1.0),
    }
    _REQUIRED_FIELD_ALIASES = {
        "ego_speed": ("ego_speed",),
        "lane_id": ("ego_lane_id", "lane_id"),
        "goal_lane_id": ("goal_lane_id",),
        "goal_distance_remaining": ("goal_distance_remaining",),
        "lane_offset": ("lane_offset", "signed_lane_offset_norm"),
        "road_boundary_margin": ("road_boundary_margin",),
    }
    _OPTIONAL_FIELD_ALIASES = {
        "front_distance": ("front_distance",),
        "front_rel_speed": ("front_rel_speed",),
        "front_ttc": ("front_ttc", "ttc"),
        "rear_distance": ("rear_distance",),
        "rear_rel_speed": ("rear_rel_speed",),
        "rear_ttc": ("rear_ttc",),
        "left_front_distance": ("left_front_distance",),
        "left_front_rel_speed": ("left_front_rel_speed",),
        "left_front_ttc": ("left_front_ttc",),
        "left_rear_distance": ("left_rear_distance",),
        "left_rear_rel_speed": ("left_rear_rel_speed",),
        "left_rear_ttc": ("left_rear_ttc",),
        "right_front_distance": ("right_front_distance",),
        "right_front_rel_speed": ("right_front_rel_speed",),
        "right_front_ttc": ("right_front_ttc",),
        "right_rear_distance": ("right_rear_distance",),
        "right_rear_rel_speed": ("right_rear_rel_speed",),
        "right_rear_ttc": ("right_rear_ttc",),
        "left_lane_available": ("left_lane_available", "left_lane_id"),
        "right_lane_available": ("right_lane_available", "right_lane_id"),
        "goal_progress": ("goal_progress", "goal_longitudinal_progress"),
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lane_count = max(1, int(self.cfg.env.lanes_count))
        self._lane_denom = max(1.0, float(self._lane_count - 1))
        self._goal_distance = max(float(self.cfg.env.goal_distance), 1e-6)
        self._max_speed = max(float(self.cfg.env.max_speed), 1e-6)
        self._max_steps = int(getattr(self.cfg.train, "max_steps_per_episode", 0) or 0)
        self._boundary_soft = max(float(self.cfg.cost.road_boundary_soft_margin), 1e-6)
        self.total_frames = 0
        self.warning_counts: Dict[str, int] = defaultdict(int)
        self.missing_optional_counts: Dict[str, int] = defaultdict(int)
        self.missing_required_counts: Dict[str, int] = defaultdict(int)
        self.memory_keys_seen: set[str] = set()
        self._last_episode_id: Optional[int] = None
        self._last_role_vehicle_ids: Dict[str, Any] = {}
        self._role_switch_counts: Dict[str, int] = defaultdict(int)
        self._role_compare_counts: Dict[str, int] = defaultdict(int)
        self._last_frame_summary: Dict[str, Any] = {}

    def schema_probe(self) -> dict:
        return {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "token_shape": [SCENARIO_TOKEN_COUNT, SCENARIO_TOKEN_DIM],
            "scalar_context_model_dim": SCENARIO_SCALAR_CONTEXT_MODEL_DIM,
            "token_order": list(SCENARIO_TOKEN_ORDER),
            "token_type_ids": SCENARIO_TOKEN_TYPE_IDS.astype(int).tolist(),
            "token_role_ids": SCENARIO_TOKEN_ROLE_IDS.astype(int).tolist(),
            "token_mask_semantics": "role-slot validity; v1 fixed slots are all 1",
            "entity_valid_mask_semantics": "neighbor entity validity; lane/goal values mean slot validity, not lane availability",
        }

    def probe_summary(self) -> dict:
        switch_total = int(sum(self._role_switch_counts.values()))
        compare_total = int(sum(self._role_compare_counts.values()))
        switch_rate = float(switch_total / max(compare_total, 1))
        return {
            **self.schema_probe(),
            "total_frames": int(self.total_frames),
            "schema_warning_count": int(sum(self.warning_counts.values())),
            "warning_counts": {str(k): int(v) for k, v in sorted(self.warning_counts.items())},
            "missing_optional_fields": {str(k): int(v) for k, v in sorted(self.missing_optional_counts.items())},
            "missing_required_fields": {str(k): int(v) for k, v in sorted(self.missing_required_counts.items())},
            "memory_key_bin_count": int(len(self.memory_keys_seen)),
            "memory_keys_seen": sorted(self.memory_keys_seen),
            "role_identity_switch_count": switch_total,
            "role_identity_compare_count": compare_total,
            "role_identity_switch_rate": switch_rate,
            "role_identity_switch_counts": {str(k): int(v) for k, v in sorted(self._role_switch_counts.items())},
            "last_frame": dict(self._last_frame_summary),
        }

    def train_log_fields(self) -> dict:
        summary = self.probe_summary()
        return {
            "scenario_frame_schema_warning_count": float(summary["schema_warning_count"]),
            "scenario_frame_missing_optional_count": float(sum(summary["missing_optional_fields"].values())),
            "scenario_frame_memory_key_bin_count": float(summary["memory_key_bin_count"]),
            "scenario_frame_identity_switch_rate": float(summary["role_identity_switch_rate"]),
        }

    def _clip(self, value: float, low: float, high: float) -> float:
        return float(np.clip(float(value), float(low), float(high)))

    def _to_float(self, value: Any, fallback: float = 0.0) -> float:
        try:
            out = float(value)
            if not np.isfinite(out):
                return float(fallback)
            return out
        except Exception:
            return float(fallback)

    def _has_present_value(self, scene: dict, aliases: Tuple[str, ...]) -> bool:
        for key in aliases:
            if key in scene and scene.get(key) is not None:
                return True
        return False

    def _get_alias_value(self, scene: dict, aliases: Tuple[str, ...], fallback: Any = None) -> Any:
        for key in aliases:
            if key in scene and scene.get(key) is not None:
                return scene.get(key)
        return fallback

    def _get_float_alias(self, scene: dict, aliases: Tuple[str, ...], fallback: float) -> float:
        return self._to_float(self._get_alias_value(scene, aliases, fallback), fallback)

    def _to_lane_id(self, value: Any, fallback: Optional[int] = None) -> Optional[int]:
        if value is None:
            return fallback
        try:
            lid = int(value)
        except Exception:
            return fallback
        if lid < 0 or lid >= self._lane_count:
            return fallback
        return lid

    def _lane_id_is_valid(self, lane_id: Optional[int]) -> bool:
        return lane_id is not None and 0 <= int(lane_id) < self._lane_count

    def _norm_distance(self, distance: float) -> float:
        return self._clip(distance / 100.0, 0.0, 1.0)

    def _norm_ttc(self, ttc: float) -> float:
        return self._clip(ttc / 10.0, 0.0, 1.0)

    def _norm_rel_speed(self, rel_speed: float) -> float:
        return self._clip(rel_speed / 20.0, -1.0, 1.0)

    def _norm_closing_speed(self, closing_speed: float) -> float:
        return self._clip(closing_speed / 20.0, 0.0, 1.0)

    def _is_valid_entity(self, distance: float) -> float:
        return 1.0 if np.isfinite(distance) and 0.0 <= float(distance) < 1e5 else 0.0

    def _zero_token(self) -> np.ndarray:
        return np.zeros((SCENARIO_TOKEN_DIM,), dtype=np.float32)

    def _validate_schema(self, scene: dict) -> Tuple[List[str], List[str]]:
        missing_required: List[str] = []
        missing_optional: List[str] = []
        for name, aliases in self._REQUIRED_FIELD_ALIASES.items():
            if not self._has_present_value(scene, aliases):
                missing_required.append(name)
        for name, aliases in self._OPTIONAL_FIELD_ALIASES.items():
            if name in {"left_lane_available", "right_lane_available"}:
                missing = not any(key in scene for key in aliases)
            else:
                missing = not self._has_present_value(scene, aliases)
            if missing:
                missing_optional.append(name)
        for name in missing_required:
            self.missing_required_counts[name] += 1
        for name in missing_optional:
            self.missing_optional_counts[name] += 1
            self.warning_counts[f"missing_optional:{name}"] += 1
        if missing_required:
            for name in missing_required:
                self.warning_counts[f"missing_required:{name}"] += 1
            raise ValueError(f"ScenarioFrame required fields missing: {missing_required}")
        return missing_required, missing_optional

    def _lane_offset_norm(self, scene: dict, lane_width: float) -> float:
        if "lane_offset" in scene and scene.get("lane_offset") is not None:
            return self._clip(self._to_float(scene.get("lane_offset"), 0.0) / max(float(lane_width), 1e-6), -1.0, 1.0)
        return self._clip(self._to_float(scene.get("signed_lane_offset_norm"), 0.0), -1.0, 1.0)

    def _lane_available(self, scene: dict, side: str, lane_id: Optional[int]) -> float:
        explicit_key = f"{side}_lane_available"
        if explicit_key in scene and scene.get(explicit_key) is not None:
            return 1.0 if self._to_float(scene.get(explicit_key), 0.0) > 0.5 else 0.0
        return 1.0 if self._lane_id_is_valid(lane_id) else 0.0

    def _estimate_target_lane_boundary_margin(self, scene: dict, current_lane: int, target_lane: Optional[int]) -> float:
        target_lane_id = self._to_lane_id(target_lane, fallback=None)
        if target_lane_id is None:
            return 0.0
        lane_width = max(self._to_float(scene.get("lane_width_runtime", self.cfg.env.lane_width), self.cfg.env.lane_width), 1e-6)
        road_left = self._to_float(scene.get("road_left_margin", 0.0), 0.0)
        road_right = self._to_float(scene.get("road_right_margin", 0.0), 0.0)
        delta_lane = float(target_lane_id - int(current_lane))
        target_left = float(road_left + delta_lane * lane_width)
        target_right = float(road_right - delta_lane * lane_width)
        return max(0.0, min(target_left, target_right))

    def _traffic_one_hot(self, traffic_density: str) -> Tuple[float, float]:
        density = str(traffic_density).lower().strip()
        if density == "sparse":
            return 1.0, 0.0
        if density == "dense":
            return 0.0, 1.0
        self.warning_counts["unknown_traffic_density"] += 1
        return 0.0, 0.0

    def _step_progress(self, step_id: int) -> float:
        if self._max_steps <= 0:
            self.warning_counts["missing_max_episode_steps"] += 1
            return 0.0
        return self._clip(float(step_id) / float(self._max_steps), 0.0, 1.0)

    def _gap_bucket(self, available: float, front_gap: float, rear_gap: float) -> str:
        if float(available) <= 0.5:
            return "blocked"
        finite_gaps = [g for g in [front_gap, rear_gap] if np.isfinite(g) and float(g) < 1e5]
        gap = min(finite_gaps) if finite_gaps else 1e6
        if gap < 15.0:
            return "blocked"
        if gap < 30.0:
            return "cautious"
        return "free"

    def _front_bucket(self, distance: float) -> str:
        if not np.isfinite(distance) or float(distance) >= 1e5:
            return "none"
        if distance < 20.0:
            return "near"
        if distance < 50.0:
            return "mid"
        return "far"

    def _memory_key(self, ego_speed: float, front_distance: float, left_available: float, right_available: float, scene: dict) -> str:
        if ego_speed < 18.0:
            speed_bucket = "low"
        elif ego_speed <= 26.0:
            speed_bucket = "mid"
        else:
            speed_bucket = "high"
        left_bucket = self._gap_bucket(
            left_available,
            self._get_float_alias(scene, ("left_front_distance",), 1e6),
            self._get_float_alias(scene, ("left_rear_distance",), 1e6),
        )
        right_bucket = self._gap_bucket(
            right_available,
            self._get_float_alias(scene, ("right_front_distance",), 1e6),
            self._get_float_alias(scene, ("right_rear_distance",), 1e6),
        )
        return f"speed={speed_bucket}|front={self._front_bucket(front_distance)}|left={left_bucket}|right={right_bucket}"

    def _extract_role_vehicle_ids(self, scene: dict) -> Dict[str, Any]:
        role_ids = scene.get("role_vehicle_ids", {})
        if not isinstance(role_ids, dict):
            role_ids = {}
        out: Dict[str, Any] = {}
        for role in SCENARIO_NEIGHBOR_ROLES:
            value = role_ids.get(role, scene.get(f"{role}_vehicle_id", None))
            out[role] = value
        return out

    def _identity_switch_flags(self, episode_id: int, role_vehicle_ids: Dict[str, Any]) -> Dict[str, int]:
        if self._last_episode_id != int(episode_id):
            self._last_episode_id = int(episode_id)
            self._last_role_vehicle_ids = {}
        flags: Dict[str, int] = {}
        for role, current_id in role_vehicle_ids.items():
            previous_id = self._last_role_vehicle_ids.get(role, None)
            switched = int(previous_id is not None and current_id is not None and previous_id != current_id)
            compared = int(previous_id is not None and current_id is not None)
            if compared:
                self._role_compare_counts[role] += 1
            if switched:
                self._role_switch_counts[role] += 1
            flags[role] = switched
        self._last_role_vehicle_ids = dict(role_vehicle_ids)
        return flags

    def _lane_token(
        self,
        lane_available: float,
        front_gap: float,
        rear_gap: float,
        front_ttc: float,
        rear_ttc: float,
        boundary_margin: float,
        is_current_lane: float,
        is_goal_lane: float,
    ) -> np.ndarray:
        token = self._zero_token()
        available = 1.0 if float(lane_available) > 0.5 else 0.0
        if available <= 0.0:
            front_gap = 0.0
            rear_gap = 0.0
            front_ttc = 0.0
            rear_ttc = 0.0
            boundary_margin = 0.0
        token[:8] = np.asarray(
            [
                available,
                self._norm_distance(front_gap),
                self._norm_distance(rear_gap),
                self._norm_ttc(front_ttc),
                self._norm_ttc(rear_ttc),
                self._clip(float(boundary_margin) / self._boundary_soft, 0.0, 1.0),
                float(is_current_lane),
                float(is_goal_lane),
            ],
            dtype=np.float32,
        )
        return token

    def _validate_frame(self, frame: ScenarioFrame) -> None:
        if frame.tokens.shape != (SCENARIO_TOKEN_COUNT, SCENARIO_TOKEN_DIM):
            raise ValueError(
                f"ScenarioFrame.tokens shape mismatch: got {tuple(frame.tokens.shape)}, "
                f"expected {(SCENARIO_TOKEN_COUNT, SCENARIO_TOKEN_DIM)}."
            )
        for name, arr in {
            "token_mask": frame.token_mask,
            "entity_valid_mask": frame.entity_valid_mask,
            "token_type_ids": frame.token_type_ids,
            "token_role_ids": frame.token_role_ids,
        }.items():
            if arr.shape != (SCENARIO_TOKEN_COUNT,):
                raise ValueError(f"ScenarioFrame.{name} shape mismatch: got {tuple(arr.shape)}, expected {(SCENARIO_TOKEN_COUNT,)}.")
        if frame.scalar_context_model.shape != (SCENARIO_SCALAR_CONTEXT_MODEL_DIM,):
            raise ValueError(
                f"ScenarioFrame.scalar_context_model shape mismatch: got {tuple(frame.scalar_context_model.shape)}, "
                f"expected {(SCENARIO_SCALAR_CONTEXT_MODEL_DIM,)}."
            )
        if not np.all(np.isfinite(frame.tokens)):
            raise ValueError("ScenarioFrame.tokens contains NaN/Inf.")
        if not np.all(np.isfinite(frame.scalar_context_model)):
            raise ValueError("ScenarioFrame.scalar_context_model contains NaN/Inf.")
        if not np.all(np.isin(frame.token_mask, np.asarray([0.0, 1.0], dtype=np.float32))):
            raise ValueError("ScenarioFrame.token_mask must be binary in {0,1}.")
        if not np.all(np.isin(frame.entity_valid_mask, np.asarray([0.0, 1.0], dtype=np.float32))):
            raise ValueError("ScenarioFrame.entity_valid_mask must be binary in {0,1}.")
        if not np.array_equal(frame.token_type_ids, SCENARIO_TOKEN_TYPE_IDS):
            raise ValueError("ScenarioFrame.token_type_ids does not match the fixed v1 enum order.")
        if not np.array_equal(frame.token_role_ids, SCENARIO_TOKEN_ROLE_IDS):
            raise ValueError("ScenarioFrame.token_role_ids does not match the fixed v1 role order.")

    def build(
        self,
        obs: Optional[np.ndarray] = None,
        scene: Optional[Dict[str, Any]] = None,
        last_action: Optional[np.ndarray] = None,
        episode_id: int = 0,
        step_id: Optional[int] = None,
        traffic_density: Optional[str] = None,
        step_in_episode: Optional[int] = None,
    ) -> ScenarioFrame:
        if scene is None and isinstance(obs, dict):
            scene = obs
            obs = None
        if scene is None:
            raise ValueError("ScenarioFrameBuilder.build requires a scene dict.")
        if step_id is None:
            step_id = int(step_in_episode) if step_in_episode is not None else 0
        self._validate_schema(scene)

        lane_width_runtime = max(self._to_float(scene.get("lane_width_runtime", self.cfg.env.lane_width), self.cfg.env.lane_width), 1e-6)
        lane_id = self._to_lane_id(self._get_alias_value(scene, ("ego_lane_id", "lane_id"), 0), fallback=0) or 0
        left_lane_id = self._to_lane_id(scene.get("left_lane_id", None), fallback=None)
        right_lane_id = self._to_lane_id(scene.get("right_lane_id", None), fallback=None)
        goal_lane_id = self._to_lane_id(scene.get("goal_lane_id", self.cfg.env.goal_lane_id), fallback=self.cfg.env.goal_lane_id)
        if goal_lane_id is None:
            goal_lane_id = int(self.cfg.env.goal_lane_id)
        goal_lane_id = int(np.clip(int(goal_lane_id), 0, self._lane_count - 1))

        ego_speed = self._get_float_alias(scene, ("ego_speed",), 0.0)
        lane_offset_norm = self._lane_offset_norm(scene, lane_width_runtime)
        lateral_speed = self._get_float_alias(scene, ("lateral_speed",), 0.0)
        heading_error = self._get_float_alias(scene, ("heading_error",), 0.0)
        road_boundary_margin = self._get_float_alias(scene, ("road_boundary_margin",), 0.0)
        goal_distance_remaining = self._get_float_alias(scene, ("goal_distance_remaining",), self.cfg.env.goal_distance)
        goal_progress = self._get_float_alias(scene, ("goal_progress", "goal_longitudinal_progress"), 0.0)
        goal_reached_flag = self._clip(self._get_float_alias(scene, ("goal_reached",), 0.0), 0.0, 1.0)
        last_action_arr = np.asarray(
            last_action if last_action is not None else scene.get("last_action", np.zeros((2,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        last_acc = float(last_action_arr[0]) if last_action_arr.shape[0] > 0 else 0.0
        last_steer = float(last_action_arr[1]) if last_action_arr.shape[0] > 1 else 0.0

        ego_token = self._zero_token()
        ego_token[:10] = np.asarray(
            [
                self._clip(ego_speed / self._max_speed, 0.0, 1.5),
                self._clip(float(lane_id) / self._lane_denom, 0.0, 1.0),
                self._clip(lane_offset_norm, -1.0, 1.0),
                self._clip(lateral_speed / self._max_speed, -1.0, 1.0),
                self._clip(heading_error / 1.2, -1.0, 1.0),
                self._clip(road_boundary_margin / self._boundary_soft, 0.0, 1.0),
                self._clip(goal_distance_remaining / self._goal_distance, 0.0, 1.0),
                self._clip((float(goal_lane_id) - float(lane_id)) / self._lane_denom, -1.0, 1.0),
                self._clip(last_acc, -1.0, 1.0),
                self._clip(last_steer, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

        neighbor_tokens: List[np.ndarray] = []
        neighbor_entity_valid: List[float] = []
        neighbor_semantics: Dict[str, Dict[str, float]] = {}
        for role in SCENARIO_NEIGHBOR_ROLES:
            dist_keys, rel_keys, closing_keys, ttc_keys, lane_relation, front_or_rear_flag, left_or_right_flag = self._NEIGHBOR_FIELD_MAP[role]
            distance = self._get_float_alias(scene, dist_keys, 1e6)
            relative_speed = self._get_float_alias(scene, rel_keys, 0.0)
            closing_speed = self._get_float_alias(scene, closing_keys, max(0.0, -relative_speed))
            ttc = self._get_float_alias(scene, ttc_keys, 99.0)
            entity_valid = self._is_valid_entity(distance)
            token = self._zero_token()
            if entity_valid <= 0.0:
                token[:8] = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0, lane_relation, front_or_rear_flag, left_or_right_flag], dtype=np.float32)
                distance = 1e6
                relative_speed = 0.0
                closing_speed = 0.0
                ttc = 99.0
            else:
                token[:8] = np.asarray(
                    [
                        1.0,
                        self._norm_distance(distance),
                        self._norm_rel_speed(relative_speed),
                        self._norm_closing_speed(closing_speed),
                        self._norm_ttc(ttc),
                        float(lane_relation),
                        float(front_or_rear_flag),
                        float(left_or_right_flag),
                    ],
                    dtype=np.float32,
                )
            neighbor_tokens.append(token)
            neighbor_entity_valid.append(float(entity_valid))
            neighbor_semantics[role] = {
                "entity_valid": float(entity_valid),
                "distance": float(distance),
                "relative_speed": float(relative_speed),
                "closing_speed": float(closing_speed),
                "ttc": float(ttc),
                "lane_relation": float(lane_relation),
            }

        left_lane_available = self._lane_available(scene, "left", left_lane_id)
        right_lane_available = self._lane_available(scene, "right", right_lane_id)
        front_distance = self._get_float_alias(scene, ("front_distance",), 1e6)
        rear_distance = self._get_float_alias(scene, ("rear_distance",), 1e6)

        current_lane_token = self._lane_token(
            lane_available=1.0,
            front_gap=front_distance,
            rear_gap=rear_distance,
            front_ttc=self._get_float_alias(scene, ("front_ttc", "ttc"), 99.0),
            rear_ttc=self._get_float_alias(scene, ("rear_ttc",), 99.0),
            boundary_margin=road_boundary_margin,
            is_current_lane=1.0,
            is_goal_lane=1.0 if int(goal_lane_id) == int(lane_id) else 0.0,
        )
        left_lane_token = self._lane_token(
            lane_available=left_lane_available,
            front_gap=self._get_float_alias(scene, ("left_front_distance",), 1e6),
            rear_gap=self._get_float_alias(scene, ("left_rear_distance",), 1e6),
            front_ttc=self._get_float_alias(scene, ("left_front_ttc",), 99.0),
            rear_ttc=self._get_float_alias(scene, ("left_rear_ttc",), 99.0),
            boundary_margin=self._estimate_target_lane_boundary_margin(scene, current_lane=lane_id, target_lane=left_lane_id),
            is_current_lane=0.0,
            is_goal_lane=1.0 if left_lane_id is not None and int(goal_lane_id) == int(left_lane_id) else 0.0,
        )
        right_lane_token = self._lane_token(
            lane_available=right_lane_available,
            front_gap=self._get_float_alias(scene, ("right_front_distance",), 1e6),
            rear_gap=self._get_float_alias(scene, ("right_rear_distance",), 1e6),
            front_ttc=self._get_float_alias(scene, ("right_front_ttc",), 99.0),
            rear_ttc=self._get_float_alias(scene, ("right_rear_ttc",), 99.0),
            boundary_margin=self._estimate_target_lane_boundary_margin(scene, current_lane=lane_id, target_lane=right_lane_id),
            is_current_lane=0.0,
            is_goal_lane=1.0 if right_lane_id is not None and int(goal_lane_id) == int(right_lane_id) else 0.0,
        )

        if int(goal_lane_id) == int(lane_id):
            goal_lane_token = self._lane_token(1.0, front_distance, rear_distance, self._get_float_alias(scene, ("front_ttc", "ttc"), 99.0), self._get_float_alias(scene, ("rear_ttc",), 99.0), road_boundary_margin, 1.0, 1.0)
        elif left_lane_id is not None and int(goal_lane_id) == int(left_lane_id):
            goal_lane_token = self._lane_token(left_lane_available, self._get_float_alias(scene, ("left_front_distance",), 1e6), self._get_float_alias(scene, ("left_rear_distance",), 1e6), self._get_float_alias(scene, ("left_front_ttc",), 99.0), self._get_float_alias(scene, ("left_rear_ttc",), 99.0), self._estimate_target_lane_boundary_margin(scene, lane_id, left_lane_id), 0.0, 1.0)
        elif right_lane_id is not None and int(goal_lane_id) == int(right_lane_id):
            goal_lane_token = self._lane_token(right_lane_available, self._get_float_alias(scene, ("right_front_distance",), 1e6), self._get_float_alias(scene, ("right_rear_distance",), 1e6), self._get_float_alias(scene, ("right_front_ttc",), 99.0), self._get_float_alias(scene, ("right_rear_ttc",), 99.0), self._estimate_target_lane_boundary_margin(scene, lane_id, right_lane_id), 0.0, 1.0)
        else:
            goal_lane_token = self._lane_token(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        goal_token = self._zero_token()
        goal_token[:4] = np.asarray(
            [
                self._clip(goal_distance_remaining / self._goal_distance, 0.0, 1.0),
                self._clip((float(goal_lane_id) - float(lane_id)) / self._lane_denom, -1.0, 1.0),
                float(goal_reached_flag),
                self._clip(goal_progress / self._goal_distance, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        density = str(traffic_density if traffic_density is not None else scene.get("traffic_density", self.cfg.env.traffic_density))
        sparse_flag, dense_flag = self._traffic_one_hot(density)
        scalar_context_model = np.asarray(
            [
                self._clip(last_acc, -1.0, 1.0),
                self._clip(last_steer, -1.0, 1.0),
                sparse_flag,
                dense_flag,
                self._step_progress(int(step_id)),
                self._clip(self._get_float_alias(scene, ("road_boundary_proximity",), 0.0), 0.0, 1.0),
                self._clip(self._get_float_alias(scene, ("lane_boundary_proximity",), 0.0), 0.0, 1.0),
                left_lane_available,
                right_lane_available,
                self._clip(self._get_float_alias(scene, ("recent_lane_change_flag",), 0.0), 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        lane_tokens = [current_lane_token, left_lane_token, right_lane_token, goal_lane_token]
        tokens = np.stack([ego_token] + neighbor_tokens + lane_tokens + [goal_token], axis=0).astype(np.float32)
        token_mask = np.ones((SCENARIO_TOKEN_COUNT,), dtype=np.float32)
        entity_valid_mask = np.asarray([1.0] + neighbor_entity_valid + [1.0, 1.0, 1.0, 1.0] + [1.0], dtype=np.float32)

        role_vehicle_ids = self._extract_role_vehicle_ids(scene)
        role_switch_flags = self._identity_switch_flags(int(episode_id), role_vehicle_ids)
        compare_total = max(int(sum(self._role_compare_counts.values())), 1)
        switch_rate = float(sum(self._role_switch_counts.values()) / compare_total)
        memory_key = self._memory_key(ego_speed, front_distance, left_lane_available, right_lane_available, scene)
        self.memory_keys_seen.add(memory_key)

        offroad = bool(
            scene.get("offroad", False)
            or self._get_float_alias(scene, ("road_boundary_margin",), 1e6) < float(self.cfg.cost.road_boundary_off_margin)
            or abs(self._get_float_alias(scene, ("abs_lane_offset_norm",), abs(lane_offset_norm))) > 1.10
        )
        diagnostic_context = {
            "collision": bool(scene.get("collision", False)),
            "offroad": bool(offroad),
            "crashed": bool(scene.get("crashed", scene.get("collision", False))),
            "done_reason": str(scene.get("done_reason", "")),
            "missing_optional_fields": [
                name
                for name, aliases in self._OPTIONAL_FIELD_ALIASES.items()
                if (
                    (name in {"left_lane_available", "right_lane_available"} and not any(key in scene for key in aliases))
                    or (name not in {"left_lane_available", "right_lane_available"} and not self._has_present_value(scene, aliases))
                )
            ],
            "schema_warning_count": int(sum(self.warning_counts.values())),
            "role_vehicle_ids": dict(role_vehicle_ids),
            "role_identity_switch_flags": dict(role_switch_flags),
            "role_identity_switch_rate": float(switch_rate),
            "neighbor_semantics": neighbor_semantics,
        }
        meta = {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "step_in_episode": int(step_id),
            "traffic_density": density,
            "memory_key": memory_key,
            "role_vehicle_ids": dict(role_vehicle_ids),
            "role_identity_switch_flags": dict(role_switch_flags),
            "token_order": list(SCENARIO_TOKEN_ORDER),
        }
        frame = ScenarioFrame(
            tokens=tokens,
            token_mask=token_mask,
            entity_valid_mask=entity_valid_mask,
            token_type_ids=SCENARIO_TOKEN_TYPE_IDS.copy(),
            token_role_ids=SCENARIO_TOKEN_ROLE_IDS.copy(),
            scalar_context_model=scalar_context_model,
            diagnostic_context=diagnostic_context,
            meta=meta,
        )
        self._validate_frame(frame)
        self.total_frames += 1
        self._last_frame_summary = {
            "episode_id": int(episode_id),
            "step_id": int(step_id),
            "memory_key": memory_key,
            "entity_valid_sum": float(np.sum(entity_valid_mask)),
            "schema_warning_count": int(sum(self.warning_counts.values())),
            "role_identity_switch_rate": float(switch_rate),
        }
        return frame


def print_scenario_frame_schema_probe(builder: ScenarioFrameBuilder, prefix: str = "TRAIN") -> None:
    probe = builder.schema_probe()
    print(
        f"[SCENARIO-FRAME-SCHEMA][{prefix}] version={probe['schema_version']} "
        f"tokens_shape=({SCENARIO_TOKEN_COUNT},{SCENARIO_TOKEN_DIM}) "
        f"scalar_context_model_dim={SCENARIO_SCALAR_CONTEXT_MODEL_DIM}"
    )


def print_scenario_frame_probe(frame: ScenarioFrame, scene: dict, detailed: bool = False) -> None:
    valid_neighbor_tokens = int(np.sum(frame.entity_valid_mask[1: 1 + len(SCENARIO_NEIGHBOR_ROLES)]))
    print(
        f"[SCENARIO-FRAME] tokens_shape={tuple(frame.tokens.shape)} "
        f"mask_sum={float(np.sum(frame.token_mask)):.0f} "
        f"entity_valid_sum={float(np.sum(frame.entity_valid_mask)):.0f} "
        f"valid_neighbors={valid_neighbor_tokens} "
        f"memory_key={frame.meta.get('memory_key', '')} "
        f"schema_warnings={int(frame.diagnostic_context.get('schema_warning_count', 0))} "
        f"identity_switch_rate={float(frame.diagnostic_context.get('role_identity_switch_rate', 0.0)):.4f} "
        f"goal_distance_remaining={float(scene.get('goal_distance_remaining', 0.0)):.1f} "
        f"front_distance={float(scene.get('front_distance', 0.0)):.2f} "
        f"ttc={float(scene.get('ttc', scene.get('front_ttc', 0.0))):.2f}"
    )
    if detailed:
        print(
            f"[SCENARIO-FRAME-DETAIL] missing_optional={frame.diagnostic_context.get('missing_optional_fields', [])} "
            f"role_switch_flags={frame.diagnostic_context.get('role_identity_switch_flags', {})}"
        )


@dataclass
class FSEBottleneckOutput:
    z_fse: torch.Tensor
    risk_logits: torch.Tensor
    risk_probs: torch.Tensor
    exposure_pred: torch.Tensor
    reg_pred: torch.Tensor
    uncertainty: torch.Tensor


@dataclass
class FSELossBreakdown:
    total: torch.Tensor
    focal_binary: torch.Tensor
    exposure: torch.Tensor
    regression: torch.Tensor
    monotonic: torch.Tensor


class FSEBottleneckTransformer(nn.Module):
    """Semantic-grounded future-safety bottleneck.

    The bottleneck is trained only through lightweight semantic decoder heads in
    this first implementation: no contrastive, policy, or VIB/KL objective is
    added here.
    """

    def __init__(
        self,
        fse_cfg: FSEConfig,
        token_dim: int = SCENARIO_TOKEN_DIM,
        n_tokens: int = SCENARIO_TOKEN_COUNT,
        action_dim: int = 0,
    ):
        super().__init__()
        self.fse_cfg = copy.deepcopy(fse_cfg)
        self.token_dim = int(token_dim)
        self.n_tokens = int(n_tokens)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(fse_cfg.hidden_dim)
        self.z_dim = int(fse_cfg.z_dim)
        self.horizons = tuple(int(v) for v in fse_cfg.horizons)
        self.entity_mask_mismatch_count = 0

        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.token_type_embed = nn.Embedding(4, self.hidden_dim)
        self.role_embed = nn.Embedding(max(SCENARIO_TOKEN_COUNT, self.n_tokens), self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.action_token_bias = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.action_proj = nn.Linear(self.action_dim, self.hidden_dim) if self.action_dim > 0 else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(fse_cfg.n_heads),
            dim_feedforward=4 * self.hidden_dim,
            dropout=float(fse_cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        try:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(fse_cfg.n_layers), enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(fse_cfg.n_layers))
        self.bottleneck = nn.Sequential(
            nn.Linear(self.hidden_dim, self.z_dim),
            nn.LayerNorm(self.z_dim),
            nn.Dropout(float(fse_cfg.dropout)),
        )
        head_hidden = max(32, self.hidden_dim)
        self.risk_head = nn.Sequential(
            nn.Linear(self.z_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, len(self.horizons) * len(FSE_RISK_NAMES)),
        )
        self.exposure_head = nn.Sequential(
            nn.Linear(self.z_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, len(self.horizons) * len(FSE_EXPOSURE_NAMES)),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(self.z_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, len(self.horizons) * len(FSE_REGRESSION_NAMES)),
        )

    def _expand_ids(self, ids: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        if ids is None:
            raise ValueError(f"FSE forward requires {name}.")
        if ids.dim() == 1:
            ids = ids.unsqueeze(0).expand(batch_size, -1)
        if ids.dim() != 2 or ids.shape[0] != batch_size or ids.shape[1] != self.n_tokens:
            raise ValueError(f"{name} must have shape [B,{self.n_tokens}] or [{self.n_tokens}], got {tuple(ids.shape)}.")
        return ids.long()

    def _check_entity_mask(self, tokens: torch.Tensor, entity_valid_mask: torch.Tensor) -> None:
        if tokens.shape[1] < 1 + len(SCENARIO_NEIGHBOR_ROLES):
            return
        with torch.no_grad():
            neighbor_valid_flag = tokens[:, 1: 1 + len(SCENARIO_NEIGHBOR_ROLES), 0] > 0.5
            neighbor_entity_mask = entity_valid_mask[:, 1: 1 + len(SCENARIO_NEIGHBOR_ROLES)] > 0.5
            mismatch = torch.logical_xor(neighbor_valid_flag, neighbor_entity_mask)
            self.entity_mask_mismatch_count += int(mismatch.sum().item())

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        entity_valid_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        token_role_ids: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> FSEBottleneckOutput:
        if entity_valid_mask is None:
            raise ValueError("FSE forward requires entity_valid_mask; it is not optional in schema v1.")
        if tokens.dim() != 3 or tokens.shape[1] != self.n_tokens or tokens.shape[2] != self.token_dim:
            raise ValueError(f"tokens must have shape [B,{self.n_tokens},{self.token_dim}], got {tuple(tokens.shape)}.")
        batch_size = int(tokens.shape[0])
        if token_mask is None:
            raise ValueError("FSE forward requires token_mask.")
        if token_mask.dim() == 1:
            token_mask = token_mask.unsqueeze(0).expand(batch_size, -1)
        if entity_valid_mask.dim() == 1:
            entity_valid_mask = entity_valid_mask.unsqueeze(0).expand(batch_size, -1)
        if token_mask.shape != (batch_size, self.n_tokens):
            raise ValueError(f"token_mask must have shape [B,{self.n_tokens}], got {tuple(token_mask.shape)}.")
        if entity_valid_mask.shape != (batch_size, self.n_tokens):
            raise ValueError(f"entity_valid_mask must have shape [B,{self.n_tokens}], got {tuple(entity_valid_mask.shape)}.")
        if self.action_dim <= 0 and action is not None:
            raise ValueError("This state-conditioned FSE was created with action_dim=0; do not pass action.")
        if self.action_dim > 0:
            if action is None:
                raise ValueError("This action-conditioned FSE requires action.")
            if action.dim() != 2 or action.shape[0] != batch_size or action.shape[1] != self.action_dim:
                raise ValueError(f"action must have shape [B,{self.action_dim}], got {tuple(action.shape)}.")

        token_type_ids = self._expand_ids(token_type_ids, batch_size, "token_type_ids").to(tokens.device)
        token_role_ids = self._expand_ids(token_role_ids, batch_size, "token_role_ids").to(tokens.device)
        token_mask = token_mask.to(device=tokens.device, dtype=torch.float32)
        entity_valid_mask = entity_valid_mask.to(device=tokens.device, dtype=torch.float32)
        self._check_entity_mask(tokens, entity_valid_mask)

        token_features = torch.cat([tokens, entity_valid_mask.unsqueeze(-1)], dim=-1)
        token_emb = self.token_mlp(token_features)
        token_emb = token_emb + self.token_type_embed(token_type_ids.clamp(min=0, max=3))
        token_emb = token_emb + self.role_embed(token_role_ids.clamp(min=0, max=self.role_embed.num_embeddings - 1))

        cls = self.cls_token.expand(batch_size, -1, -1)
        seq_parts = [cls, token_emb]
        pad_parts = [
            torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device),
            token_mask <= 0.5,
        ]
        if self.action_dim > 0 and self.action_proj is not None:
            action_token = self.action_proj(action)[:, None, :] + self.action_token_bias
            seq_parts.append(action_token)
            pad_parts.append(torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device))
        seq = torch.cat(seq_parts, dim=1)
        padding_mask = torch.cat(pad_parts, dim=1)
        encoded = self.encoder(seq, src_key_padding_mask=padding_mask)
        z_fse = self.bottleneck(encoded[:, 0, :])

        n_h = len(self.horizons)
        risk_logits = self.risk_head(z_fse).view(batch_size, n_h, len(FSE_RISK_NAMES))
        risk_probs = torch.sigmoid(risk_logits)
        exposure_pred = torch.sigmoid(self.exposure_head(z_fse).view(batch_size, n_h, len(FSE_EXPOSURE_NAMES)))
        reg_pred = torch.sigmoid(self.reg_head(z_fse).view(batch_size, n_h, len(FSE_REGRESSION_NAMES)))
        eps = 1e-6
        entropy = -(risk_probs * torch.log(risk_probs + eps) + (1.0 - risk_probs) * torch.log(1.0 - risk_probs + eps))
        uncertainty = torch.clamp(entropy.mean(dim=-1, keepdim=True) / math.log(2.0), 0.0, 1.0)
        return FSEBottleneckOutput(
            z_fse=z_fse,
            risk_logits=risk_logits,
            risk_probs=risk_probs,
            exposure_pred=exposure_pred,
            reg_pred=reg_pred,
            uncertainty=uncertainty,
        )


def fse_masked_mean(value: torch.Tensor, valid_mask: torch.Tensor, trailing_dim: int) -> torch.Tensor:
    mask = valid_mask.to(dtype=value.dtype)
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    denom = torch.clamp(mask.sum() * float(trailing_dim), min=1.0)
    return (value * mask).sum() / denom


def compute_fse_loss(
    out: FSEBottleneckOutput,
    labels_binary: torch.Tensor,
    labels_exposure: torch.Tensor,
    labels_reg: torch.Tensor,
    valid_mask: torch.Tensor,
    fse_cfg: FSEConfig,
) -> FSELossBreakdown:
    labels_binary = labels_binary.float()
    labels_exposure = labels_exposure.float()
    labels_reg = labels_reg.float()
    valid_mask = valid_mask.float()

    bce = F.binary_cross_entropy_with_logits(out.risk_logits, labels_binary, reduction="none")
    pt = torch.exp(-bce)
    alpha = torch.as_tensor(fse_cfg.focal_alpha, dtype=bce.dtype, device=bce.device).view(1, 1, -1)
    alpha_t = labels_binary * alpha + (1.0 - labels_binary) * (1.0 - alpha)
    focal = alpha_t * torch.pow(1.0 - pt, float(fse_cfg.focal_gamma)) * bce
    focal_binary = fse_masked_mean(focal, valid_mask, trailing_dim=len(FSE_RISK_NAMES))

    exposure_loss_raw = F.smooth_l1_loss(out.exposure_pred, labels_exposure, reduction="none")
    exposure_loss = fse_masked_mean(exposure_loss_raw, valid_mask, trailing_dim=len(FSE_EXPOSURE_NAMES))

    reg_loss_raw = F.smooth_l1_loss(out.reg_pred, labels_reg, reduction="none")
    reg_loss = fse_masked_mean(reg_loss_raw, valid_mask, trailing_dim=len(FSE_REGRESSION_NAMES))

    mono_terms: List[torch.Tensor] = []
    mono_masks: List[torch.Tensor] = []
    for i in range(max(0, out.risk_probs.shape[1] - 1)):
        mono_terms.append(F.relu(out.risk_probs[:, i, :] - out.risk_probs[:, i + 1, :]))
        mono_masks.append(valid_mask[:, i] * valid_mask[:, i + 1])
    if mono_terms:
        mono_value = torch.stack(mono_terms, dim=1)
        mono_mask = torch.stack(mono_masks, dim=1)
        mono_loss = fse_masked_mean(mono_value, mono_mask, trailing_dim=len(FSE_RISK_NAMES))
    else:
        mono_loss = torch.tensor(0.0, dtype=out.risk_probs.dtype, device=out.risk_probs.device)

    total = (
        focal_binary
        + float(fse_cfg.lambda_exposure) * exposure_loss
        + float(fse_cfg.lambda_reg) * reg_loss
        + float(fse_cfg.lambda_mono) * mono_loss
    )
    return FSELossBreakdown(
        total=total,
        focal_binary=focal_binary,
        exposure=exposure_loss,
        regression=reg_loss,
        monotonic=mono_loss,
    )


def _fse_required_keys() -> Tuple[str, ...]:
    return (
        "tokens",
        "token_mask",
        "token_type_ids",
        "token_role_ids",
        "step_id",
        "labels_binary",
        "labels_exposure",
        "labels_reg",
        "valid_mask",
    )


def _as_fse_array(npz, key: str, dtype=None):
    arr = np.asarray(npz[key])
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def load_fse_npz_dataset(path: str, fse_cfg: FSEConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not str(path).strip():
        raise ValueError("--fse-dataset-path is required for fse train/eval.")
    with np.load(path, allow_pickle=True) as npz:
        available = set(npz.files)
        missing = [key for key in _fse_required_keys() if key not in available]
        if missing:
            raise ValueError(f"FSE dataset missing required key(s): {missing}")
        data = {
            "tokens": _as_fse_array(npz, "tokens", np.float32),
            "token_mask": _as_fse_array(npz, "token_mask", np.float32),
            "token_type_ids": _as_fse_array(npz, "token_type_ids", np.int64),
            "token_role_ids": _as_fse_array(npz, "token_role_ids", np.int64),
            "step_id": _as_fse_array(npz, "step_id", np.int64).reshape(-1),
            "labels_binary": _as_fse_array(npz, "labels_binary", np.float32),
            "labels_exposure": _as_fse_array(npz, "labels_exposure", np.float32),
            "labels_reg": _as_fse_array(npz, "labels_reg", np.float32),
            "valid_mask": _as_fse_array(npz, "valid_mask", np.float32),
        }
        warnings: Dict[str, Any] = {"legacy_entity_valid_mask_reconstructed": False, "entity_valid_mismatch_count": 0}
        if "episode_uid_id" in available:
            data["episode_uid_id"] = _as_fse_array(npz, "episode_uid_id", np.int64).reshape(-1)
            data["episode_id"] = data["episode_uid_id"]
        elif "episode_id" in available:
            data["episode_id"] = _as_fse_array(npz, "episode_id", np.int64).reshape(-1)
            data["episode_uid_id"] = data["episode_id"]
        else:
            raise ValueError("FSE dataset requires episode_uid_id (preferred) or legacy episode_id.")
        if "entity_valid_mask" in available:
            data["entity_valid_mask"] = _as_fse_array(npz, "entity_valid_mask", np.float32)
        else:
            # Kept for old Step-2 artifacts only. New module-2 datasets must include this key.
            entity_valid = np.ones((data["tokens"].shape[0], SCENARIO_TOKEN_COUNT), dtype=np.float32)
            entity_valid[:, 1: 1 + len(SCENARIO_NEIGHBOR_ROLES)] = (
                data["tokens"][:, 1: 1 + len(SCENARIO_NEIGHBOR_ROLES), 0] > 0.5
            ).astype(np.float32)
            data["entity_valid_mask"] = entity_valid
            warnings["legacy_entity_valid_mask_reconstructed"] = True
        if "actions" in available:
            data["actions"] = _as_fse_array(npz, "actions", np.float32)
        if "actions_policy_norm" in available:
            data["actions_policy_norm"] = _as_fse_array(npz, "actions_policy_norm", np.float32)
        if "memory_key" in available:
            data["memory_key"] = np.asarray(npz["memory_key"])
        for key in (
            "episode_id_local",
            "mode_id",
            "seed_id",
            "raw_source_id",
            "env_seed_id",
            "checkpoint_seed_id",
            "policy_train_seed_id",
            "collect_policy_id",
            "source_group_id",
            "policy_checkpoint_uid_id",
            "traffic_density_id",
        ):
            if key in available:
                data[key] = _as_fse_array(npz, key, np.int64).reshape(-1)

    n = int(data["tokens"].shape[0])
    if data["tokens"].shape != (n, SCENARIO_TOKEN_COUNT, SCENARIO_TOKEN_DIM):
        raise ValueError(f"tokens must have shape [N,{SCENARIO_TOKEN_COUNT},{SCENARIO_TOKEN_DIM}], got {data['tokens'].shape}.")
    if data["token_mask"].shape == (SCENARIO_TOKEN_COUNT,):
        data["token_mask"] = np.repeat(data["token_mask"][None, :], n, axis=0).astype(np.float32)
    if data["token_type_ids"].shape == (SCENARIO_TOKEN_COUNT,):
        data["token_type_ids"] = np.repeat(data["token_type_ids"][None, :], n, axis=0).astype(np.int64)
    if data["token_role_ids"].shape == (SCENARIO_TOKEN_COUNT,):
        data["token_role_ids"] = np.repeat(data["token_role_ids"][None, :], n, axis=0).astype(np.int64)
    expected_h = len(tuple(fse_cfg.horizons))
    shape_checks = {
        "token_mask": (n, SCENARIO_TOKEN_COUNT),
        "entity_valid_mask": (n, SCENARIO_TOKEN_COUNT),
        "token_type_ids": (n, SCENARIO_TOKEN_COUNT),
        "token_role_ids": (n, SCENARIO_TOKEN_COUNT),
        "labels_binary": (n, expected_h, len(FSE_RISK_NAMES)),
        "labels_exposure": (n, expected_h, len(FSE_EXPOSURE_NAMES)),
        "labels_reg": (n, expected_h, len(FSE_REGRESSION_NAMES)),
        "valid_mask": (n, expected_h),
    }
    for key, expected in shape_checks.items():
        if tuple(data[key].shape) != tuple(expected):
            raise ValueError(f"{key} must have shape {expected}, got {tuple(data[key].shape)}.")
    if data["episode_id"].shape[0] != n or data["episode_uid_id"].shape[0] != n or data["step_id"].shape[0] != n:
        raise ValueError("episode_uid_id/episode_id and step_id must have length N.")
    if "actions_policy_norm" in data:
        if data["actions_policy_norm"].ndim != 2 or data["actions_policy_norm"].shape[0] != n:
            raise ValueError("actions_policy_norm must have shape [N, action_dim].")
        data["actions_for_fse"] = np.clip(data["actions_policy_norm"].astype(np.float32), -1.0, 1.0)
    elif "actions" in data:
        if data["actions"].ndim != 2 or data["actions"].shape[0] != n:
            raise ValueError("actions must have shape [N, action_dim].")
        data["actions_for_fse"] = data["actions"].astype(np.float32)
        warnings["legacy_actions_used_for_action_fse"] = True
    neighbor_flag = data["tokens"][:, 1: 1 + len(SCENARIO_NEIGHBOR_ROLES), 0] > 0.5
    neighbor_mask = data["entity_valid_mask"][:, 1: 1 + len(SCENARIO_NEIGHBOR_ROLES)] > 0.5
    warnings["entity_valid_mismatch_count"] = int(np.logical_xor(neighbor_flag, neighbor_mask).sum())
    return data, warnings


def split_fse_by_episode(data: Dict[str, np.ndarray], seed: int, fse_cfg: FSEConfig) -> Dict[str, np.ndarray]:
    ep_key = "episode_uid_id" if "episode_uid_id" in data else "episode_id"
    episodes = np.unique(data[ep_key].astype(np.int64))
    if episodes.shape[0] < 3:
        raise ValueError(f"FSE train/val/test split requires at least 3 distinct {ep_key} values.")
    rng = np.random.default_rng(int(seed))

    # Episode-level split, with a lightweight source stratification when the
    # dataset carries collect_policy_id. Small buckets fall back into train and
    # are reported by build-dataset in fse_split.json.
    episode_groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    ep_to_first_idx: Dict[int, int] = {}
    for i, ep in enumerate(data[ep_key].astype(np.int64).tolist()):
        ep_to_first_idx.setdefault(int(ep), int(i))
    requested = [s.strip().lower() for s in str(getattr(fse_cfg, "split_stratify_by", "")).split(",") if s.strip()]

    def group_key_for_episode(ep: int) -> Tuple[int, ...]:
        i = ep_to_first_idx[int(ep)]
        values: List[int] = []
        for name in requested:
            if name in ("collect_policy", "collect_policy_id") and "collect_policy_id" in data:
                values.append(int(data["collect_policy_id"][i]))
            elif name in ("traffic_density", "traffic_density_id") and "traffic_density_id" in data:
                values.append(int(data["traffic_density_id"][i]))
            elif name in ("mode", "mode_id") and "mode_id" in data:
                values.append(int(data["mode_id"][i]))
        return tuple(values) if values else (0,)

    for ep in episodes.tolist():
        episode_groups[group_key_for_episode(int(ep))].append(int(ep))

    train_eps: set[int] = set()
    val_eps: set[int] = set()
    test_eps: set[int] = set()

    def split_bucket(bucket: List[int]) -> Tuple[List[int], List[int], List[int]]:
        local = np.asarray(bucket, dtype=np.int64)
        rng.shuffle(local)
        n_ep = int(local.shape[0])
        if n_ep < 3:
            return local.tolist(), [], []
        n_val = max(1, int(round(n_ep * float(fse_cfg.val_split))))
        n_test = max(1, int(round(n_ep * float(fse_cfg.test_split))))
        if n_val + n_test >= n_ep:
            n_val = 1
            n_test = 1
        n_train = n_ep - n_val - n_test
        return (
            [int(v) for v in local[:n_train].tolist()],
            [int(v) for v in local[n_train: n_train + n_val].tolist()],
            [int(v) for v in local[n_train + n_val:].tolist()],
        )

    for _, bucket_eps in episode_groups.items():
        tr, va, te = split_bucket(bucket_eps)
        train_eps.update(tr)
        val_eps.update(va)
        test_eps.update(te)

    if not val_eps or not test_eps:
        episodes2 = episodes.copy()
        rng.shuffle(episodes2)
        n_ep = int(episodes2.shape[0])
        n_val = max(1, int(round(n_ep * float(fse_cfg.val_split))))
        n_test = max(1, int(round(n_ep * float(fse_cfg.test_split))))
        if n_val + n_test >= n_ep:
            n_val = 1
            n_test = 1
        n_train = n_ep - n_val - n_test
        train_eps = set(int(v) for v in episodes2[:n_train].tolist())
        val_eps = set(int(v) for v in episodes2[n_train: n_train + n_val].tolist())
        test_eps = set(int(v) for v in episodes2[n_train + n_val:].tolist())

    split = {}
    for name, ep_set in [("train", train_eps), ("val", val_eps), ("test", test_eps)]:
        mask = np.asarray([int(ep) in ep_set for ep in data[ep_key].tolist()], dtype=bool)
        split[name] = np.where(mask)[0].astype(np.int64)
    overlap = (train_eps & val_eps) or (train_eps & test_eps) or (val_eps & test_eps)
    if overlap:
        raise RuntimeError(f"Episode split overlap detected: {sorted(overlap)}")
    return split


def _fse_risk_masks(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    labels = data["labels_binary"] > 0.5
    valid = data["valid_mask"] > 0.5
    valid3 = valid[:, :, None]
    any_risk = np.logical_and(labels, valid3).any(axis=(1, 2))
    rare_indices = [FSE_RISK_NAMES.index("collision"), FSE_RISK_NAMES.index("low_ttc"), FSE_RISK_NAMES.index("offroad")]
    rare_risk = np.logical_and(labels[:, :, rare_indices], valid3).any(axis=(1, 2))
    return any_risk, rare_risk


def sample_fse_priority_batch(
    data: Dict[str, np.ndarray],
    split_indices: np.ndarray,
    batch_size: int,
    fse_cfg: FSEConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    split_indices = np.asarray(split_indices, dtype=np.int64)
    any_risk, rare_risk = _fse_risk_masks(data)
    risk_pool = split_indices[any_risk[split_indices]]
    rare_pool = split_indices[rare_risk[split_indices]]
    n_rare = int(math.ceil(int(batch_size) * float(fse_cfg.rare_window_ratio)))
    n_risk_total = int(math.ceil(int(batch_size) * float(fse_cfg.risk_window_ratio)))
    n_risk_extra = max(0, n_risk_total - n_rare)
    fallback_count = 0

    def draw(pool: np.ndarray, n: int, fallback: np.ndarray) -> np.ndarray:
        nonlocal fallback_count
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)
        source = pool
        if source.size <= 0:
            source = fallback
            fallback_count += n
        replace = bool(source.size < n)
        if replace:
            fallback_count += max(0, n - int(source.size))
        return rng.choice(source, size=n, replace=replace).astype(np.int64)

    picked = [
        draw(rare_pool, n_rare, risk_pool if risk_pool.size > 0 else split_indices),
        draw(risk_pool, n_risk_extra, split_indices),
    ]
    used = int(sum(arr.shape[0] for arr in picked))
    picked.append(draw(split_indices, max(0, int(batch_size) - used), split_indices))
    batch_idx = np.concatenate(picked, axis=0)
    if batch_idx.shape[0] > int(batch_size):
        batch_idx = batch_idx[: int(batch_size)]
    rng.shuffle(batch_idx)
    return batch_idx.astype(np.int64), {
        "fallback_count": float(fallback_count),
        "risk_pool_size": float(risk_pool.size),
        "rare_pool_size": float(rare_pool.size),
    }


def parse_fse_source_ratio(ratio_text: str) -> Dict[int, float]:
    text = str(ratio_text or "").strip()
    if not text:
        text = "online_train=0.65,eval_policy=0.25,noisy_eval=0.10"
    out: Dict[int, float] = {}
    for raw in re.split(r"[,;\s]+", text):
        token = raw.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid --fse-source-ratio token '{token}'. Expected name=value.")
        name, value = token.split("=", 1)
        key = name.strip().lower()
        if key not in FSE_COLLECT_POLICY_TO_ID:
            raise ValueError(f"Unknown FSE collect source '{key}' in --fse-source-ratio.")
        out[int(FSE_COLLECT_POLICY_TO_ID[key])] = max(0.0, float(value))
    total = float(sum(out.values()))
    if total <= 0.0:
        raise ValueError("--fse-source-ratio must contain positive mass.")
    return {int(k): float(v) / total for k, v in out.items()}


def _draw_from_fse_pool(
    pool: np.ndarray,
    n: int,
    rng: np.random.Generator,
    fallback_pool: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64), 0
    source = np.asarray(pool, dtype=np.int64)
    fallback_count = 0
    if source.size <= 0:
        source = np.asarray(fallback_pool if fallback_pool is not None else [], dtype=np.int64)
        fallback_count += n
    if source.size <= 0:
        return np.zeros((0,), dtype=np.int64), n
    replace = bool(source.size < n)
    if replace:
        fallback_count += max(0, n - int(source.size))
    return rng.choice(source, size=int(n), replace=replace).astype(np.int64), fallback_count


def sample_fse_source_priority_batch(
    data: Dict[str, np.ndarray],
    split_indices: np.ndarray,
    batch_size: int,
    fse_cfg: FSEConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if "collect_policy_id" not in data:
        idx, stats = sample_fse_priority_batch(data, split_indices, batch_size, fse_cfg, rng)
        stats["source_balanced_sampler_active"] = 0.0
        return idx, stats
    split_indices = np.asarray(split_indices, dtype=np.int64)
    ratio = parse_fse_source_ratio(str(getattr(fse_cfg, "source_ratio", "")))
    if COLLECT_RANDOM in ratio:
        ratio[COLLECT_RANDOM] = min(float(ratio[COLLECT_RANDOM]), 0.05)
        total = sum(ratio.values())
        ratio = {k: v / max(total, 1e-12) for k, v in ratio.items()}
    ordered_sources = list(ratio.keys())
    desired = {src: int(math.floor(float(batch_size) * ratio[src])) for src in ordered_sources}
    remaining = int(batch_size) - int(sum(desired.values()))
    for src in ordered_sources[:remaining]:
        desired[src] = desired.get(src, 0) + 1

    any_risk, rare_risk = _fse_risk_masks(data)
    collect_ids = data["collect_policy_id"].astype(np.int64)
    picked: List[np.ndarray] = []
    source_fallback_count = 0
    risk_fallback_count = 0
    global_pool = split_indices
    batch_source_counts: Dict[int, int] = {}

    for src, n_src in desired.items():
        if n_src <= 0:
            continue
        src_pool = split_indices[collect_ids[split_indices] == int(src)]
        if src_pool.size <= 0:
            draw, fb = _draw_from_fse_pool(global_pool, n_src, rng, global_pool)
            source_fallback_count += fb + n_src
            picked.append(draw)
            batch_source_counts[int(src)] = batch_source_counts.get(int(src), 0) + int(draw.shape[0])
            continue
        risk_pool = src_pool[any_risk[src_pool]]
        rare_pool = src_pool[rare_risk[src_pool]]
        n_rare = int(math.ceil(n_src * float(fse_cfg.rare_window_ratio)))
        n_risk_total = int(math.ceil(n_src * float(fse_cfg.risk_window_ratio)))
        n_risk_extra = max(0, n_risk_total - n_rare)
        rare_draw, fb = _draw_from_fse_pool(rare_pool, n_rare, rng, risk_pool if risk_pool.size > 0 else src_pool)
        risk_fallback_count += fb
        risk_draw, fb = _draw_from_fse_pool(risk_pool, n_risk_extra, rng, src_pool)
        risk_fallback_count += fb
        used = int(rare_draw.shape[0] + risk_draw.shape[0])
        natural_draw, fb = _draw_from_fse_pool(src_pool, max(0, n_src - used), rng, global_pool)
        source_fallback_count += fb if src_pool.size <= 0 else 0
        picked.extend([rare_draw, risk_draw, natural_draw])
        batch_source_counts[int(src)] = batch_source_counts.get(int(src), 0) + int(rare_draw.shape[0] + risk_draw.shape[0] + natural_draw.shape[0])

    if picked:
        batch_idx = np.concatenate(picked, axis=0)
    else:
        batch_idx, source_fallback_count = _draw_from_fse_pool(global_pool, int(batch_size), rng, global_pool)
    if batch_idx.shape[0] < int(batch_size):
        extra, fb = _draw_from_fse_pool(global_pool, int(batch_size) - int(batch_idx.shape[0]), rng, global_pool)
        source_fallback_count += fb
        batch_idx = np.concatenate([batch_idx, extra], axis=0)
    if batch_idx.shape[0] > int(batch_size):
        batch_idx = batch_idx[: int(batch_size)]
    rng.shuffle(batch_idx)
    actual_counts = {
        f"batch_source_ratio_{FSE_COLLECT_ID_TO_POLICY.get(int(src), str(src))}": float(np.mean(collect_ids[batch_idx] == int(src)))
        for src in sorted(set(collect_ids[split_indices].tolist()) | set(ordered_sources))
    }
    stats: Dict[str, float] = {
        "source_balanced_sampler_active": 1.0,
        "source_fallback_count": float(source_fallback_count),
        "risk_fallback_count": float(risk_fallback_count),
        "fallback_count": float(source_fallback_count + risk_fallback_count),
    }
    stats.update(actual_counts)
    return batch_idx.astype(np.int64), stats


def make_fse_batch(
    data: Dict[str, np.ndarray],
    idx: np.ndarray,
    device: torch.device,
    action_conditioned: bool = False,
) -> Dict[str, torch.Tensor]:
    batch = {
        "tokens": torch.as_tensor(data["tokens"][idx], dtype=torch.float32, device=device),
        "token_mask": torch.as_tensor(data["token_mask"][idx], dtype=torch.float32, device=device),
        "entity_valid_mask": torch.as_tensor(data["entity_valid_mask"][idx], dtype=torch.float32, device=device),
        "token_type_ids": torch.as_tensor(data["token_type_ids"][idx], dtype=torch.long, device=device),
        "token_role_ids": torch.as_tensor(data["token_role_ids"][idx], dtype=torch.long, device=device),
        "labels_binary": torch.as_tensor(data["labels_binary"][idx], dtype=torch.float32, device=device),
        "labels_exposure": torch.as_tensor(data["labels_exposure"][idx], dtype=torch.float32, device=device),
        "labels_reg": torch.as_tensor(data["labels_reg"][idx], dtype=torch.float32, device=device),
        "valid_mask": torch.as_tensor(data["valid_mask"][idx], dtype=torch.float32, device=device),
    }
    if action_conditioned:
        action_key = "actions_for_fse" if "actions_for_fse" in data else "actions_policy_norm" if "actions_policy_norm" in data else "actions"
        if action_key not in data:
            raise ValueError("Action-conditioned FSE requires dataset key 'actions_policy_norm' or 'actions'.")
        batch["action"] = torch.as_tensor(data[action_key][idx], dtype=torch.float32, device=device)
    return batch


def _fse_forward_from_batch(model: FSEBottleneckTransformer, batch: Dict[str, torch.Tensor], action_conditioned: bool) -> FSEBottleneckOutput:
    return model(
        batch["tokens"],
        batch["token_mask"],
        batch["entity_valid_mask"],
        batch["token_type_ids"],
        batch["token_role_ids"],
        action=batch.get("action") if action_conditioned else None,
    )


def _finite_or_nan(value: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _nanmean_or_zero(values: List[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size <= 0:
        return 0.0
    return float(np.mean(arr))


def _nanmean_or_nan(values: List[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size <= 0:
        return float("nan")
    return float(np.mean(arr))


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(scores, dtype=np.float64)
    sorted_scores = scores[order]
    n = int(scores.shape[0])
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def binary_roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.int32)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos <= 0 or neg <= 0:
        return float("nan")
    ranks = _average_ranks(p.astype(np.float64))
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / max(pos * neg, 1))


def binary_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.int32)
    pos = int(np.sum(y == 1))
    if pos <= 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(pos), 1.0)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def binary_ece(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    if y.size <= 0:
        return float("nan")
    bins = max(1, int(bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(y.shape[0])
    ece = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(ece / max(total / y.shape[0], 1e-12))


def recall_at_fixed_fpr(y: np.ndarray, p: np.ndarray, fixed_fpr: float) -> float:
    y = y.astype(np.int32)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos <= 0 or neg <= 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    fpr = fp / max(float(neg), 1.0)
    tpr = tp / max(float(pos), 1.0)
    valid = fpr <= float(fixed_fpr)
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def precision_at_fixed_recall(y: np.ndarray, p: np.ndarray, fixed_recall: float) -> float:
    y = y.astype(np.int32)
    pos = int(np.sum(y == 1))
    if pos <= 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(pos), 1.0)
    valid = recall >= float(fixed_recall)
    if not np.any(valid):
        return 0.0
    return float(np.max(precision[valid]))


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float, fse_cfg: FSEConfig) -> Dict[str, float]:
    y = y.astype(np.float32).reshape(-1)
    p = np.clip(p.astype(np.float32).reshape(-1), 0.0, 1.0)
    support = int(y.shape[0])
    if support <= 0:
        return {
            "support": 0.0,
            "positive_support": 0.0,
            "positive_rate": float("nan"),
            "predicted_positive_rate": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "F1": float("nan"),
            "ROC_AUC": float("nan"),
            "PR_AUC": float("nan"),
            "Brier": float("nan"),
            "ECE": float("nan"),
            "recall_at_fixed_fpr": float("nan"),
            "precision_at_fixed_recall": float("nan"),
        }
    pred = p >= float(threshold)
    y_bool = y >= 0.5
    tp = float(np.sum(pred & y_bool))
    fp = float(np.sum(pred & ~y_bool))
    fn = float(np.sum(~pred & y_bool))
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "support": float(support),
        "positive_support": float(np.sum(y_bool)),
        "positive_rate": float(np.mean(y_bool)),
        "predicted_positive_rate": float(np.mean(pred)),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "ROC_AUC": binary_roc_auc(y_bool.astype(np.int32), p),
        "PR_AUC": binary_pr_auc(y_bool.astype(np.int32), p),
        "Brier": float(np.mean(np.square(p - y))),
        "ECE": binary_ece(y, p, bins=int(fse_cfg.ece_bins)),
        "recall_at_fixed_fpr": recall_at_fixed_fpr(y_bool.astype(np.int32), p, fixed_fpr=float(fse_cfg.fixed_fpr)),
        "precision_at_fixed_recall": precision_at_fixed_recall(y_bool.astype(np.int32), p, fixed_recall=float(fse_cfg.fixed_recall)),
    }


def predict_fse(
    model: FSEBottleneckTransformer,
    data: Dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    fse_cfg: FSEConfig,
    action_conditioned: bool,
) -> Dict[str, np.ndarray]:
    model.eval()
    risk_probs: List[np.ndarray] = []
    z_values: List[np.ndarray] = []
    losses: List[float] = []
    batch_size = max(1, int(fse_cfg.batch_size))
    with torch.no_grad():
        for start in range(0, int(indices.shape[0]), batch_size):
            idx = indices[start: start + batch_size]
            batch = make_fse_batch(data, idx, device=device, action_conditioned=action_conditioned)
            out = _fse_forward_from_batch(model, batch, action_conditioned=action_conditioned)
            loss = compute_fse_loss(out, batch["labels_binary"], batch["labels_exposure"], batch["labels_reg"], batch["valid_mask"], fse_cfg)
            risk_probs.append(out.risk_probs.detach().cpu().numpy())
            z_values.append(out.z_fse.detach().cpu().numpy())
            losses.append(float(loss.total.item()))
    if risk_probs:
        risk_arr = np.concatenate(risk_probs, axis=0)
        z_arr = np.concatenate(z_values, axis=0)
    else:
        risk_arr = np.zeros((0, len(fse_cfg.horizons), len(FSE_RISK_NAMES)), dtype=np.float32)
        z_arr = np.zeros((0, int(fse_cfg.z_dim)), dtype=np.float32)
    return {
        "risk_probs": risk_arr,
        "z_fse": z_arr,
        "loss": float(np.mean(losses)) if losses else float("nan"),
    }


def evaluate_fse_predictions(
    data: Dict[str, np.ndarray],
    indices: np.ndarray,
    pred: Dict[str, np.ndarray],
    fse_cfg: FSEConfig,
) -> Dict[str, Any]:
    labels = data["labels_binary"][indices]
    valid = data["valid_mask"][indices] > 0.5
    probs = pred["risk_probs"]
    threshold = float(fse_cfg.threshold)
    per_horizon: Dict[str, Any] = {}
    all_ece: List[float] = []
    all_brier: List[float] = []
    for h_i, horizon in enumerate(tuple(fse_cfg.horizons)):
        h_key = f"H{int(horizon)}"
        per_horizon[h_key] = {}
        for r_i, risk_name in enumerate(FSE_RISK_NAMES):
            mask = valid[:, h_i]
            metrics = binary_metrics(labels[mask, h_i, r_i], probs[mask, h_i, r_i], threshold=threshold, fse_cfg=fse_cfg)
            per_horizon[h_key][risk_name] = metrics
            all_ece.append(metrics["ECE"])
            all_brier.append(metrics["Brier"])

    support_summary: Dict[str, float] = {}
    for risk_name in ("collision", "offroad", "low_ttc", "unsafe_headway"):
        r_i = FSE_RISK_NAMES.index(risk_name)
        valid_labels = labels[:, :, r_i][valid]
        support_summary[f"support_{risk_name}"] = float(np.sum(valid_labels >= 0.5))
        support_summary[f"valid_count_{risk_name}"] = float(valid_labels.shape[0])
    support_summary["support_headway"] = float(support_summary.get("support_unsafe_headway", 0.0))
    support_summary["valid_count_headway"] = float(support_summary.get("valid_count_unsafe_headway", 0.0))

    violations = []
    for h_i in range(max(0, len(tuple(fse_cfg.horizons)) - 1)):
        pair_valid = valid[:, h_i] & valid[:, h_i + 1]
        if np.any(pair_valid):
            violations.append((probs[pair_valid, h_i, :] > probs[pair_valid, h_i + 1, :]).astype(np.float32).reshape(-1))
    if violations:
        monotonic_violation_rate = float(np.mean(np.concatenate(violations, axis=0)))
    else:
        monotonic_violation_rate = float("nan")
    z = pred["z_fse"]
    z_norm = float(np.linalg.norm(z, axis=1).mean()) if z.shape[0] > 0 else float("nan")
    z_variance = float(np.var(z, axis=0).mean()) if z.shape[0] > 1 else 0.0
    return {
        "loss": float(pred.get("loss", float("nan"))),
        "per_horizon": per_horizon,
        "support_summary": support_summary,
        **support_summary,
        "mean_ECE": _nanmean_or_nan([_finite_or_nan(v) for v in all_ece]),
        "mean_Brier": _nanmean_or_nan([_finite_or_nan(v) for v in all_brier]),
        "monotonic_violation_rate": monotonic_violation_rate,
        "z_norm": z_norm,
        "z_variance": z_variance,
    }


def fse_threshold_map(fse_cfg: FSEConfig) -> Dict[str, Dict[str, float]]:
    return {
        f"H{int(h)}": {risk_name: float(fse_cfg.threshold) for risk_name in FSE_RISK_NAMES}
        for h in tuple(fse_cfg.horizons)
    }


def fse_best_score(metrics: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    per_horizon = metrics.get("per_horizon", {})
    f1_low_headway: List[float] = []
    pr_collision_offroad: List[float] = []
    recall_collision_offroad: List[float] = []
    for h_payload in per_horizon.values():
        for risk_name in ("low_ttc", "unsafe_headway"):
            f1_low_headway.append(_finite_or_nan(h_payload.get(risk_name, {}).get("F1", float("nan"))))
        for risk_name in ("collision", "offroad"):
            pr_collision_offroad.append(_finite_or_nan(h_payload.get(risk_name, {}).get("PR_AUC", float("nan"))))
            recall_collision_offroad.append(_finite_or_nan(h_payload.get(risk_name, {}).get("recall", float("nan"))))
    ece_raw = _finite_or_nan(metrics.get("mean_ECE", 1.0))
    mono_raw = _finite_or_nan(metrics.get("monotonic_violation_rate", 1.0))
    if not np.isfinite(ece_raw):
        ece_raw = 1.0
    if not np.isfinite(mono_raw):
        mono_raw = 1.0
    ece_component = 1.0 - float(np.clip(ece_raw, 0.0, 1.0))
    mono_component = 1.0 - float(np.clip(mono_raw, 0.0, 1.0))
    components = {
        "macro_F1_low_ttc_headway": _nanmean_or_zero(f1_low_headway),
        "mean_PR_AUC_collision_offroad": _nanmean_or_zero(pr_collision_offroad),
        "recall_collision_offroad": _nanmean_or_zero(recall_collision_offroad),
        "one_minus_ECE": ece_component,
        "one_minus_monotonic_violation_rate": mono_component,
    }
    score = (
        0.35 * components["macro_F1_low_ttc_headway"]
        + 0.30 * components["mean_PR_AUC_collision_offroad"]
        + 0.15 * components["recall_collision_offroad"]
        + 0.10 * components["one_minus_ECE"]
        + 0.10 * components["one_minus_monotonic_violation_rate"]
    )
    return float(score), components


def fse_make_checkpoint(
    model: FSEBottleneckTransformer,
    fse_cfg: FSEConfig,
    action_conditioned: bool,
    action_dim: int,
    best_score: float,
    best_metric: Dict[str, Any],
    support_summary: Dict[str, Any],
) -> Dict[str, Any]:
    normalization_config = {
        "scenario_schema": SCENARIO_SCHEMA_VERSION,
        "token_count": int(SCENARIO_TOKEN_COUNT),
        "token_dim": int(SCENARIO_TOKEN_DIM),
        "token_order": list(SCENARIO_TOKEN_ORDER),
        "token_type_ids": SCENARIO_TOKEN_TYPE_IDS.astype(int).tolist(),
        "token_role_ids": SCENARIO_TOKEN_ROLE_IDS.astype(int).tolist(),
        "scalar_context_model_dim": int(SCENARIO_SCALAR_CONTEXT_MODEL_DIM),
        "normalization_version": "scenario_frame_v1_12x24",
    }
    return {
        "model_state_dict": model.state_dict(),
        "fse_config": asdict(fse_cfg),
        "scenario_frame_schema_version": SCENARIO_SCHEMA_VERSION,
        "risk_names": list(FSE_RISK_NAMES),
        "exposure_names": list(FSE_EXPOSURE_NAMES),
        "regression_names": list(FSE_REGRESSION_NAMES),
        "horizons": [int(v) for v in tuple(fse_cfg.horizons)],
        "z_dim": int(fse_cfg.z_dim),
        "token_dim": int(SCENARIO_TOKEN_DIM),
        "n_tokens": int(SCENARIO_TOKEN_COUNT),
        "action_conditioned": bool(action_conditioned),
        "action_dim": int(action_dim if action_conditioned else 0),
        "action_metadata": {
            "action_space": FSE_ACTION_SPACE_POLICY_NORMALIZED if action_conditioned else "none",
            "action_dim": int(action_dim if action_conditioned else 0),
            "action_low": [-1.0, -1.0] if action_conditioned else [],
            "action_high": [1.0, 1.0] if action_conditioned else [],
            "action_order": list(FSE_ACTION_ORDER) if action_conditioned else [],
        },
        "normalization_spec": {
            "tokens": "ScenarioFrame v1 normalized 12x24 tokens",
            "labels_binary": "binary in {0,1}",
            "labels_exposure": "normalized [0,1]",
            "labels_reg": "normalized [0,1]",
            "valid_mask": "1 means horizon contributes to loss/metrics",
        },
        "normalization_config": normalization_config,
        "label_metadata": {
            "horizons": [int(v) for v in tuple(fse_cfg.horizons)],
            "env_dt": float(1.0 / 20.0),
            "horizon_seconds": [float(int(v) / 20.0) for v in tuple(fse_cfg.horizons)],
            "unsafe_headway_tau": float(FSE_UNSAFE_HEADWAY_TAU),
            "low_ttc_tau": float(FSE_LOW_TTC_TAU),
            "lane_boundary_margin_threshold": float(FSE_LANE_BOUNDARY_MARGIN_THRESHOLD),
            "lane_offset_threshold": f"{FSE_LANE_OFFSET_RATIO_THRESHOLD} * lane_width",
            "cost_safety_definition": FSE_COST_SAFETY_DEFINITION,
            "safety_cost_sum_name": "safety_exposure_cost_sum",
            "cost_lane_definition": FSE_COST_LANE_DEFINITION,
        },
        "best_metric": best_metric,
        "best_score": float(best_score),
        "threshold_by_class_by_horizon": fse_threshold_map(fse_cfg),
        "support_summary": _jsonable_plain(support_summary),
    }


def _fse_config_from_payload(payload: Dict[str, Any]) -> FSEConfig:
    cfg = FSEConfig()
    fields = getattr(FSEConfig, "__dataclass_fields__", {})
    for key, value in (payload or {}).items():
        if key in fields:
            setattr(cfg, key, value)
    cfg.horizons = tuple(int(v) for v in cfg.horizons)
    cfg.focal_alpha = tuple(float(v) for v in cfg.focal_alpha)
    return cfg


def fse_structured_normalization_payload(checkpoint: Dict[str, Any]) -> Any:
    payload = checkpoint.get("normalization_config", None)
    if payload is None:
        payload = checkpoint.get("normalization_spec", None)
    return payload


def build_fse_model(fse_cfg: FSEConfig, action_dim: int = 0, device: Optional[torch.device] = None) -> FSEBottleneckTransformer:
    model = FSEBottleneckTransformer(fse_cfg, action_dim=int(action_dim))
    if device is not None:
        model = model.to(device)
    return model


def save_fse_z_stats(split_metrics: Dict[str, Dict[str, Any]], output_dir: str) -> Dict[str, float]:
    stats = {
        "z_norm_train": float(split_metrics.get("train", {}).get("z_norm", float("nan"))),
        "z_norm_val": float(split_metrics.get("val", {}).get("z_norm", float("nan"))),
        "z_norm_test": float(split_metrics.get("test", {}).get("z_norm", float("nan"))),
        "z_variance_train": float(split_metrics.get("train", {}).get("z_variance", float("nan"))),
        "z_variance_val": float(split_metrics.get("val", {}).get("z_variance", float("nan"))),
        "z_variance_test": float(split_metrics.get("test", {}).get("z_variance", float("nan"))),
    }
    export_csv([stats], os.path.join(output_dir, "fse_z_stats.csv"), list(stats.keys()))
    return stats


def run_fse_smoke(cfg: Config) -> Dict[str, Any]:
    output_dir = str(cfg.fse.output_dir).strip() or "results/fse"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(cfg.train.device)
    fse_cfg = copy.deepcopy(cfg.fse)
    batch = 4
    tokens = torch.randn(batch, SCENARIO_TOKEN_COUNT, SCENARIO_TOKEN_DIM, device=device)
    token_mask = torch.ones(batch, SCENARIO_TOKEN_COUNT, device=device)
    entity_valid_mask = torch.ones(batch, SCENARIO_TOKEN_COUNT, device=device)
    entity_valid_mask[:, 1] = 0.0
    tokens[:, 1, 0] = 1.0
    token_type_ids = torch.as_tensor(SCENARIO_TOKEN_TYPE_IDS, device=device).unsqueeze(0).expand(batch, -1)
    token_role_ids = torch.as_tensor(SCENARIO_TOKEN_ROLE_IDS, device=device).unsqueeze(0).expand(batch, -1)
    labels_binary = torch.zeros(batch, len(fse_cfg.horizons), len(FSE_RISK_NAMES), device=device)
    labels_binary[:, :, FSE_RISK_NAMES.index("unsafe_headway")] = 1.0
    labels_exposure = torch.rand(batch, len(fse_cfg.horizons), len(FSE_EXPOSURE_NAMES), device=device)
    labels_reg = torch.rand(batch, len(fse_cfg.horizons), len(FSE_REGRESSION_NAMES), device=device)
    valid_mask = torch.ones(batch, len(fse_cfg.horizons), device=device)
    valid_mask[0, -1] = 0.0

    model = build_fse_model(fse_cfg, action_dim=0, device=device)
    missing_mask_error = False
    try:
        model(tokens, token_mask, None, token_type_ids, token_role_ids)
    except ValueError:
        missing_mask_error = True
    out = model(tokens, token_mask, entity_valid_mask, token_type_ids, token_role_ids)
    loss = compute_fse_loss(out, labels_binary, labels_exposure, labels_reg, valid_mask, fse_cfg)

    action_model = build_fse_model(fse_cfg, action_dim=2, device=device)
    for p in action_model.parameters():
        p.requires_grad_(False)
    action = torch.zeros(batch, 2, device=device, requires_grad=True)
    action_out = action_model(tokens.detach(), token_mask, entity_valid_mask, token_type_ids, token_role_ids, action=action)
    risk_scalar = action_out.risk_probs[..., FSE_RISK_NAMES.index("collision")].mean()
    risk_scalar.backward()
    risk_grad_norm = float(action.grad.detach().norm().item()) if action.grad is not None else 0.0

    result = {
        "status": "ok",
        "missing_entity_valid_mask_error": bool(missing_mask_error),
        "risk_shape": list(out.risk_probs.shape),
        "exposure_shape": list(out.exposure_pred.shape),
        "reg_shape": list(out.reg_pred.shape),
        "z_shape": list(out.z_fse.shape),
        "risk_min": float(out.risk_probs.min().item()),
        "risk_max": float(out.risk_probs.max().item()),
        "exposure_min": float(out.exposure_pred.min().item()),
        "exposure_max": float(out.exposure_pred.max().item()),
        "reg_min": float(out.reg_pred.min().item()),
        "reg_max": float(out.reg_pred.max().item()),
        "uncertainty_min": float(out.uncertainty.min().item()),
        "uncertainty_max": float(out.uncertainty.max().item()),
        "loss_total": float(loss.total.item()),
        "entity_mask_mismatch_count": int(model.entity_mask_mismatch_count),
        "risk_grad_norm": risk_grad_norm,
    }
    result["fse_smoke_result_json_path"] = export_json(result, os.path.join(output_dir, "fse_smoke_result.json"))
    return result


def evaluate_fse_model_on_splits(
    model: FSEBottleneckTransformer,
    data: Dict[str, np.ndarray],
    splits: Dict[str, np.ndarray],
    device: torch.device,
    fse_cfg: FSEConfig,
    action_conditioned: bool,
) -> Dict[str, Dict[str, Any]]:
    split_metrics: Dict[str, Dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        idx = splits.get(split_name, np.zeros((0,), dtype=np.int64))
        pred = predict_fse(model, data, idx, device=device, fse_cfg=fse_cfg, action_conditioned=action_conditioned)
        split_metrics[split_name] = evaluate_fse_predictions(data, idx, pred, fse_cfg=fse_cfg)
    return split_metrics


def evaluate_fse_model_grouped(
    model: FSEBottleneckTransformer,
    data: Dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    fse_cfg: FSEConfig,
    action_conditioned: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    idx_all = np.asarray(indices, dtype=np.int64)
    pred_all = predict_fse(model, data, idx_all, device=device, fse_cfg=fse_cfg, action_conditioned=action_conditioned)
    payload["metrics_all_mixed"] = evaluate_fse_predictions(data, idx_all, pred_all, fse_cfg=fse_cfg)
    by_policy: Dict[str, Any] = {}
    if "collect_policy_id" in data:
        collect = data["collect_policy_id"].astype(np.int64)
        for policy_id, policy_name in FSE_COLLECT_ID_TO_POLICY.items():
            sub_idx = idx_all[collect[idx_all] == int(policy_id)]
            if sub_idx.shape[0] <= 0:
                continue
            pred = predict_fse(model, data, sub_idx, device=device, fse_cfg=fse_cfg, action_conditioned=action_conditioned)
            by_policy[str(policy_name)] = evaluate_fse_predictions(data, sub_idx, pred, fse_cfg=fse_cfg)
        eval_idx = idx_all[collect[idx_all] == COLLECT_EVAL_POLICY]
        if eval_idx.shape[0] > 0:
            pred_eval = predict_fse(model, data, eval_idx, device=device, fse_cfg=fse_cfg, action_conditioned=action_conditioned)
            payload["metrics_eval_policy_only"] = evaluate_fse_predictions(data, eval_idx, pred_eval, fse_cfg=fse_cfg)
    payload["metrics_by_collect_policy"] = by_policy
    payload["eval_source_ratio_natural"] = fse_source_distribution_for_indices(data, idx_all)
    return payload


def fse_source_distribution_for_indices(data: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, Any]:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.shape[0] <= 0 or "collect_policy_id" not in data:
        return {}
    collect = data["collect_policy_id"].astype(np.int64)[idx]
    out: Dict[str, Any] = {"num_samples": int(idx.shape[0])}
    for policy_id, policy_name in FSE_COLLECT_ID_TO_POLICY.items():
        count = int(np.sum(collect == int(policy_id)))
        out[str(policy_name)] = {"sample_count": count, "sample_ratio": float(count / max(1, idx.shape[0]))}
    if "episode_uid_id" in data:
        ep = data["episode_uid_id"].astype(np.int64)[idx]
        out["num_episodes"] = int(np.unique(ep).shape[0])
    return out


def run_fse_train(cfg: Config) -> Dict[str, Any]:
    output_dir = str(cfg.fse.output_dir).strip() or "results/fse"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(cfg.train.device)
    fse_cfg = copy.deepcopy(cfg.fse)
    data, load_warnings = load_fse_npz_dataset(str(fse_cfg.dataset_path), fse_cfg)
    splits = split_fse_by_episode(data, seed=int(cfg.train.seed), fse_cfg=fse_cfg)
    action_conditioned = bool(fse_cfg.action_conditioned)
    action_dim = int(data["actions_for_fse"].shape[1]) if action_conditioned and "actions_for_fse" in data else 0
    if action_conditioned and action_dim <= 0:
        raise ValueError("--fse-action-conditioned requires dataset key 'actions_policy_norm' or 'actions' with shape [N, action_dim].")
    model = build_fse_model(fse_cfg, action_dim=action_dim, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(fse_cfg.lr), weight_decay=float(fse_cfg.weight_decay))
    rng = np.random.default_rng(int(cfg.train.seed))
    train_rows: List[Dict[str, float]] = []
    checkpoint_path = str(fse_cfg.checkpoint_path).strip() or os.path.join(output_dir, "fse_checkpoint.pt")
    best_score = -float("inf")
    best_tiebreak = (float("inf"), float("inf"), float("inf"))
    best_metric: Dict[str, Any] = {}
    train_idx = splits["train"]
    steps_per_epoch = max(1, int(math.ceil(train_idx.shape[0] / max(1, int(fse_cfg.batch_size)))))

    for epoch in range(1, max(1, int(fse_cfg.epochs)) + 1):
        model.train()
        total_losses: List[float] = []
        focal_losses: List[float] = []
        exposure_losses: List[float] = []
        reg_losses: List[float] = []
        mono_losses: List[float] = []
        fallback_count = 0.0
        source_fallback_count = 0.0
        risk_fallback_count = 0.0
        source_ratio_accum: Dict[str, List[float]] = defaultdict(list)
        for _ in range(steps_per_epoch):
            if bool(getattr(fse_cfg, "source_balanced_sampler", False)):
                idx, sample_stats = sample_fse_source_priority_batch(data, train_idx, int(fse_cfg.batch_size), fse_cfg, rng)
            else:
                idx, sample_stats = sample_fse_priority_batch(data, train_idx, int(fse_cfg.batch_size), fse_cfg, rng)
            fallback_count += float(sample_stats.get("fallback_count", 0.0))
            source_fallback_count += float(sample_stats.get("source_fallback_count", 0.0))
            risk_fallback_count += float(sample_stats.get("risk_fallback_count", 0.0))
            for key, value in sample_stats.items():
                if str(key).startswith("batch_source_ratio_"):
                    source_ratio_accum[str(key)].append(float(value))
            batch = make_fse_batch(data, idx, device=device, action_conditioned=action_conditioned)
            out = _fse_forward_from_batch(model, batch, action_conditioned=action_conditioned)
            loss = compute_fse_loss(out, batch["labels_binary"], batch["labels_exposure"], batch["labels_reg"], batch["valid_mask"], fse_cfg)
            optimizer.zero_grad()
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_losses.append(float(loss.total.item()))
            focal_losses.append(float(loss.focal_binary.item()))
            exposure_losses.append(float(loss.exposure.item()))
            reg_losses.append(float(loss.regression.item()))
            mono_losses.append(float(loss.monotonic.item()))

        val_pred = predict_fse(model, data, splits["val"], device=device, fse_cfg=fse_cfg, action_conditioned=action_conditioned)
        val_metrics = evaluate_fse_predictions(data, splits["val"], val_pred, fse_cfg=fse_cfg)
        val_score, val_components = fse_best_score(val_metrics)
        def tie_value(raw_value: Any) -> float:
            numeric = _finite_or_nan(raw_value)
            return float(numeric) if np.isfinite(numeric) else float("inf")
        tiebreak = (
            tie_value(val_metrics.get("mean_ECE", float("inf"))),
            tie_value(val_metrics.get("mean_Brier", float("inf"))),
            tie_value(val_metrics.get("loss", float("inf"))),
        )
        improved = bool(val_score > best_score or (abs(val_score - best_score) <= 1e-12 and tiebreak < best_tiebreak))
        if improved:
            best_score = float(val_score)
            best_tiebreak = tiebreak
            best_metric = {
                "split": "val",
                "score_components": val_components,
                "mean_ECE": val_metrics.get("mean_ECE", float("nan")),
                "mean_Brier": val_metrics.get("mean_Brier", float("nan")),
                "loss": val_metrics.get("loss", float("nan")),
                "monotonic_violation_rate": val_metrics.get("monotonic_violation_rate", float("nan")),
            }
            checkpoint = fse_make_checkpoint(
                model,
                fse_cfg,
                action_conditioned=action_conditioned,
                action_dim=action_dim,
                best_score=best_score,
                best_metric=best_metric,
                support_summary=val_metrics.get("support_summary", {}),
            )
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
        train_rows.append(
            {
                "epoch": float(epoch),
                "loss_total": float(np.mean(total_losses)),
                "loss_focal_binary": float(np.mean(focal_losses)),
                "loss_exposure": float(np.mean(exposure_losses)),
                "loss_regression": float(np.mean(reg_losses)),
                "loss_monotonic": float(np.mean(mono_losses)),
                "val_loss": float(val_metrics.get("loss", float("nan"))),
                "val_best_score": float(val_score),
                "best_score_so_far": float(best_score),
                "fallback_count": float(fallback_count),
                "source_fallback_count": float(source_fallback_count),
                "risk_fallback_count": float(risk_fallback_count),
                "entity_mask_mismatch_count": float(model.entity_mask_mismatch_count + int(load_warnings.get("entity_valid_mismatch_count", 0))),
                "support_collision": float(val_metrics.get("support_collision", 0.0)),
                "support_offroad": float(val_metrics.get("support_offroad", 0.0)),
                "support_low_ttc": float(val_metrics.get("support_low_ttc", 0.0)),
                "support_headway": float(val_metrics.get("support_unsafe_headway", 0.0)),
                **{key: float(np.mean(values)) for key, values in source_ratio_accum.items()},
            }
        )
        print(
            f"[FSE][epoch={epoch}] loss={np.mean(total_losses):.4f} val_score={val_score:.4f} "
            f"val_loss={float(val_metrics.get('loss', float('nan'))):.4f} fallback={fallback_count:.0f}"
        )

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    split_metrics = evaluate_fse_model_on_splits(model, data, splits, device=device, fse_cfg=fse_cfg, action_conditioned=action_conditioned)
    grouped_metrics = evaluate_fse_model_grouped(model, data, splits["test"], device=device, fse_cfg=fse_cfg, action_conditioned=action_conditioned)
    z_stats = save_fse_z_stats(split_metrics, output_dir)
    train_log_path = export_csv(train_rows, os.path.join(output_dir, "fse_train_log.csv"), list(train_rows[0].keys()) if train_rows else [])
    eval_payload = {
        "task": "train",
        "dataset_path": os.path.abspath(str(fse_cfg.dataset_path)),
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "warnings": load_warnings,
        "split_sizes": {name: int(idx.shape[0]) for name, idx in splits.items()},
        "split_episode_counts": {name: int(np.unique(data[("episode_uid_id" if "episode_uid_id" in data else "episode_id")][idx]).shape[0]) for name, idx in splits.items()},
        "threshold_by_class_by_horizon": fse_threshold_map(fse_cfg),
        "natural_distribution_metrics": split_metrics,
        "grouped_test_metrics": grouped_metrics,
        **grouped_metrics,
        "train_source_ratio_actual": fse_source_distribution_for_indices(data, splits["train"]),
        "train_batch_source_ratio_mean": {
            key: float(np.mean([float(row.get(key, float("nan"))) for row in train_rows if key in row]))
            for key in sorted({k for row in train_rows for k in row.keys() if k.startswith("batch_source_ratio_")})
        },
        "eval_source_ratio_natural": fse_source_distribution_for_indices(data, splits["test"]),
        "best_score": float(best_score),
        "best_metric": best_metric,
        "z_stats": z_stats,
    }
    eval_metrics_path = export_json(eval_payload, os.path.join(output_dir, "fse_eval_metrics.json"))
    return {
        "status": "ok",
        "fse_train_log_csv_path": train_log_path,
        "fse_eval_metrics_json_path": eval_metrics_path,
        "fse_checkpoint_path": os.path.abspath(checkpoint_path),
        "best_score": float(best_score),
        "z_stats": z_stats,
    }


def run_fse_eval(cfg: Config) -> Dict[str, Any]:
    output_dir = str(cfg.fse.output_dir).strip() or "results/fse"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(cfg.train.device)
    if not str(cfg.fse.checkpoint_path).strip():
        raise ValueError("--fse-checkpoint-path is required for --fse-task eval.")
    try:
        checkpoint = torch.load(str(cfg.fse.checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(cfg.fse.checkpoint_path), map_location=device)
    fse_cfg = _fse_config_from_payload(checkpoint.get("fse_config", asdict(cfg.fse)))
    fse_cfg.dataset_path = str(cfg.fse.dataset_path)
    fse_cfg.output_dir = output_dir
    data, load_warnings = load_fse_npz_dataset(str(fse_cfg.dataset_path), fse_cfg)
    action_conditioned = bool(checkpoint.get("action_conditioned", False))
    action_dim = int(data["actions_for_fse"].shape[1]) if action_conditioned and "actions_for_fse" in data else 0
    if action_conditioned and action_dim <= 0:
        raise ValueError("Action-conditioned FSE eval requires dataset key 'actions_policy_norm' or 'actions' with shape [N, action_dim].")
    model = build_fse_model(fse_cfg, action_dim=action_dim, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    splits = split_fse_by_episode(data, seed=int(cfg.train.seed), fse_cfg=fse_cfg)
    split_metrics = evaluate_fse_model_on_splits(model, data, splits, device=device, fse_cfg=fse_cfg, action_conditioned=action_conditioned)
    grouped_metrics = evaluate_fse_model_grouped(model, data, splits["test"], device=device, fse_cfg=fse_cfg, action_conditioned=action_conditioned)
    z_stats = save_fse_z_stats(split_metrics, output_dir)
    eval_payload = {
        "task": "eval",
        "dataset_path": os.path.abspath(str(fse_cfg.dataset_path)),
        "checkpoint_path": os.path.abspath(str(cfg.fse.checkpoint_path)),
        "warnings": load_warnings,
        "split_sizes": {name: int(idx.shape[0]) for name, idx in splits.items()},
        "threshold_by_class_by_horizon": checkpoint.get("threshold_by_class_by_horizon", fse_threshold_map(fse_cfg)),
        "natural_distribution_metrics": split_metrics,
        "grouped_test_metrics": grouped_metrics,
        **grouped_metrics,
        "eval_source_ratio_natural": fse_source_distribution_for_indices(data, splits["test"]),
        "checkpoint_metadata": {
            "scenario_frame_schema_version": checkpoint.get("scenario_frame_schema_version", ""),
            "risk_names": checkpoint.get("risk_names", []),
            "horizons": checkpoint.get("horizons", []),
            "action_conditioned": bool(action_conditioned),
            "action_dim": int(checkpoint.get("action_dim", action_dim)),
            "action_metadata": _jsonable_plain(checkpoint.get("action_metadata", {})),
            "best_score": checkpoint.get("best_score", float("nan")),
            "support_summary": checkpoint.get("support_summary", {}),
        },
        "z_stats": z_stats,
    }
    eval_metrics_path = export_json(eval_payload, os.path.join(output_dir, "fse_eval_metrics.json"))
    return {
        "status": "ok",
        "fse_eval_metrics_json_path": eval_metrics_path,
        "z_stats": z_stats,
    }


@dataclass
class FSEStepZ:
    z_real: np.ndarray
    z_real_valid: bool
    z_used: np.ndarray
    z_used_valid: bool
    z_source: str
    fse_forward_executed: bool
    uncertainty_mean: float = float("nan")
    risk_mean: float = float("nan")
    shuffled_source_episode_id: int = -1
    shuffled_source_step_id: int = -1
    shuffled_fallback_flag: bool = False
    shuffled_self_sample_flag: bool = False
    shuffled_nearby_episode_sample_flag: bool = False
    random_std_clipped_dims: int = 0


@dataclass
class ActionRiskPenaltyOutput:
    risk_raw: torch.Tensor
    risk_norm: torch.Tensor
    uncertainty_scalar: torch.Tensor
    uncertainty_gate: torch.Tensor
    risk_penalty: torch.Tensor
    risk_probs: torch.Tensor
    uncertainty: torch.Tensor


class RunningMean:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.skipped_nonfinite = 0

    def update(self, value: float) -> None:
        try:
            v = float(value)
        except Exception:
            return
        if not np.isfinite(v):
            self.skipped_nonfinite += 1
            return
        self.count += 1
        self.total += v

    def mean(self, empty: float = float("nan")) -> float:
        return float(self.total / float(self.count)) if self.count > 0 else float(empty)


class RunningVectorVariance:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.mean = np.zeros((self.dim,), dtype=np.float64)
        self.m2 = np.zeros((self.dim,), dtype=np.float64)
        self.skipped_nonfinite = 0

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"RunningVectorVariance expected dim={self.dim}, got {arr.shape[0]}.")
        if not np.all(np.isfinite(arr)):
            self.skipped_nonfinite += 1
            return
        self.count += 1
        delta = arr - self.mean
        self.mean += delta / float(self.count)
        delta2 = arr - self.mean
        self.m2 += delta * delta2

    def batch_variance_mean(self, empty: float = float("nan")) -> float:
        if self.count <= 0:
            return float(empty)
        return float(np.mean(self.m2 / float(self.count)))


class FSEZDiagnostics:
    def __init__(self, z_dim: int, trace_stride: int = 1, trace_max_rows: int = 200000):
        self.z_dim = int(z_dim)
        self.trace_stride = max(1, int(trace_stride))
        self.trace_max_rows = max(0, int(trace_max_rows))
        self.trace_rows: List[Dict[str, Any]] = []
        self._step_counter = 0
        self.z_used_dim_var = RunningMean()
        self.z_real_dim_var = RunningMean()
        self.z_used_batch_var = RunningVectorVariance(self.z_dim)
        self.z_real_batch_var = RunningVectorVariance(self.z_dim)
        self.z_used_norm_mean = RunningMean()
        self.z_real_norm_mean = RunningMean()
        self.z_used_abs_mean = RunningMean()
        self.z_real_abs_mean = RunningMean()
        self.z_used_temporal_delta_mean = RunningMean()
        self.z_real_temporal_delta_mean = RunningMean()
        self.z_norm_ratios: List[float] = []
        self.reset_episode()
        self.total_records = 0
        self.shuffle_cross_episode = 0
        self.shuffle_same_episode_far = 0
        self.shuffle_random_fallback = 0
        self.shuffle_self_sample = 0
        self.shuffle_nearby = 0
        self.state_aug_nan_count = 0
        self.state_aug_inf_count = 0
        self.next_state_aug_nan_count = 0
        self.next_state_aug_inf_count = 0

    def reset_episode(self) -> None:
        self._prev_z_real: Optional[np.ndarray] = None
        self._prev_z_used: Optional[np.ndarray] = None

    @staticmethod
    def _finite_vec(z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float32).reshape(-1)

    def record(
        self,
        episode_id: int,
        step_id: int,
        raw_state: np.ndarray,
        current: FSEStepZ,
        next_info: Optional[FSEStepZ],
        state_aug: np.ndarray,
        next_state_aug: Optional[np.ndarray],
        mode: str,
    ) -> None:
        self.total_records += 1
        z_used = self._finite_vec(current.z_used)
        self.z_used_batch_var.update(z_used)
        self.z_used_dim_var.update(float(np.var(z_used)))
        self.z_used_norm_mean.update(float(np.linalg.norm(z_used)))
        self.z_used_abs_mean.update(float(np.mean(np.abs(z_used))))
        if self._prev_z_used is not None:
            self.z_used_temporal_delta_mean.update(float(np.linalg.norm(z_used - self._prev_z_used)))
        self._prev_z_used = z_used.copy()
        if bool(current.z_real_valid):
            z_real = self._finite_vec(current.z_real)
            self.z_real_batch_var.update(z_real)
            self.z_real_dim_var.update(float(np.var(z_real)))
            self.z_real_norm_mean.update(float(np.linalg.norm(z_real)))
            self.z_real_abs_mean.update(float(np.mean(np.abs(z_real))))
            if self._prev_z_real is not None:
                self.z_real_temporal_delta_mean.update(float(np.linalg.norm(z_real - self._prev_z_real)))
            self._prev_z_real = z_real.copy()
        raw = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        self.z_norm_ratios.append(float(np.linalg.norm(z_used) / (np.linalg.norm(raw) + 1e-6)))
        self.state_aug_nan_count += int(np.any(np.isnan(state_aug)))
        self.state_aug_inf_count += int(np.any(np.isinf(state_aug)))
        if next_state_aug is not None:
            self.next_state_aug_nan_count += int(np.any(np.isnan(next_state_aug)))
            self.next_state_aug_inf_count += int(np.any(np.isinf(next_state_aug)))
        if str(current.z_source).startswith("shuffled_cross_episode"):
            self.shuffle_cross_episode += 1
        if str(current.z_source).startswith("shuffled_same_episode_far"):
            self.shuffle_same_episode_far += 1
        if bool(current.shuffled_fallback_flag):
            self.shuffle_random_fallback += 1
        if bool(current.shuffled_self_sample_flag):
            self.shuffle_self_sample += 1
        if bool(current.shuffled_nearby_episode_sample_flag):
            self.shuffle_nearby += 1
        if self.trace_max_rows > 0 and self._step_counter % self.trace_stride == 0 and len(self.trace_rows) < self.trace_max_rows:
            self.trace_rows.append({
                "episode_id": int(episode_id),
                "step_id": int(step_id),
                "mode": str(mode),
                "z_source": str(current.z_source),
                "z_real_valid": int(bool(current.z_real_valid)),
                "z_used_valid": int(bool(current.z_used_valid)),
                "fse_forward_executed": int(bool(current.fse_forward_executed)),
                "z_real_norm": float(np.linalg.norm(current.z_real)) if bool(current.z_real_valid) else float("nan"),
                "z_used_norm": float(np.linalg.norm(current.z_used)),
                "z_real_dim_variance": float(np.var(current.z_real)) if bool(current.z_real_valid) else float("nan"),
                "z_used_dim_variance": float(np.var(current.z_used)),
                "next_z_source": str(next_info.z_source) if next_info is not None else "",
                "next_z_real_valid": int(bool(next_info.z_real_valid)) if next_info is not None else 0,
                "next_z_used_valid": int(bool(next_info.z_used_valid)) if next_info is not None else 0,
                "next_z_real_norm": float(np.linalg.norm(next_info.z_real)) if next_info is not None and bool(next_info.z_real_valid) else float("nan"),
                "next_z_used_norm": float(np.linalg.norm(next_info.z_used)) if next_info is not None else float("nan"),
                "fse_state_aug_norm_ratio": float(self.z_norm_ratios[-1]),
                "uncertainty_mean": float(current.uncertainty_mean),
                "risk_mean": float(current.risk_mean),
                "shuffled_source_episode_id": int(current.shuffled_source_episode_id),
                "shuffled_source_step_id": int(current.shuffled_source_step_id),
                "shuffled_fallback_flag": int(bool(current.shuffled_fallback_flag)),
                "shuffled_self_sample_flag": int(bool(current.shuffled_self_sample_flag)),
                "shuffled_nearby_episode_sample_flag": int(bool(current.shuffled_nearby_episode_sample_flag)),
            })
        self._step_counter += 1

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    @staticmethod
    def _safe_percentile(values: List[float], q: float) -> float:
        return float(np.percentile(values, q)) if values else float("nan")

    def fields(self) -> Dict[str, float]:
        total = max(1, int(self.total_records))
        return {
            "fse_z_real_norm_mean": self.z_real_norm_mean.mean(),
            "fse_z_used_norm_mean": self.z_used_norm_mean.mean(),
            "fse_z_used_abs_mean": self.z_used_abs_mean.mean(),
            "fse_z_real_abs_mean": self.z_real_abs_mean.mean(),
            "fse_z_real_dim_variance_mean": self.z_real_dim_var.mean(),
            "fse_z_used_dim_variance_mean": self.z_used_dim_var.mean(),
            "fse_z_real_batch_variance_mean": self.z_real_batch_var.batch_variance_mean(),
            "fse_z_used_batch_variance_mean": self.z_used_batch_var.batch_variance_mean(),
            "fse_z_real_temporal_delta_mean": self.z_real_temporal_delta_mean.mean(),
            "fse_z_used_temporal_delta_mean": self.z_used_temporal_delta_mean.mean(),
            "fse_state_aug_norm_ratio_p10": self._safe_percentile(self.z_norm_ratios, 10),
            "fse_state_aug_norm_ratio_p50": self._safe_percentile(self.z_norm_ratios, 50),
            "fse_state_aug_norm_ratio_p90": self._safe_percentile(self.z_norm_ratios, 90),
            "shuffled_cross_episode_rate": float(self.shuffle_cross_episode / total),
            "shuffled_same_episode_far_rate": float(self.shuffle_same_episode_far / total),
            "shuffled_random_fallback_rate": float(self.shuffle_random_fallback / total),
            "shuffled_self_sample_count": float(self.shuffle_self_sample),
            "shuffled_nearby_episode_sample_count": float(self.shuffle_nearby),
            "state_aug_has_nan": float(self.state_aug_nan_count),
            "state_aug_has_inf": float(self.state_aug_inf_count),
            "next_state_aug_has_nan": float(self.next_state_aug_nan_count),
            "next_state_aug_has_inf": float(self.next_state_aug_inf_count),
        }


class FrozenFSEBottleneckRuntime:
    def __init__(self, cfg: Config, raw_state_dim: int, output_dir: str = ""):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)
        self.raw_state_dim = int(raw_state_dim)
        self.z_mode = str(cfg.fse.z_mode).lower().strip()
        self.fusion_mode = str(cfg.fse.fusion_mode).lower().strip()
        self.run_tier = str(cfg.fse.run_tier).lower().strip()
        self.output_dir = str(output_dir or "").strip()
        self.warning_messages: List[str] = []
        self.error_type = ""
        self.random_std_clipped_dims = 0
        self.fse_forward_count = 0
        self.shuffle_buffer: List[Tuple[int, int, np.ndarray]] = []
        self.shuffle_pool_z: Optional[np.ndarray] = None
        self.shuffle_pool_episode: Optional[np.ndarray] = None
        self.shuffle_pool_step: Optional[np.ndarray] = None
        self.global_mu_z: Optional[np.ndarray] = None
        self.global_std_z: Optional[np.ndarray] = None
        self.global_stats_hash = ""
        self.shuffle_pool_hash = ""
        self.global_stats_metadata: Dict[str, Any] = {}
        self.shuffle_pool_metadata: Dict[str, Any] = {}

        if self.z_mode not in FSE_Z_MODES:
            raise ValueError(f"Unsupported fse_z_mode={self.z_mode}; choose from {FSE_Z_MODES}.")
        if self.fusion_mode not in FSE_FUSION_MODES:
            raise ValueError(f"Unsupported fse_fusion_mode={self.fusion_mode}; choose from {FSE_FUSION_MODES}.")
        self.random_z_distribution = str(cfg.fse.random_z_distribution).lower().strip()
        self._validate_random_z_distribution()
        checkpoint_path = str(cfg.fse.checkpoint_path).strip()
        if not checkpoint_path:
            self.error_type = "missing_checkpoint_error"
            raise ValueError("FSE-RL modes require --fse-checkpoint-path.")
        if not os.path.exists(checkpoint_path):
            self.error_type = "missing_checkpoint_error"
            raise FileNotFoundError(f"FSE checkpoint not found: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.checkpoint = checkpoint
        self.fse_cfg = _fse_config_from_payload(checkpoint.get("fse_config", asdict(cfg.fse)))
        self.z_dim = int(checkpoint.get("z_dim", getattr(self.fse_cfg, "z_dim", 64)))
        self.state_aug_dim = int(self.raw_state_dim + self.z_dim)
        self._validate_checkpoint(checkpoint)
        self.model = build_fse_model(self.fse_cfg, action_dim=0, device=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.fse_param_count_frozen = count_all_params(self.model)
        self.trainable_fse_param_count = count_trainable_params(self.model)
        self.fse_weight_sha256_before = self._model_weight_sha256()
        self.fse_weight_sha256_after = self.fse_weight_sha256_before
        self.model_config_hash = sha256_jsonable(checkpoint.get("fse_config", {}))
        self.scenario_builder_config_hash = scenario_builder_config_hash(cfg)
        self.script_file_hash = sha256_file(__file__)
        self.normalization_payload = fse_structured_normalization_payload(checkpoint)
        self.normalization_legacy_warning = not isinstance(self.normalization_payload, dict)
        if self.normalization_legacy_warning:
            self.warning_messages.append("normalization_config_legacy_warning")
            if self.run_tier == "formal" and not bool(cfg.fse.allow_legacy_normalization):
                self.error_type = "schema_mismatch_error"
                raise ValueError("Formal FSE-RL runs require structured normalization_config or --allow-legacy-fse-normalization.")
        self._load_or_prepare_global_stats()
        self._load_shuffle_pool_if_needed()

    def _validate_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if str(checkpoint.get("scenario_frame_schema_version", "")) != SCENARIO_SCHEMA_VERSION:
            self.error_type = "schema_mismatch_error"
            raise ValueError("FSE checkpoint schema mismatch.")
        if int(checkpoint.get("z_dim", -1)) != 64:
            self.error_type = "schema_mismatch_error"
            raise ValueError("FSE checkpoint z_dim must be 64 for module-3 experiments.")
        if int(checkpoint.get("token_dim", -1)) != SCENARIO_TOKEN_DIM or int(checkpoint.get("n_tokens", -1)) != SCENARIO_TOKEN_COUNT:
            self.error_type = "schema_mismatch_error"
            raise ValueError("FSE checkpoint token shape mismatch.")
        if [int(v) for v in checkpoint.get("horizons", [])] != [10, 20, 40]:
            self.error_type = "schema_mismatch_error"
            raise ValueError("FSE checkpoint horizons must be [10,20,40].")
        if list(checkpoint.get("risk_names", [])) != list(FSE_RISK_NAMES):
            self.error_type = "schema_mismatch_error"
            raise ValueError("FSE checkpoint risk names mismatch.")
        if bool(checkpoint.get("action_conditioned", False)):
            self.error_type = "action_conditioned_checkpoint_error"
            raise ValueError("Module-3 state FSE requires action_conditioned=False checkpoint.")

    def _validate_random_z_distribution(self) -> None:
        if self.random_z_distribution not in FSE_RANDOM_Z_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported fse_random_z_distribution={self.random_z_distribution}; "
                f"choose from {FSE_RANDOM_Z_DISTRIBUTIONS}."
            )
        if self.z_mode not in ("random", "shuffled"):
            return
        if self.run_tier == "formal" and self.random_z_distribution != "global_empirical_matched":
            raise ValueError("Formal random/shuffled FSE-RL runs only support --fse-random-z-distribution global_empirical_matched.")
        if self.run_tier == "smoke" and self.random_z_distribution not in ("global_empirical_matched", "standard_normal"):
            raise ValueError("Smoke random/shuffled FSE-RL runs only support global_empirical_matched or standard_normal.")

    def _model_weight_sha256(self) -> str:
        h = hashlib.sha256()
        for _, tensor in sorted(self.model.state_dict().items(), key=lambda kv: kv[0]):
            arr = tensor.detach().cpu().contiguous().numpy()
            h.update(arr.tobytes())
        return h.hexdigest()

    @staticmethod
    def _plain_npz_value(value: Any) -> Any:
        arr = np.asarray(value)
        if arr.shape == ():
            item = arr.item()
            return item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else _jsonable_plain(item)
        if arr.size == 1:
            item = arr.reshape(-1)[0]
            return item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else _jsonable_plain(item)
        return _jsonable_plain(arr)

    def _metadata_from_npz(self, data) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if "metadata_json" in data.files:
            raw = self._plain_npz_value(data["metadata_json"])
            if str(raw).strip():
                metadata.update(json.loads(str(raw)))
        for key in (
            "num_samples",
            "source_dataset",
            "source_split",
            "checkpoint_sha256",
            "fse_checkpoint_sha256",
            "checkpoint_hash",
            "scenario_builder_config_hash",
            "builder_hash",
            "traffic_density",
        ):
            if key in data.files:
                metadata[key] = self._plain_npz_value(data[key])
        return self._canonical_z_artifact_metadata(metadata)

    def _metadata_from_json_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        if isinstance(payload.get("metadata", None), dict):
            metadata.update(payload["metadata"])
        for key in (
            "num_samples",
            "source_dataset",
            "source_split",
            "checkpoint_sha256",
            "fse_checkpoint_sha256",
            "checkpoint_hash",
            "scenario_builder_config_hash",
            "builder_hash",
            "traffic_density",
        ):
            if key in payload:
                metadata[key] = payload[key]
        return self._canonical_z_artifact_metadata(metadata)

    @staticmethod
    def _canonical_z_artifact_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        out = {str(k): _jsonable_plain(v) for k, v in (metadata or {}).items()}
        if "checkpoint_sha256" not in out:
            for alias in ("fse_checkpoint_sha256", "checkpoint_hash"):
                if alias in out:
                    out["checkpoint_sha256"] = out[alias]
                    break
        if "scenario_builder_config_hash" not in out and "builder_hash" in out:
            out["scenario_builder_config_hash"] = out["builder_hash"]
        return out

    def _validate_z_artifact_metadata(self, metadata: Dict[str, Any], artifact_name: str, sample_count: Optional[int] = None) -> None:
        if self.run_tier != "formal":
            return
        required = ("num_samples", "source_dataset", "source_split", "checkpoint_sha256", "scenario_builder_config_hash", "traffic_density")
        missing = [key for key in required if key not in metadata or str(metadata.get(key, "")).strip() == ""]
        if missing:
            raise ValueError(f"Formal {artifact_name} requires metadata field(s): {missing}.")
        split = str(metadata.get("source_split", "")).lower().strip().replace("-", "_").replace(" ", "_")
        if split in FSE_FORBIDDEN_FORMAL_Z_SOURCE_SPLITS:
            raise ValueError(f"Formal {artifact_name} cannot be sourced from split='{metadata.get('source_split')}'.")
        if str(metadata.get("checkpoint_sha256", "")).strip() != str(self.checkpoint_sha256):
            raise ValueError(f"Formal {artifact_name} checkpoint_sha256 does not match the loaded FSE checkpoint.")
        if str(metadata.get("scenario_builder_config_hash", "")).strip() != str(self.scenario_builder_config_hash):
            raise ValueError(f"Formal {artifact_name} scenario_builder_config_hash does not match current builder config.")
        if str(metadata.get("traffic_density", "")).lower().strip() != str(self.cfg.env.traffic_density).lower().strip():
            raise ValueError(f"Formal {artifact_name} traffic_density does not match current run.")
        try:
            n = int(float(metadata.get("num_samples")))
        except Exception as exc:
            raise ValueError(f"Formal {artifact_name} num_samples must be an integer.") from exc
        min_n = max(1, int(getattr(self.cfg.fse, "formal_min_z_artifact_samples", 32)))
        if n < min_n:
            raise ValueError(f"Formal {artifact_name} num_samples={n} is below required minimum {min_n}.")
        if sample_count is not None and int(sample_count) > 0 and n != int(sample_count):
            raise ValueError(f"Formal {artifact_name} metadata num_samples={n} does not match artifact rows={sample_count}.")

    def _load_stats_file(self, path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if path.lower().endswith(".npz"):
            data = np.load(path, allow_pickle=False)
            mu = np.asarray(data["mu_z"], dtype=np.float32).reshape(-1)
            std = np.asarray(data["std_z"], dtype=np.float32).reshape(-1)
            metadata = self._metadata_from_npz(data)
        else:
            payload = json.loads(open(path, "r", encoding="utf-8").read())
            mu = np.asarray(payload.get("mu_z", []), dtype=np.float32).reshape(-1)
            std = np.asarray(payload.get("std_z", []), dtype=np.float32).reshape(-1)
            metadata = self._metadata_from_json_payload(payload)
        if mu.shape != (self.z_dim,) or std.shape != (self.z_dim,):
            raise ValueError(f"Global z stats must contain mu_z/std_z with shape [{self.z_dim}].")
        self._validate_z_artifact_metadata(metadata, artifact_name="global z stats", sample_count=None)
        floor = max(0.0, float(self.cfg.fse.random_z_std_floor))
        clipped = std < floor
        self.random_std_clipped_dims = int(np.sum(clipped))
        std = np.maximum(std, floor).astype(np.float32)
        return mu.astype(np.float32), std.astype(np.float32), metadata

    def _load_or_prepare_global_stats(self) -> None:
        path = str(self.cfg.fse.z_global_stats_path).strip()
        if self.z_mode in ("random", "shuffled") and self.random_z_distribution == "standard_normal":
            self.global_mu_z = np.zeros((self.z_dim,), dtype=np.float32)
            self.global_std_z = np.ones((self.z_dim,), dtype=np.float32)
            self.global_stats_metadata = {
                "source_split": "synthetic_standard_normal_smoke",
                "num_samples": 0,
                "traffic_density": str(self.cfg.env.traffic_density),
            }
            return
        needs_stats = self.z_mode == "random" or (self.z_mode == "shuffled" and not str(self.cfg.fse.shuffle_pool_path).strip())
        if path and os.path.exists(path):
            self.global_mu_z, self.global_std_z, self.global_stats_metadata = self._load_stats_file(path)
            self.global_stats_hash = sha256_file(path)
            return
        if self.z_mode == "random" and self.run_tier == "formal":
            raise ValueError("Formal random z runs require --fse-z-global-stats-path.")
        if needs_stats:
            self.warning_messages.append("temporary_standard_normal_z_stats_warning")
            self.global_mu_z = np.zeros((self.z_dim,), dtype=np.float32)
            self.global_std_z = np.ones((self.z_dim,), dtype=np.float32)

    def _load_shuffle_pool_if_needed(self) -> None:
        path = str(self.cfg.fse.shuffle_pool_path).strip()
        if self.z_mode != "shuffled":
            return
        if path and os.path.exists(path):
            if path.lower().endswith(".npz"):
                data = np.load(path, allow_pickle=False)
                z = np.asarray(data["z"], dtype=np.float32)
                ep = np.asarray(data["episode_id"], dtype=np.int64).reshape(-1) if "episode_id" in data.files else np.full((z.shape[0],), -1, dtype=np.int64)
                step = np.asarray(data["step_id"], dtype=np.int64).reshape(-1) if "step_id" in data.files else np.full((z.shape[0],), -1, dtype=np.int64)
                metadata = self._metadata_from_npz(data)
            else:
                payload = json.loads(open(path, "r", encoding="utf-8").read())
                z = np.asarray(payload.get("z", []), dtype=np.float32)
                ep = np.asarray(payload.get("episode_id", [-1] * int(z.shape[0])), dtype=np.int64)
                step = np.asarray(payload.get("step_id", [-1] * int(z.shape[0])), dtype=np.int64)
                metadata = self._metadata_from_json_payload(payload)
            if z.ndim != 2 or z.shape[1] != self.z_dim or z.shape[0] <= 0:
                raise ValueError(f"Shuffle pool z must have shape [N,{self.z_dim}] with N>0.")
            self._validate_z_artifact_metadata(metadata, artifact_name="shuffle pool", sample_count=int(z.shape[0]))
            self.shuffle_pool_z = z.astype(np.float32)
            self.shuffle_pool_episode = ep.reshape(-1)
            self.shuffle_pool_step = step.reshape(-1)
            self.shuffle_pool_hash = sha256_file(path)
            self.shuffle_pool_metadata = metadata
            return
        if self.run_tier == "formal":
            raise ValueError("Formal shuffled z runs require --fse-shuffle-pool-path.")
        self.warning_messages.append("temporary_online_shuffle_pool_warning")

    def audit_metadata(self) -> Dict[str, Any]:
        train_log_path = os.path.join(os.path.dirname(os.path.abspath(self.checkpoint_path)), "fse_train_log.csv")
        best_epoch = float("nan")
        final_epoch = float("nan")
        final_epoch_score = float("nan")
        if os.path.exists(train_log_path):
            try:
                with open(train_log_path, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    best = max(rows, key=lambda r: float(r.get("val_best_score", "-inf")))
                    best_epoch = float(best.get("epoch", float("nan")))
                    final_epoch = float(rows[-1].get("epoch", float("nan")))
                    final_epoch_score = float(rows[-1].get("val_best_score", float("nan")))
            except Exception:
                self.warning_messages.append("checkpoint_epoch_audit_infer_failed")
        return {
            "path": os.path.abspath(self.checkpoint_path),
            "selection": "best_val",
            "best_epoch": best_epoch,
            "best_score": float(self.checkpoint.get("best_score", float("nan"))),
            "final_epoch": final_epoch,
            "final_epoch_score": final_epoch_score,
            "schema_version": str(self.checkpoint.get("scenario_frame_schema_version", "")),
            "z_dim": int(self.z_dim),
            "horizons": [int(v) for v in self.checkpoint.get("horizons", [])],
            "risk_heads": list(self.checkpoint.get("risk_names", [])),
            "normalization_spec": _jsonable_plain(self.checkpoint.get("normalization_spec", {})),
            "normalization_config": _jsonable_plain(self.checkpoint.get("normalization_config", {})),
            "normalization_config_legacy_warning": bool(self.normalization_legacy_warning),
            "traffic_density": str(self.cfg.env.traffic_density),
            "fse_model_class": self.model.__class__.__name__,
            "fse_model_config_hash": self.model_config_hash,
            "fse_checkpoint_sha256": self.checkpoint_sha256,
            "scenario_builder_config_hash": self.scenario_builder_config_hash,
            "script_git_sha_or_file_hash": self.script_file_hash,
            "global_z_stats_sha256": self.global_stats_hash,
            "offline_shuffle_pool_sha256": self.shuffle_pool_hash,
            "global_z_stats_metadata": _jsonable_plain(self.global_stats_metadata),
            "offline_shuffle_pool_metadata": _jsonable_plain(self.shuffle_pool_metadata),
            "warnings": list(self.warning_messages),
        }

    def freeze_fields(self) -> Dict[str, Any]:
        self.fse_weight_sha256_after = self._model_weight_sha256()
        delta = 0.0 if self.fse_weight_sha256_after == self.fse_weight_sha256_before else float("nan")
        grad_norm_sq = 0.0
        nonzero = 0
        for p in self.model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                grad_norm_sq += float(torch.sum(g * g).item())
                nonzero += int(torch.any(g != 0).item())
        return {
            "fse_param_count_frozen": int(self.fse_param_count_frozen),
            "trainable_fse_param_count": int(self.trainable_fse_param_count),
            "fse_optimizer_param_count": 0,
            "fse_nonzero_grad_param_count": int(nonzero),
            "fse_grad_norm": float(math.sqrt(max(0.0, grad_norm_sq))),
            "fse_weight_sha256_before": self.fse_weight_sha256_before,
            "fse_weight_sha256_after": self.fse_weight_sha256_after,
            "fse_weight_delta_norm": delta,
        }

    def _forward_real(self, frame: ScenarioFrame) -> Tuple[np.ndarray, float, float]:
        with torch.no_grad():
            tokens = torch.as_tensor(frame.tokens[None, :, :], dtype=torch.float32, device=self.device)
            token_mask = torch.as_tensor(frame.token_mask[None, :], dtype=torch.float32, device=self.device)
            entity_valid = torch.as_tensor(frame.entity_valid_mask[None, :], dtype=torch.float32, device=self.device)
            token_types = torch.as_tensor(frame.token_type_ids[None, :], dtype=torch.long, device=self.device)
            token_roles = torch.as_tensor(frame.token_role_ids[None, :], dtype=torch.long, device=self.device)
            out = self.model(tokens, token_mask, entity_valid, token_types, token_roles)
        z = out.z_fse.detach().cpu().numpy()[0].astype(np.float32)
        if z.shape != (self.z_dim,):
            self.error_type = "shape_mismatch_error"
            raise ValueError(f"z_fse shape mismatch: got {z.shape}, expected {(self.z_dim,)}.")
        if np.any(np.isnan(z)):
            self.error_type = "nan_z_error"
            raise ValueError("FSE produced NaN z.")
        if np.any(np.isinf(z)):
            self.error_type = "inf_z_error"
            raise ValueError("FSE produced Inf z.")
        self.fse_forward_count += 1
        return z, float(out.uncertainty.mean().item()), float(out.risk_probs.mean().item())

    def _random_z(self, episode_id: int, step_id: int, salt: str) -> Tuple[np.ndarray, int]:
        if self.random_z_distribution == "standard_normal":
            mu = np.zeros((self.z_dim,), dtype=np.float32)
            std = np.ones((self.z_dim,), dtype=np.float32)
        else:
            mu = self.global_mu_z if self.global_mu_z is not None else np.zeros((self.z_dim,), dtype=np.float32)
            std = self.global_std_z if self.global_std_z is not None else np.ones((self.z_dim,), dtype=np.float32)
        seed = stable_int_hash(self.cfg.train.seed, episode_id, step_id, self.cfg.train.mode, salt)
        rng = np.random.default_rng(seed)
        z = rng.normal(loc=mu, scale=std).astype(np.float32)
        return z, int(self.random_std_clipped_dims)

    def _sample_shuffled(self, episode_id: int, step_id: int) -> FSEStepZ:
        if self.shuffle_pool_z is not None and self.shuffle_pool_z.shape[0] > 0:
            i = stable_int_hash(self.cfg.train.seed, episode_id, step_id, self.cfg.train.mode, "shuffle_pool") % int(self.shuffle_pool_z.shape[0])
            return FSEStepZ(
                z_real=np.full((self.z_dim,), np.nan, dtype=np.float32),
                z_real_valid=False,
                z_used=self.shuffle_pool_z[int(i)].astype(np.float32),
                z_used_valid=True,
                z_source="shuffled_offline_pool",
                fse_forward_executed=False,
                shuffled_source_episode_id=int(self.shuffle_pool_episode[int(i)]) if self.shuffle_pool_episode is not None else -1,
                shuffled_source_step_id=int(self.shuffle_pool_step[int(i)]) if self.shuffle_pool_step is not None else -1,
            )
        candidates_cross = [(ep, st, z) for ep, st, z in self.shuffle_buffer if int(ep) != int(episode_id)]
        candidates_far = [(ep, st, z) for ep, st, z in self.shuffle_buffer if int(ep) == int(episode_id) and abs(int(st) - int(step_id)) > 50]
        pool = candidates_cross if candidates_cross else candidates_far
        if pool:
            i = stable_int_hash(self.cfg.train.seed, episode_id, step_id, self.cfg.train.mode, "shuffle_buffer") % len(pool)
            ep, st, z = pool[int(i)]
            same_near = int(ep) == int(episode_id) and abs(int(st) - int(step_id)) <= 50
            return FSEStepZ(
                z_real=np.full((self.z_dim,), np.nan, dtype=np.float32),
                z_real_valid=False,
                z_used=np.asarray(z, dtype=np.float32).copy(),
                z_used_valid=True,
                z_source="shuffled_cross_episode" if int(ep) != int(episode_id) else "shuffled_same_episode_far",
                fse_forward_executed=False,
                shuffled_source_episode_id=int(ep),
                shuffled_source_step_id=int(st),
                shuffled_self_sample_flag=bool(int(ep) == int(episode_id) and int(st) == int(step_id)),
                shuffled_nearby_episode_sample_flag=bool(same_near),
            )
        z, clipped = self._random_z(episode_id, step_id, "shuffle_fallback")
        return FSEStepZ(
            z_real=np.full((self.z_dim,), np.nan, dtype=np.float32),
            z_real_valid=False,
            z_used=z,
            z_used_valid=True,
            z_source="shuffled_random_fallback",
            fse_forward_executed=False,
            shuffled_fallback_flag=True,
            random_std_clipped_dims=clipped,
        )

    def append_shuffle_buffer(self, episode_id: int, step_id: int, info: FSEStepZ) -> None:
        if self.z_mode != "shuffled" or not bool(info.z_real_valid):
            return
        self.shuffle_buffer.append((int(episode_id), int(step_id), np.asarray(info.z_real, dtype=np.float32).copy()))
        if len(self.shuffle_buffer) > 20000:
            self.shuffle_buffer = self.shuffle_buffer[-20000:]

    def compute(self, frame: Optional[ScenarioFrame], episode_id: int, step_id: int) -> FSEStepZ:
        nan_real = np.full((self.z_dim,), np.nan, dtype=np.float32)
        if self.z_mode == "zero":
            return FSEStepZ(nan_real, False, np.zeros((self.z_dim,), dtype=np.float32), True, "zero", False)
        if self.z_mode == "random":
            z, clipped = self._random_z(episode_id, step_id, "random")
            return FSEStepZ(nan_real, False, z, True, "random_global_empirical_matched", False, random_std_clipped_dims=clipped)
        if frame is None:
            self.error_type = "shape_mismatch_error"
            raise ValueError("FSE real/shuffled modes require ScenarioFrame.")
        z_real, uncertainty, risk = self._forward_real(frame)
        if self.z_mode == "real":
            return FSEStepZ(z_real, True, z_real.copy(), True, "real", True, uncertainty_mean=uncertainty, risk_mean=risk)
        if self.z_mode == "shuffled":
            used = self._sample_shuffled(episode_id, step_id)
            used.z_real = z_real
            used.z_real_valid = True
            used.fse_forward_executed = True
            used.uncertainty_mean = uncertainty
            used.risk_mean = risk
            return used
        raise ValueError(f"Unsupported z_mode: {self.z_mode}")


class FrozenActionFSEPenaltyRuntime:
    def __init__(self, cfg: Config, action_dim: int, output_dir: str = ""):
        self.cfg = cfg
        self.device = torch.device(cfg.train.device)
        self.expected_action_dim = int(action_dim)
        self.output_dir = str(output_dir or "").strip()
        self.warning_messages: List[str] = []
        self.error_type = ""
        self.forward_count = 0
        checkpoint_path = str(cfg.fse.action_risk_checkpoint_path).strip()
        if not checkpoint_path:
            self.error_type = "missing_action_risk_checkpoint_error"
            raise ValueError("Action-risk modes require --fse-action-risk-checkpoint-path.")
        if not os.path.exists(checkpoint_path):
            self.error_type = "missing_action_risk_checkpoint_error"
            raise FileNotFoundError(f"Action-FSE checkpoint not found: {checkpoint_path}")
        gate_path = str(cfg.fse.action_risk_acceptance_json).strip()
        if not gate_path:
            self.error_type = "missing_action_risk_acceptance_gate_error"
            raise ValueError("Action-risk modes require --fse-action-risk-acceptance-json.")
        if not os.path.exists(gate_path):
            self.error_type = "missing_action_risk_acceptance_gate_error"
            raise FileNotFoundError(f"Action-FSE acceptance gate not found: {gate_path}")

        self.checkpoint_path = checkpoint_path
        self.acceptance_gate_path = gate_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.acceptance_gate_sha256 = sha256_file(gate_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.checkpoint = checkpoint
        self.acceptance_gate = json.loads(open(gate_path, "r", encoding="utf-8").read())
        self.fse_cfg = _fse_config_from_payload(checkpoint.get("fse_config", asdict(cfg.fse)))
        self.action_dim = self._checkpoint_action_dim(checkpoint)
        self.z_dim = int(checkpoint.get("z_dim", getattr(self.fse_cfg, "z_dim", 64)))
        self._validate_checkpoint(checkpoint)
        self._validate_acceptance_gate(self.acceptance_gate)
        self.model = build_fse_model(self.fse_cfg, action_dim=int(self.action_dim), device=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.param_count = count_all_params(self.model)
        self.trainable_param_count = count_trainable_params(self.model)
        self.hash_before = model_weight_sha256(self.model)
        self.hash_after = self.hash_before

    def _checkpoint_action_dim(self, checkpoint: Dict[str, Any]) -> int:
        for key in ("action_dim",):
            if key in checkpoint:
                try:
                    return int(checkpoint[key])
                except Exception:
                    pass
        meta = checkpoint.get("action_metadata", {})
        if isinstance(meta, dict) and "action_dim" in meta:
            try:
                return int(meta["action_dim"])
            except Exception:
                pass
        state = checkpoint.get("model_state_dict", {})
        weight = state.get("action_proj.weight", None) if isinstance(state, dict) else None
        if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
            return int(weight.shape[1])
        return 0

    def _validate_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if str(checkpoint.get("scenario_frame_schema_version", "")) != SCENARIO_SCHEMA_VERSION:
            self.error_type = "schema_mismatch_error"
            raise ValueError("Action-FSE checkpoint schema mismatch.")
        if int(checkpoint.get("token_dim", -1)) != SCENARIO_TOKEN_DIM or int(checkpoint.get("n_tokens", -1)) != SCENARIO_TOKEN_COUNT:
            self.error_type = "schema_mismatch_error"
            raise ValueError("Action-FSE checkpoint token shape mismatch.")
        if [int(v) for v in checkpoint.get("horizons", [])] != [10, 20, 40]:
            self.error_type = "schema_mismatch_error"
            raise ValueError("Action-FSE checkpoint horizons must be [10,20,40].")
        if list(checkpoint.get("risk_names", [])) != list(FSE_RISK_NAMES):
            self.error_type = "schema_mismatch_error"
            raise ValueError("Action-FSE checkpoint risk names mismatch.")
        if not bool(checkpoint.get("action_conditioned", False)):
            self.error_type = "state_conditioned_checkpoint_error"
            raise ValueError("Action-risk runtime requires action_conditioned=True checkpoint.")
        if int(self.action_dim) != 2 or int(self.expected_action_dim) != 2:
            self.error_type = "action_dim_mismatch_error"
            raise ValueError("Action-risk runtime requires action_dim=2.")
        meta = checkpoint.get("action_metadata", {})
        if not isinstance(meta, dict):
            self.error_type = "action_metadata_error"
            raise ValueError("Action-FSE checkpoint requires action_metadata.")
        if str(meta.get("action_space", "")).strip() != FSE_ACTION_SPACE_POLICY_NORMALIZED:
            self.error_type = "action_space_mismatch_error"
            raise ValueError("Action-FSE checkpoint action_space must be policy_normalized_tanh.")
        if [str(v) for v in meta.get("action_order", [])] != list(FSE_ACTION_ORDER):
            self.error_type = "action_order_mismatch_error"
            raise ValueError("Action-FSE checkpoint action_order mismatch.")
        low = np.asarray(meta.get("action_low", []), dtype=np.float32).reshape(-1)
        high = np.asarray(meta.get("action_high", []), dtype=np.float32).reshape(-1)
        if low.shape != (2,) or high.shape != (2,) or not np.allclose(low, [-1.0, -1.0], atol=1e-6) or not np.allclose(high, [1.0, 1.0], atol=1e-6):
            self.error_type = "action_bounds_mismatch_error"
            raise ValueError("Action-FSE checkpoint action bounds must be [-1,1]^2.")

    @staticmethod
    def _gate_get(gate: Dict[str, Any], path: Tuple[str, ...], default: Any = None) -> Any:
        cur: Any = gate
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur

    def _metric_float(self, gate: Dict[str, Any], path: Tuple[str, ...]) -> float:
        raw = self._gate_get(gate, path, float("nan"))
        try:
            return float(raw)
        except Exception:
            return float("nan")

    def _validate_acceptance_gate(self, gate: Dict[str, Any]) -> None:
        blocking: List[str] = []
        warnings: List[str] = []
        if str(gate.get("status", "")).lower().strip() != "passed":
            blocking.append("status != passed")
        gate_hash = str(
            gate.get("checkpoint_sha256")
            or gate.get("fse_checkpoint_sha256")
            or gate.get("checkpoint_hash")
            or ""
        ).strip()
        if gate_hash != str(self.checkpoint_sha256):
            blocking.append("checkpoint_sha256 mismatch")
        if str(gate.get("schema_version", "")).strip() != SCENARIO_SCHEMA_VERSION:
            blocking.append("schema_version mismatch")
        if str(gate.get("dataset_type", "")).strip() != "same_state_branch_rollout":
            blocking.append("dataset_type must be same_state_branch_rollout")
        gate_action_conditioned = gate.get("action_conditioned", self._gate_get(gate, ("action_metadata", "action_conditioned"), False))
        gate_action_dim = gate.get("action_dim", self._gate_get(gate, ("action_metadata", "action_dim"), -1))
        gate_action_space = gate.get("action_space", self._gate_get(gate, ("action_metadata", "action_space"), ""))
        if not bool(gate_action_conditioned):
            blocking.append("action_conditioned must be true")
        if int(gate_action_dim) != 2:
            blocking.append("action_dim must be 2")
        if str(gate_action_space).strip() != FSE_ACTION_SPACE_POLICY_NORMALIZED:
            blocking.append("action_space must be policy_normalized_tanh")
        pair_auc = self._metric_float(gate, ("ranking", "same_state_pair_auc"))
        if not np.isfinite(pair_auc) or pair_auc < 0.65:
            blocking.append("same_state_pair_auc < 0.65")
        z_variance = self._metric_float(gate, ("z_variance",))
        if not np.isfinite(z_variance) or z_variance <= 1e-6:
            blocking.append("z_variance <= 1e-6")
        if not bool(gate.get("risk_grad_smoke_passed", False)):
            blocking.append("risk_grad_smoke_passed is not true")

        profile = str(gate.get("gate_profile", "dev")).lower().strip() or "dev"
        formal_profile = profile == "formal" or str(self.cfg.fse.run_tier).lower().strip() == "formal"
        report_checks = [
            ("risk_recall.collision", self._metric_float(gate, ("risk_recall", "collision")), 0.35, "min"),
            ("risk_recall.offroad", self._metric_float(gate, ("risk_recall", "offroad")), 0.45, "min"),
            ("risk_recall.low_ttc", self._metric_float(gate, ("risk_recall", "low_ttc")), 0.50, "min"),
            ("calibration.ece", self._metric_float(gate, ("calibration", "ece")), 0.10, "max"),
            ("calibration.brier", self._metric_float(gate, ("calibration", "brier")), 0.15, "max"),
            ("monotonic_violation_rate", self._metric_float(gate, ("monotonic_violation_rate",)), 0.05, "max"),
        ]
        for name, value, threshold, kind in report_checks:
            failed = (not np.isfinite(value)) or (value < threshold if kind == "min" else value > threshold)
            if failed and formal_profile:
                blocking.append(f"{name} failed formal threshold")
            elif failed:
                warnings.append(f"{name} failed dev report-only threshold")
        if str(self.cfg.fse.run_tier).lower().strip() == "formal" and profile != "formal":
            blocking.append("formal action-risk run requires gate_profile=formal")
        if bool(self.cfg.fse.strict_action_gate) and warnings:
            blocking.extend(warnings)
        gate_blocking = gate.get("blocking_reasons", [])
        if isinstance(gate_blocking, list) and gate_blocking:
            blocking.extend([f"gate blocking reason: {str(v)}" for v in gate_blocking])
        self.warning_messages.extend(warnings)
        if blocking:
            self.error_type = "action_risk_acceptance_gate_error"
            raise ValueError("Action-FSE acceptance gate failed: " + "; ".join(blocking))

    def freeze_fields(self) -> Dict[str, Any]:
        self.hash_after = model_weight_sha256(self.model)
        return {
            "fse_action_param_count": int(self.param_count),
            "fse_action_trainable_param_count": int(self.trainable_param_count),
            "fse_action_nonzero_grad_param_count": int(count_nonzero_grad_params(self.model)),
            "fse_action_hash_unchanged": float(1.0 if self.hash_before == self.hash_after else 0.0),
        }

    def audit_metadata(self) -> Dict[str, Any]:
        return {
            "path": os.path.abspath(self.checkpoint_path),
            "acceptance_gate_path": os.path.abspath(self.acceptance_gate_path),
            "fse_action_checkpoint_sha256": str(self.checkpoint_sha256),
            "fse_action_acceptance_gate_sha256": str(self.acceptance_gate_sha256),
            "action_dim": int(self.action_dim),
            "action_metadata": _jsonable_plain(self.checkpoint.get("action_metadata", {})),
            "acceptance_gate": _jsonable_plain(self.acceptance_gate),
            "warnings": list(self.warning_messages),
        }

    def compute_penalty(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        entity_valid_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        token_role_ids: torch.Tensor,
        action: torch.Tensor,
        beta_risk: float,
        kappa: float,
    ) -> ActionRiskPenaltyOutput:
        if action.dim() != 2 or action.shape[1] != int(self.action_dim):
            raise ValueError(f"action_for_fse must have shape [B,{self.action_dim}], got {tuple(action.shape)}.")
        action_min = float(action.detach().min().item())
        action_max = float(action.detach().max().item())
        if action_min < -1.0001 or action_max > 1.0001:
            raise ValueError(f"action_for_fse must be in [-1,1], got min={action_min:.6f}, max={action_max:.6f}.")
        # Tokens and masks are replay observations; only the action path remains differentiable.
        out = self.model(
            tokens.detach(),
            token_mask.detach(),
            entity_valid_mask.detach(),
            token_type_ids.detach(),
            token_role_ids.detach(),
            action=action,
        )
        risk_w = torch.as_tensor(FSE_ACTION_RISK_WEIGHTS, dtype=out.risk_probs.dtype, device=out.risk_probs.device).view(1, 1, -1)
        horizon_w = torch.as_tensor(FSE_ACTION_RISK_HORIZON_WEIGHTS, dtype=out.risk_probs.dtype, device=out.risk_probs.device).view(1, -1, 1)
        risk_h = (out.risk_probs * risk_w).sum(dim=-1, keepdim=True)
        risk_raw = (risk_h * horizon_w).sum(dim=1).squeeze(-1)
        risk_norm = risk_raw / torch.clamp(risk_w.sum(), min=1e-6)
        uncertainty_scalar = (out.uncertainty * horizon_w).sum(dim=1).squeeze(-1)
        uncertainty_scalar = torch.clamp(uncertainty_scalar, min=0.0, max=5.0)
        uncertainty_gate = torch.exp(-float(kappa) * uncertainty_scalar).detach()
        risk_penalty = float(beta_risk) * uncertainty_gate * risk_norm
        self.forward_count += 1
        return ActionRiskPenaltyOutput(
            risk_raw=risk_raw,
            risk_norm=risk_norm,
            uncertainty_scalar=uncertainty_scalar,
            uncertainty_gate=uncertainty_gate,
            risk_penalty=risk_penalty,
            risk_probs=out.risk_probs,
            uncertainty=out.uncertainty,
        )


def scenario_builder_config_hash(cfg: Config) -> str:
    payload = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "token_count": SCENARIO_TOKEN_COUNT,
        "token_dim": SCENARIO_TOKEN_DIM,
        "token_order": list(SCENARIO_TOKEN_ORDER),
        "token_type_ids": SCENARIO_TOKEN_TYPE_IDS.astype(int).tolist(),
        "token_role_ids": SCENARIO_TOKEN_ROLE_IDS.astype(int).tolist(),
        "max_speed": float(cfg.env.max_speed),
        "goal_distance": float(cfg.env.goal_distance),
        "lanes_count": int(cfg.env.lanes_count),
        "max_steps": int(cfg.train.max_steps_per_episode),
        "road_boundary_soft_margin": float(cfg.cost.road_boundary_soft_margin),
    }
    return sha256_jsonable(payload)


def fse_env_dt(cfg: Config) -> Tuple[float, List[str]]:
    warnings: List[str] = []
    try:
        freq = float(getattr(cfg.env, "policy_frequency", 0.0))
        if freq > 0.0:
            return float(1.0 / freq), warnings
    except Exception:
        pass
    warnings.append("env_dt fallback to 0.05 because env.policy_frequency is unavailable or invalid.")
    return 0.05, warnings


def fse_label_metadata(cfg: Config, fse_cfg: Optional[FSEConfig] = None) -> Dict[str, Any]:
    local_fse = fse_cfg or cfg.fse
    dt, warnings = fse_env_dt(cfg)
    horizons = [int(v) for v in tuple(local_fse.horizons)]
    return {
        "horizons": horizons,
        "env_dt": float(dt),
        "horizon_seconds": [float(h * dt) for h in horizons],
        "env_dt_warnings": warnings,
        "unsafe_headway_tau": float(FSE_UNSAFE_HEADWAY_TAU),
        "low_ttc_tau": float(FSE_LOW_TTC_TAU),
        "lane_boundary_margin_threshold": float(FSE_LANE_BOUNDARY_MARGIN_THRESHOLD),
        "lane_offset_threshold": f"{FSE_LANE_OFFSET_RATIO_THRESHOLD} * lane_width",
        "cost_safety_definition": FSE_COST_SAFETY_DEFINITION,
        "safety_cost_sum_name": "safety_exposure_cost_sum",
        "cost_lane_definition": FSE_COST_LANE_DEFINITION,
        "terminal_mask_rule": "failure terminal padded as risk=1, normal truncation valid_mask=0",
    }


def fse_is_csac_main_path(path: str) -> bool:
    norm = os.path.normpath(str(path or "")).replace("\\", "/").lower()
    return "/csac_main/" in f"/{norm}/" or norm.endswith("/csac_main")


def fse_source_group_id(mode: str, collect_policy: str, output_dir: str) -> int:
    mode_l = str(mode).lower().strip()
    policy_l = str(collect_policy).lower().strip()
    if policy_l == "random":
        return SOURCE_GROUP_AUX_RANDOM_OOD
    if mode_l == "paper_sac_pure":
        return SOURCE_GROUP_AUX_SAC_RISK
    if fse_is_csac_main_path(output_dir) or mode_l == "paper_csac_lagrangian":
        return SOURCE_GROUP_MAIN_CSAC
    return SOURCE_GROUP_MIXED_ABLATION


def fse_enforce_csac_main_admission(cfg: Config, mode: str, collect_policy: Optional[str] = None) -> None:
    if not fse_is_csac_main_path(str(cfg.fse.output_dir)):
        return
    policy = str(collect_policy if collect_policy is not None else cfg.fse.collect_policy).lower().strip()
    if str(mode).lower().strip() != "paper_csac_lagrangian":
        raise ValueError("csac_main FSE data only admits mode=paper_csac_lagrangian.")
    if policy not in ("online_train", "eval_policy", "noisy_eval"):
        raise ValueError("csac_main FSE data rejects random and auxiliary collect policies.")
    if bool(cfg.fse.allow_untrained_eval_policy) and policy in ("eval_policy", "noisy_eval"):
        raise ValueError("--allow-untrained-eval-policy is invalid for csac_main eval/noisy data.")


def parse_policy_checkpoint_map(text: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for raw in re.split(r"[,;\n]+", str(text or "")):
        token = raw.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid --policy-checkpoint-map token '{token}'. Expected seedN=path.")
        left, right = token.split("=", 1)
        m = re.fullmatch(r"seed(-?\d+)", left.strip().lower())
        if m is None:
            raise ValueError(f"Invalid checkpoint map key '{left}'. Expected seedN.")
        out[int(m.group(1))] = right.strip()
    return out


def infer_checkpoint_seed_id(path: str, fallback: int = -1) -> int:
    text = os.path.normpath(str(path or "")).replace("\\", "/")
    for pattern in (r"/(?:seed[_-]?)?(-?\d+)/checkpoint\.pt$", r"/baseline_csac/(-?\d+)/"):
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))
    return int(fallback)


def resolve_policy_checkpoint_for_seed(cfg: Config, env_seed: int) -> Tuple[str, int]:
    mapping = parse_policy_checkpoint_map(str(cfg.fse.policy_checkpoint_map))
    if int(env_seed) in mapping:
        path = mapping[int(env_seed)]
        return path, infer_checkpoint_seed_id(path, fallback=int(env_seed))
    templ = str(cfg.fse.policy_checkpoint_path_template).strip()
    if templ:
        path = templ.replace("{seed}", str(int(env_seed)))
        return path, infer_checkpoint_seed_id(path, fallback=int(env_seed))
    fixed = str(cfg.fse.policy_checkpoint_path).strip()
    if fixed:
        return fixed, infer_checkpoint_seed_id(fixed, fallback=int(env_seed))
    return "", -1


def fse_mode_id(mode: str) -> int:
    return int(ALL_MODES.index(str(mode).lower().strip())) if str(mode).lower().strip() in ALL_MODES else -1


def fse_traffic_density_id(traffic_density: str) -> int:
    return 1 if str(traffic_density).lower().strip() == "dense" else 0


def fse_done_reason_id(cfg: Config, next_scene: dict, terminated: bool, truncated: bool, timeout_failure: bool, success: bool) -> int:
    if bool(next_scene.get("collision", False)):
        return DONE_COLLISION
    if is_offroad_scene(cfg, next_scene):
        return DONE_OFFROAD
    if bool(success):
        return DONE_GOAL
    if bool(timeout_failure or truncated):
        return DONE_TIMEOUT
    if bool(terminated):
        return DONE_OTHER
    return DONE_NONE


def fse_float(scene: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(scene.get(key, default))
    except Exception:
        return float(default)


def fse_front_vehicle_valid(scene: dict) -> bool:
    return bool(fse_float(scene, "front_distance", 1e6) < 1e5)


def fse_lane_offset_m(scene: dict, cfg: Config) -> float:
    if "lane_offset" in scene:
        return fse_float(scene, "lane_offset", 0.0)
    signed_norm = fse_float(scene, "lane_offset_norm", fse_float(scene, "abs_lane_offset_norm", 0.0))
    return float(signed_norm * max(float(scene.get("lane_width_runtime", cfg.env.lane_width)), 1e-6))


def make_raw_record(
    cfg: Config,
    frame: ScenarioFrame,
    scene: dict,
    next_scene: dict,
    action: np.ndarray,
    reward: float,
    done: bool,
    done_reason_id: int,
    terminated: bool,
    truncated: bool,
    timeout_failure: bool,
    success: bool,
    cost_dict: dict,
    episode_uid_id: int,
    local_episode_id: int,
    step_id: int,
    mode: str,
    env_seed: int,
    checkpoint_seed_id: int,
    policy_train_seed_id: int,
    raw_source_id: int,
    collect_policy_id: int,
    source_group_id: int,
    policy_checkpoint_uid_id: int,
) -> Dict[str, Any]:
    lane_width = float(next_scene.get("lane_width_runtime", cfg.env.lane_width))
    cost_collision = float(cost_dict.get("collision_cost", 0.0))
    cost_headway = float(cost_dict.get("headway_cost", 0.0))
    cost_lane = float(cost_dict.get("lane_cost", 0.0))
    return {
        "tokens": frame.tokens.astype(np.float32),
        "token_mask": frame.token_mask.astype(np.float32),
        "entity_valid_mask": frame.entity_valid_mask.astype(np.float32),
        "token_type_ids": frame.token_type_ids.astype(np.int64),
        "token_role_ids": frame.token_role_ids.astype(np.int64),
        "episode_uid_id": int(episode_uid_id),
        "episode_id_local": int(local_episode_id),
        "step_id": int(step_id),
        "mode_id": int(fse_mode_id(mode)),
        "seed_id": int(env_seed),
        "raw_source_id": int(raw_source_id),
        "env_seed_id": int(env_seed),
        "checkpoint_seed_id": int(checkpoint_seed_id),
        "policy_train_seed_id": int(policy_train_seed_id),
        "collect_policy_id": int(collect_policy_id),
        "source_group_id": int(source_group_id),
        "policy_checkpoint_uid_id": int(policy_checkpoint_uid_id),
        "traffic_density_id": int(fse_traffic_density_id(cfg.env.traffic_density)),
        "action": np.asarray(action, dtype=np.float32).reshape(-1),
        "reward": float(reward),
        "done": int(bool(done)),
        "done_reason_id": int(done_reason_id),
        "terminated": int(bool(terminated)),
        "truncated": int(bool(truncated)),
        "timeout": int(bool(timeout_failure)),
        "goal_reached": int(bool(success)),
        "success": int(bool(success) and not terminal_failure(cfg, next_scene)),
        "ego_x": fse_float(next_scene, "ego_x", fse_float(scene, "ego_x", 0.0)),
        "ego_y": fse_float(next_scene, "ego_y", fse_float(scene, "ego_y", 0.0)),
        "ego_speed": fse_float(next_scene, "ego_speed", 0.0),
        "ego_lane_id": int(fse_float(next_scene, "lane_id", fse_float(next_scene, "ego_lane_id", 0.0))),
        "lane_offset": fse_lane_offset_m(next_scene, cfg),
        "road_boundary_margin": fse_float(next_scene, "road_boundary_margin", 1e6),
        "lane_width": float(lane_width),
        "front_vehicle_valid": int(fse_front_vehicle_valid(next_scene)),
        "front_distance": fse_float(next_scene, "front_distance", 1e6),
        "front_rel_speed": fse_float(next_scene, "front_rel_speed", 0.0),
        "front_ttc": fse_float(next_scene, "front_ttc", fse_float(next_scene, "ttc", 99.0)),
        "collision": int(bool(next_scene.get("collision", False))),
        "crashed": int(bool(next_scene.get("crashed", False)) or bool(next_scene.get("collision", False))),
        "offroad": int(is_offroad_scene(cfg, next_scene)),
        "cost_collision": float(cost_collision),
        "cost_headway": float(cost_headway),
        "cost_lane": float(cost_lane),
        "cost_speed": float(cost_dict.get("overspeed_cost", 0.0)),
        "cost_total": float(cost_dict.get("total_cost", 0.0)),
        "cost_safety": float(max(cost_collision, cost_headway, cost_lane)),
        "memory_key": str(frame.meta.get("memory_key", "")),
    }


def save_fse_raw_records(records: List[Dict[str, Any]], meta: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    if not records:
        raise RuntimeError("FSE collect produced no records.")
    keys = [k for k in records[0].keys() if k != "memory_key"]
    arrays: Dict[str, np.ndarray] = {}
    for key in keys:
        values = [r[key] for r in records]
        first = values[0]
        if isinstance(first, np.ndarray):
            arrays[key] = np.stack(values, axis=0)
        else:
            dtype = np.float32 if isinstance(first, float) else np.int64
            arrays[key] = np.asarray(values, dtype=dtype)
    arrays["actions"] = arrays["action"].astype(np.float32)
    action_low = np.asarray(meta.get("action_space_low", []), dtype=np.float32).reshape(-1)
    action_high = np.asarray(meta.get("action_space_high", []), dtype=np.float32).reshape(-1)
    if action_low.shape[0] == arrays["actions"].shape[1] and action_high.shape[0] == arrays["actions"].shape[1]:
        arrays["actions_policy_norm"] = env_action_to_policy_norm_np(arrays["actions"], action_low, action_high)
    arrays["memory_key"] = np.asarray([str(r.get("memory_key", "")) for r in records])
    raw_path = os.path.join(output_dir, "fse_raw_trajectories.npz")
    np.savez_compressed(raw_path, **arrays)
    meta_path = export_json(meta, os.path.join(output_dir, "fse_raw_meta.json"))
    episodes_path = os.path.join(output_dir, "fse_raw_episodes.jsonl")
    seen_eps: set[int] = set()
    with open(episodes_path, "w", encoding="utf-8") as f:
        for r in records:
            ep = int(r["episode_uid_id"])
            if ep in seen_eps:
                continue
            seen_eps.add(ep)
            f.write(json.dumps(_jsonable_plain({
                "episode_uid_id": ep,
                "episode_id_local": int(r["episode_id_local"]),
                "mode_id": int(r["mode_id"]),
                "env_seed_id": int(r["env_seed_id"]),
                "collect_policy_id": int(r["collect_policy_id"]),
                "source_group_id": int(r["source_group_id"]),
                "policy_checkpoint_uid_id": int(r["policy_checkpoint_uid_id"]),
            }), ensure_ascii=False) + "\n")
    return {
        "fse_raw_trajectories_npz_path": os.path.abspath(raw_path),
        "fse_raw_meta_json_path": os.path.abspath(meta_path),
        "fse_raw_episodes_jsonl_path": os.path.abspath(episodes_path),
    }


def run_fse_collect(cfg: Config, modes: Optional[List[str]] = None, seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    output_dir = str(cfg.fse.output_dir).strip() or "results/fse/collect"
    os.makedirs(output_dir, exist_ok=True)
    modes = modes or [str(cfg.train.mode)]
    seeds = seeds or [int(cfg.train.seed)]
    collect_policy = str(cfg.fse.collect_policy).lower().strip()
    collect_policy_id = int(FSE_COLLECT_POLICY_TO_ID[collect_policy])
    records: List[Dict[str, Any]] = []
    policy_checkpoint_uid_mapping: Dict[int, str] = {0: "NO_CHECKPOINT"}
    policy_checkpoint_uid_lookup: Dict[str, int] = {}
    seed_mapping: Dict[str, Any] = {}
    warning_messages: List[str] = []
    raw_source_id = 0
    episode_uid_counter = 0
    action_space_low_seen: List[float] = []
    action_space_high_seen: List[float] = []

    for mode in modes:
        fse_enforce_csac_main_admission(cfg, mode, collect_policy)
        for seed in seeds:
            local_cfg = copy.deepcopy(cfg)
            local_cfg.train.mode = str(mode)
            local_cfg.train.seed = int(seed)
            local_cfg.train.enable_scenario_frame = True
            local_cfg.eval.enabled = False
            configure_mode_behavior(local_cfg, local_cfg.train.mode)
            source_group_id = fse_source_group_id(local_cfg.train.mode, collect_policy, output_dir)
            checkpoint_path = ""
            checkpoint_payload: Dict[str, Any] = {}
            checkpoint_seed_id = -1
            policy_checkpoint_uid_id = 0
            if collect_policy in ("eval_policy", "noisy_eval"):
                checkpoint_path, checkpoint_seed_id = resolve_policy_checkpoint_for_seed(local_cfg, int(seed))
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    if fse_is_csac_main_path(output_dir) or not bool(local_cfg.fse.allow_untrained_eval_policy):
                        raise FileNotFoundError(
                            f"{collect_policy} requires a trained baseline checkpoint for seed {seed}; got '{checkpoint_path}'."
                        )
                    warning_messages.append(f"untrained eval/noisy policy used for seed {seed}; debug only")
                if checkpoint_path:
                    ck_abs = os.path.abspath(checkpoint_path)
                    if ck_abs not in policy_checkpoint_uid_lookup:
                        new_id = len(policy_checkpoint_uid_lookup) + 1
                        policy_checkpoint_uid_lookup[ck_abs] = new_id
                        policy_checkpoint_uid_mapping[new_id] = ck_abs
                    policy_checkpoint_uid_id = int(policy_checkpoint_uid_lookup[ck_abs])
                seed_mapping[f"env_seed_{int(seed)}"] = {
                    "checkpoint_seed": int(checkpoint_seed_id),
                    "checkpoint_path": os.path.abspath(checkpoint_path) if checkpoint_path else "",
                }
            policy_train_seed_id = int(checkpoint_seed_id if checkpoint_seed_id >= 0 else (seed if collect_policy == "online_train" else -1))

            env = HighwayEnvWrapper(local_cfg, render=False)
            state, _ = env.reset(seed=int(seed))
            state_dim = int(state.shape[0])
            action_dim = int(env.action_dim())
            action_space_low_seen = np.asarray(env.action_low, dtype=np.float32).reshape(-1).tolist()
            action_space_high_seen = np.asarray(env.action_high, dtype=np.float32).reshape(-1).tolist()
            agent = build_agent(state_dim, action_dim, local_cfg, env.action_low, env.action_high)
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint_payload = load_agent_checkpoint_for_eval(agent, local_cfg, checkpoint_path, state_dim, action_dim, env.action_low, env.action_high)
            scenario_builder = ScenarioFrameBuilder(local_cfg)
            global_env_step = 0
            actor_update_count = 0
            critic_update_count = 0
            lambda_update_count = 0
            rng = np.random.default_rng(int(seed) + 1009)
            try:
                for ep in range(int(local_cfg.train.episodes)):
                    episode_uid_id = int(episode_uid_counter)
                    episode_uid_counter += 1
                    state, _ = env.reset(seed=int(seed) + int(ep))
                    agent.reset_episode_context()
                    for t in range(int(local_cfg.train.max_steps_per_episode)):
                        scene = env.get_scene_dict()
                        frame = scenario_builder.build(
                            obs=state,
                            scene=scene,
                            last_action=env.last_action,
                            episode_id=int(episode_uid_id),
                            step_id=int(t),
                            traffic_density=local_cfg.env.traffic_density,
                        )
                        if collect_policy == "random":
                            action = env.sample_random_action()
                        elif collect_policy == "online_train":
                            if global_env_step < int(local_cfg.sac.start_steps):
                                action = env.sample_random_action()
                            else:
                                action = env.clip_action(agent.select_action(state, evaluate=False))
                        else:
                            action = env.clip_action(agent.select_action(state, evaluate=True))
                            if collect_policy == "noisy_eval":
                                noise = rng.normal(
                                    loc=0.0,
                                    scale=np.asarray([float(local_cfg.fse.noise_std_acc), float(local_cfg.fse.noise_std_steer)], dtype=np.float32),
                                    size=(action_dim,),
                                ).astype(np.float32)
                                noise = np.clip(noise, -float(local_cfg.fse.noise_clip), float(local_cfg.fse.noise_clip))
                                action = env.clip_action(action + noise)

                        next_state, env_reward, terminated, truncated, _ = env.step(action)
                        next_scene = env.get_scene_dict()
                        next_is_success = terminal_success(local_cfg, next_scene)
                        next_is_failure = terminal_failure(local_cfg, next_scene)
                        timeout_failure = bool((truncated or (t + 1 >= local_cfg.train.max_steps_per_episode)) and not next_is_success and not next_is_failure)
                        reward, _ = compute_reward(local_cfg, scene, next_scene, action, env_reward, timeout_failure=timeout_failure)
                        cost_dict = compute_costs(local_cfg, scene, next_scene, action)
                        done = bool(terminated or truncated or next_is_failure or next_is_success or timeout_failure)
                        done_reason = fse_done_reason_id(local_cfg, next_scene, terminated, truncated, timeout_failure, next_is_success)
                        if collect_policy == "online_train":
                            danger_info = classify_transition_danger(local_cfg, scene, next_scene, cost_dict)
                            agent.replay.add(
                                state,
                                action,
                                reward,
                                next_state,
                                float(done),
                                cost_dict=cost_dict,
                                transition_meta={
                                    "episode_id": int(episode_uid_id),
                                    "step_in_episode": int(t),
                                    "danger_label": int(danger_info["danger_label"]),
                                    "is_collision": bool(danger_info["is_collision"]),
                                    "is_near_danger": bool(danger_info["is_near_danger"]),
                                },
                            )
                            if global_env_step >= local_cfg.sac.update_after and global_env_step % local_cfg.sac.update_every == 0:
                                info = agent.update()
                                if info:
                                    actor_update_count += int("actor_loss" in info)
                                    critic_update_count += int("q1_loss" in info or "safety_q1_loss" in info)
                                    lambda_update_count += int(any(k.startswith("lambda") for k in info.keys()))
                        records.append(
                            make_raw_record(
                                local_cfg,
                                frame,
                                scene,
                                next_scene,
                                action,
                                reward,
                                done,
                                done_reason,
                                terminated,
                                truncated,
                                timeout_failure,
                                next_is_success,
                                cost_dict,
                                episode_uid_id,
                                ep,
                                t,
                                local_cfg.train.mode,
                                int(seed),
                                checkpoint_seed_id,
                                policy_train_seed_id,
                                raw_source_id,
                                collect_policy_id,
                                source_group_id,
                                policy_checkpoint_uid_id,
                            )
                        )
                        state = next_state
                        global_env_step += 1
                        if done:
                            break
            finally:
                env.close()

            meta_run = {
                "mode": local_cfg.train.mode,
                "env_seed": int(seed),
                "collect_policy": collect_policy,
                "policy_update_enabled": bool(collect_policy == "online_train"),
                "start_steps": int(local_cfg.sac.start_steps),
                "update_after": int(local_cfg.sac.update_after),
                "update_every": int(local_cfg.sac.update_every),
                "updates_per_step": 1,
                "global_env_step": int(global_env_step),
                "actor_update_count": int(actor_update_count),
                "critic_update_count": int(critic_update_count),
                "lambda_update_count": int(lambda_update_count),
                "start_step_random_rate": float(min(1.0, float(local_cfg.sac.start_steps) / max(1.0, float(global_env_step)))),
                "train_action_stochastic": bool(collect_policy == "online_train"),
                "checkpoint_path": os.path.abspath(checkpoint_path) if checkpoint_path else "",
                "checkpoint_seed_id": int(checkpoint_seed_id),
                "checkpoint_mode": checkpoint_payload.get("mode", "") if checkpoint_payload else "",
            }
            seed_mapping[f"run_seed_{int(seed)}"] = meta_run

    meta = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "raw_contract_version": "fse_raw_v1_csac_mixed",
        "collect_policy": collect_policy,
        "collect_policy_id": int(collect_policy_id),
        "source_group_contract": {
            "MAIN_CSAC": SOURCE_GROUP_MAIN_CSAC,
            "AUX_RANDOM_OOD": SOURCE_GROUP_AUX_RANDOM_OOD,
            "AUX_SAC_RISK": SOURCE_GROUP_AUX_SAC_RISK,
            "MIXED_ABLATION": SOURCE_GROUP_MIXED_ABLATION,
        },
        "policy_checkpoint_uid_mapping": policy_checkpoint_uid_mapping,
        "seed_mapping": seed_mapping,
        "online_train_training_run_id": "fse_collect_online_train_separate_from_stage0" if collect_policy == "online_train" else "",
        "eval_noisy_checkpoint_run_id": "stage0_baseline_csac_checkpoint" if collect_policy in ("eval_policy", "noisy_eval") else "",
        "action_space_low": action_space_low_seen,
        "action_space_high": action_space_high_seen,
        "noise_space": "normalized_action",
        "action_order": ["acceleration", "steering"],
        "noise_std_acc": float(cfg.fse.noise_std_acc),
        "noise_std_steer": float(cfg.fse.noise_std_steer),
        "noise_clip": float(cfg.fse.noise_clip),
        "label_metadata": fse_label_metadata(cfg),
        "warnings": warning_messages,
        "num_records": int(len(records)),
    }
    paths = save_fse_raw_records(records, meta, output_dir)
    if records:
        scenario_probe = {"schema_version": SCENARIO_SCHEMA_VERSION, "first_record_memory_key": str(records[0].get("memory_key", ""))}
        paths["scenario_frame_probe_json_path"] = export_json(scenario_probe, os.path.join(output_dir, "scenario_frame_probe.json"))
    return {"status": "ok", "num_records": int(len(records)), **paths}


def fse_load_raw_files(raw_path_expr: str) -> Tuple[List[Dict[str, np.ndarray]], List[str]]:
    paths = [p.strip() for p in str(raw_path_expr or "").split(",") if p.strip()]
    if not paths:
        raise ValueError("--fse-raw-path is required for --fse-task build-dataset.")
    raw_list: List[Dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            raw_list.append({key: np.asarray(npz[key]) for key in npz.files})
    return raw_list, paths


def fse_merge_raw(raw_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    required = (
        "tokens", "token_mask", "entity_valid_mask", "token_type_ids", "token_role_ids",
        "episode_uid_id", "episode_id_local", "step_id", "mode_id", "seed_id", "raw_source_id",
        "env_seed_id", "checkpoint_seed_id", "policy_train_seed_id", "collect_policy_id", "source_group_id",
        "policy_checkpoint_uid_id", "actions", "reward", "done", "done_reason_id", "terminated", "truncated",
        "timeout", "goal_reached", "success", "ego_x", "ego_y", "ego_speed", "ego_lane_id", "lane_offset",
        "road_boundary_margin", "lane_width", "front_vehicle_valid", "front_distance", "front_rel_speed",
        "front_ttc", "collision", "crashed", "offroad", "cost_collision", "cost_headway", "cost_lane",
        "cost_speed", "cost_total", "cost_safety",
    )
    for raw in raw_list:
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Raw FSE file missing required key(s): {missing}")
    normalized_raw_list: List[Dict[str, np.ndarray]] = []
    for raw_i, raw in enumerate(raw_list):
        copied = dict(raw)
        copied["raw_source_id"] = np.full_like(raw["raw_source_id"].astype(np.int64), int(raw_i), dtype=np.int64)
        normalized_raw_list.append(copied)
    raw_list = normalized_raw_list
    merged: Dict[str, np.ndarray] = {}
    for key in required:
        merged[key] = np.concatenate([raw[key] for raw in raw_list], axis=0)
    for optional in ("memory_key", "traffic_density_id", "actions_policy_norm"):
        if all(optional in raw for raw in raw_list):
            merged[optional] = np.concatenate([raw[optional] for raw in raw_list], axis=0)
    # Raw files have local episode ids; re-map to globally unique integer ids after merge.
    tuples = list(
        zip(
            merged["raw_source_id"].astype(np.int64).tolist(),
            merged["mode_id"].astype(np.int64).tolist(),
            merged["env_seed_id"].astype(np.int64).tolist(),
            merged["collect_policy_id"].astype(np.int64).tolist(),
            merged["policy_checkpoint_uid_id"].astype(np.int64).tolist(),
            merged["episode_uid_id"].astype(np.int64).tolist(),
        )
    )
    uid_map: Dict[Tuple[int, int, int, int, int, int], int] = {}
    remap = np.zeros((len(tuples),), dtype=np.int64)
    for i, tup in enumerate(tuples):
        if tup not in uid_map:
            uid_map[tup] = len(uid_map)
        remap[i] = int(uid_map[tup])
    merged["episode_uid_id"] = remap
    merged["episode_id"] = remap
    return merged


def fse_step_risk_arrays(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    front_valid = raw["front_vehicle_valid"].astype(bool)
    ego_speed = np.maximum(raw["ego_speed"].astype(np.float32), 1e-3)
    front_distance = raw["front_distance"].astype(np.float32)
    front_ttc = raw["front_ttc"].astype(np.float32)
    closing_speed = np.maximum(0.0, -raw["front_rel_speed"].astype(np.float32))
    lane_width = np.maximum(raw["lane_width"].astype(np.float32), 1e-6)
    lane_boundary = np.logical_or(
        raw["road_boundary_margin"].astype(np.float32) < float(FSE_LANE_BOUNDARY_MARGIN_THRESHOLD),
        np.abs(raw["lane_offset"].astype(np.float32)) > float(FSE_LANE_OFFSET_RATIO_THRESHOLD) * lane_width,
    )
    return {
        "collision": np.logical_or(raw["collision"] > 0, raw["crashed"] > 0),
        "offroad": raw["offroad"] > 0,
        "unsafe_headway": np.logical_and(front_valid, front_distance / ego_speed < float(FSE_UNSAFE_HEADWAY_TAU)),
        "low_ttc": np.logical_and.reduce((front_valid, closing_speed > 0.0, front_ttc < float(FSE_LOW_TTC_TAU))),
        "lane_boundary": lane_boundary,
    }


def build_fse_labels_from_raw(raw: Dict[str, np.ndarray], cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    n = int(raw["tokens"].shape[0])
    horizons = [int(v) for v in tuple(cfg.fse.horizons)]
    labels_binary = np.zeros((n, len(horizons), len(FSE_RISK_NAMES)), dtype=np.float32)
    labels_exposure = np.zeros((n, len(horizons), len(FSE_EXPOSURE_NAMES)), dtype=np.float32)
    labels_reg = np.zeros((n, len(horizons), len(FSE_REGRESSION_NAMES)), dtype=np.float32)
    valid_mask = np.zeros((n, len(horizons)), dtype=np.float32)
    risks = fse_step_risk_arrays(raw)
    ep_ids = raw["episode_uid_id"].astype(np.int64)
    done_reason = raw["done_reason_id"].astype(np.int64)
    failure_step = np.logical_or.reduce((
        raw["collision"] > 0,
        raw["crashed"] > 0,
        raw["offroad"] > 0,
        np.isin(done_reason, np.asarray([DONE_COLLISION, DONE_OFFROAD], dtype=np.int64)),
    ))
    done_step = raw["done"] > 0
    debug_samples: List[Dict[str, Any]] = []

    done_other = done_step & (done_reason == DONE_OTHER)
    label_stats: Dict[str, Any] = {
        "done_other_count": int(np.sum(done_other)),
        "done_other_with_failure_flags_count": int(np.sum(done_other & failure_step)),
        "done_other_without_failure_flags_count": int(np.sum(done_other & ~failure_step)),
    }
    done_count = int(np.sum(done_step))
    if done_count > 0 and float(label_stats["done_other_count"]) / max(1.0, float(done_count)) > 0.10:
        label_stats["warning_done_other_ratio"] = "DONE_OTHER ratio > 0.10; check terminal mapping."

    for ep in np.unique(ep_ids):
        idx = np.where(ep_ids == int(ep))[0]
        idx = idx[np.argsort(raw["step_id"][idx].astype(np.int64))]
        if idx.size <= 0:
            continue
        ep_failure_any = bool(np.any(failure_step[idx]))
        terminal_pos = int(np.argmax(done_step[idx])) if np.any(done_step[idx]) else int(idx.size - 1)
        terminal_global_idx = int(idx[terminal_pos])
        terminal_failure = bool(ep_failure_any and np.any(failure_step[idx[: terminal_pos + 1]]))
        for local_pos, global_i in enumerate(idx.tolist()):
            for h_i, H in enumerate(horizons):
                end_pos = local_pos + int(H)
                future_pos = np.arange(local_pos + 1, min(end_pos, idx.size - 1) + 1, dtype=np.int64)
                insufficient = end_pos >= idx.size
                if insufficient and not terminal_failure:
                    valid_mask[global_i, h_i] = 0.0
                    continue
                if future_pos.size <= 0:
                    if terminal_failure:
                        future_idx = np.asarray([terminal_global_idx], dtype=np.int64)
                    else:
                        continue
                else:
                    future_idx = idx[future_pos]
                valid_mask[global_i, h_i] = 1.0
                for r_i, name in enumerate(FSE_RISK_NAMES):
                    value = bool(np.any(risks[name][future_idx]))
                    if insufficient and terminal_failure and name in ("collision", "offroad", "lane_boundary"):
                        value = value or bool(np.any(risks[name][idx[local_pos + 1: terminal_pos + 1]]))
                    labels_binary[global_i, h_i, r_i] = 1.0 if value else 0.0
                for e_i, name in enumerate(FSE_EXPOSURE_NAMES):
                    observed = risks[name][future_idx].astype(np.float32)
                    if insufficient and terminal_failure and name in ("offroad", "lane_boundary"):
                        pad = np.ones((max(0, int(H) - int(observed.shape[0])),), dtype=np.float32)
                        observed = np.concatenate([observed, pad], axis=0)
                    labels_exposure[global_i, h_i, e_i] = float(np.mean(observed)) if observed.size > 0 else 0.0
                min_ttc = float(np.min(raw["front_ttc"][future_idx])) if future_idx.size > 0 else 99.0
                min_front = float(np.min(raw["front_distance"][future_idx])) if future_idx.size > 0 else 1e6
                min_margin = float(np.min(raw["road_boundary_margin"][future_idx])) if future_idx.size > 0 else 3.0
                progress = float(raw["ego_x"][future_idx[-1]] - raw["ego_x"][global_i]) if future_idx.size > 0 else 0.0
                safety_sum = float(np.sum(raw["cost_safety"][future_idx]))
                labels_reg[global_i, h_i, :] = np.asarray(
                    [
                        np.clip(min_ttc / 10.0, 0.0, 1.0),
                        np.clip(min_front / 100.0, 0.0, 1.0),
                        np.clip(min_margin / 3.0, 0.0, 1.0),
                        np.clip(progress / 100.0, 0.0, 1.0),
                        np.clip(safety_sum / max(float(H), 1.0), 0.0, 1.0),
                    ],
                    dtype=np.float32,
                )
            if len(debug_samples) < 200:
                debug_samples.append({
                    "episode_uid_id": int(ep),
                    "step_id": int(raw["step_id"][global_i]),
                    "done_reason": DONE_REASON_ID_TO_NAME.get(int(done_reason[global_i]), "unknown"),
                    "valid_mask": valid_mask[global_i].tolist(),
                    "labels_binary": labels_binary[global_i].tolist(),
                    "labels_exposure": labels_exposure[global_i].tolist(),
                    "labels_reg": labels_reg[global_i].tolist(),
                })

    for h_i, h in enumerate(horizons):
        label_stats[f"valid_rate_H{h}"] = float(np.mean(valid_mask[:, h_i] > 0.5))
        for r_i, name in enumerate(FSE_RISK_NAMES):
            mask = valid_mask[:, h_i] > 0.5
            label_stats[f"positive_rate_H{h}_{name}"] = float(np.mean(labels_binary[mask, h_i, r_i] > 0.5)) if np.any(mask) else float("nan")
    label_stats["failure_terminal_count"] = int(np.sum(done_step & failure_step))
    label_stats["normal_truncation_count"] = int(np.sum(done_step & ~failure_step))
    return labels_binary, labels_exposure, labels_reg, valid_mask, label_stats, debug_samples


def split_raw_dataset_indices(data: Dict[str, np.ndarray], cfg: Config) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    fse_cfg = copy.deepcopy(cfg.fse)
    splits = split_fse_by_episode(data, seed=int(cfg.train.seed), fse_cfg=fse_cfg)
    ep_key = "episode_uid_id"
    requested = [s.strip().lower() for s in str(cfg.fse.split_stratify_by).split(",") if s.strip()]
    ep_first = {int(ep): int(np.where(data[ep_key] == ep)[0][0]) for ep in np.unique(data[ep_key])}
    group_counts: Dict[str, int] = defaultdict(int)
    for ep, first_idx in ep_first.items():
        values: List[str] = []
        for name in requested:
            if name in ("collect_policy", "collect_policy_id") and "collect_policy_id" in data:
                values.append(f"collect_policy={int(data['collect_policy_id'][first_idx])}")
            elif name in ("traffic_density", "traffic_density_id") and "traffic_density_id" in data:
                values.append(f"traffic_density={int(data['traffic_density_id'][first_idx])}")
            elif name in ("mode", "mode_id") and "mode_id" in data:
                values.append(f"mode={int(data['mode_id'][first_idx])}")
        group_counts["|".join(values) if values else "all"] += 1
    groups_too_small = [name for name, count in group_counts.items() if int(count) < 3]
    split_report = {
        "train_episode_ids": sorted([int(v) for v in np.unique(data[ep_key][splits["train"]]).tolist()]),
        "val_episode_ids": sorted([int(v) for v in np.unique(data[ep_key][splits["val"]]).tolist()]),
        "test_episode_ids": sorted([int(v) for v in np.unique(data[ep_key][splits["test"]]).tolist()]),
        "stratification_keys": str(cfg.fse.split_stratify_by),
        "stratification_fallback_count": int(len(groups_too_small)),
        "groups_too_small": groups_too_small,
    }
    return splits, split_report


def fse_source_distribution(data: Dict[str, np.ndarray], labels_binary: np.ndarray, valid_mask: np.ndarray) -> Dict[str, Any]:
    out = fse_source_distribution_for_indices(data, np.arange(data["tokens"].shape[0], dtype=np.int64))
    if "collect_policy_id" in data:
        collect = data["collect_policy_id"].astype(np.int64)
        for policy_id, policy_name in FSE_COLLECT_ID_TO_POLICY.items():
            mask = collect == int(policy_id)
            if not np.any(mask):
                continue
            payload = out.setdefault(str(policy_name), {})
            payload["episode_count"] = int(np.unique(data["episode_uid_id"][mask]).shape[0])
            payload["valid_rate_by_horizon"] = {
                f"H{int(Config().fse.horizons[h_i]) if h_i < len(Config().fse.horizons) else h_i}": float(np.mean(valid_mask[mask, h_i] > 0.5))
                for h_i in range(int(valid_mask.shape[1]))
            }
            payload["major_risk_support"] = int(np.sum((labels_binary[mask][:, :, [0, 2, 3]] > 0.5) & (valid_mask[mask][:, :, None] > 0.5)))
    return out


def run_fse_build_dataset(cfg: Config) -> Dict[str, Any]:
    if bool(getattr(cfg.fse, "source_balanced_sampler", False)):
        raise ValueError("--fse-source-balanced-sampler is a train-time option and is rejected by build-dataset.")
    output_dir = str(cfg.fse.output_dir).strip() or "results/fse/dataset"
    os.makedirs(output_dir, exist_ok=True)
    raw_list, raw_paths = fse_load_raw_files(str(cfg.fse.raw_path))
    raw = fse_merge_raw(raw_list)
    if fse_is_csac_main_path(output_dir):
        bad_source = raw["source_group_id"].astype(np.int64) != SOURCE_GROUP_MAIN_CSAC
        bad_mode = raw["mode_id"].astype(np.int64) != fse_mode_id("paper_csac_lagrangian")
        bad_policy = ~np.isin(raw["collect_policy_id"].astype(np.int64), np.asarray([COLLECT_ONLINE_TRAIN, COLLECT_EVAL_POLICY, COLLECT_NOISY_EVAL], dtype=np.int64))
        if np.any(bad_source) or np.any(bad_mode) or np.any(bad_policy):
            raise ValueError("csac_main build rejects auxiliary source groups, paper_sac_pure, and random data.")
        eval_noisy = np.isin(raw["collect_policy_id"].astype(np.int64), np.asarray([COLLECT_EVAL_POLICY, COLLECT_NOISY_EVAL], dtype=np.int64))
        if np.any(eval_noisy & (raw["policy_checkpoint_uid_id"].astype(np.int64) <= 0)):
            raise ValueError("csac_main eval_policy/noisy_eval rows must carry trained checkpoint uid ids.")
    labels_binary, labels_exposure, labels_reg, valid_mask, label_stats, debug_samples = build_fse_labels_from_raw(raw, cfg)
    dataset = {
        "tokens": raw["tokens"].astype(np.float32),
        "token_mask": raw["token_mask"].astype(np.float32),
        "entity_valid_mask": raw["entity_valid_mask"].astype(np.float32),
        "token_type_ids": raw["token_type_ids"].astype(np.int64),
        "token_role_ids": raw["token_role_ids"].astype(np.int64),
        "episode_uid_id": raw["episode_uid_id"].astype(np.int64),
        "episode_id": raw["episode_uid_id"].astype(np.int64),
        "episode_id_local": raw["episode_id_local"].astype(np.int64),
        "step_id": raw["step_id"].astype(np.int64),
        "labels_binary": labels_binary,
        "labels_exposure": labels_exposure,
        "labels_reg": labels_reg,
        "valid_mask": valid_mask,
        "actions": raw["actions"].astype(np.float32),
    }
    if "actions_policy_norm" in raw:
        dataset["actions_policy_norm"] = np.clip(raw["actions_policy_norm"].astype(np.float32), -1.0, 1.0)
    else:
        action_low, action_high = default_policy_action_bounds(raw["actions"].shape[1])
        dataset["actions_policy_norm"] = env_action_to_policy_norm_np(raw["actions"].astype(np.float32), action_low, action_high)
    for key in (
        "mode_id", "seed_id", "raw_source_id", "env_seed_id", "checkpoint_seed_id", "policy_train_seed_id",
        "collect_policy_id", "source_group_id", "policy_checkpoint_uid_id", "traffic_density_id",
    ):
        if key in raw:
            dataset[key] = raw[key].astype(np.int64)
    if "memory_key" in raw:
        dataset["memory_key"] = raw["memory_key"]
    dataset_path = os.path.join(output_dir, "fse_dataset.npz")
    np.savez_compressed(dataset_path, **dataset)
    splits, split_report = split_raw_dataset_indices(dataset, cfg)
    split_path = export_json(split_report, os.path.join(output_dir, "fse_split.json"))
    meta = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "dataset_contract_version": "fse_dataset_v1_csac_mixed",
        "risk_names": list(FSE_RISK_NAMES),
        "exposure_names": list(FSE_EXPOSURE_NAMES),
        "regression_names": list(FSE_REGRESSION_NAMES),
        "target_source_ratio": str(cfg.fse.source_ratio),
        "actual_source_ratio": fse_source_distribution_for_indices(dataset, np.arange(dataset["tokens"].shape[0], dtype=np.int64)),
        "label_metadata": fse_label_metadata(cfg),
        "action_metadata": {
            "action_space": FSE_ACTION_SPACE_POLICY_NORMALIZED,
            "action_low": [-1.0, -1.0],
            "action_high": [1.0, 1.0],
            "action_order": list(FSE_ACTION_ORDER),
            "source": "actions_policy_norm",
        },
    }
    meta_path = export_json(meta, os.path.join(output_dir, "fse_dataset_meta.json"))
    label_distribution = {
        "num_samples": int(dataset["tokens"].shape[0]),
        "num_episodes": int(np.unique(dataset["episode_uid_id"]).shape[0]),
        "samples_per_split": {k: int(v.shape[0]) for k, v in splits.items()},
        **label_stats,
    }
    quality_warnings: List[str] = []
    if label_distribution["num_episodes"] < 50:
        quality_warnings.append("num_episodes < 50; acceptable for tiny smoke only.")
    if float(label_stats.get("valid_rate_H20", 0.0)) < 0.70:
        quality_warnings.append("valid_rate_H20 < 0.70.")
    if float(label_stats.get("valid_rate_H40", 0.0)) < 0.50:
        quality_warnings.append("valid_rate_H40 < 0.50.")
    if float(label_stats.get("positive_rate_H20_low_ttc", 0.0) or 0.0) <= 0.0:
        quality_warnings.append("low_ttc_positive_rate_H20 is zero; low_ttc head cannot be meaningfully trained/evaluated.")
    if float(label_stats.get("positive_rate_H20_unsafe_headway", 0.0) or 0.0) <= 0.0:
        quality_warnings.append("unsafe_headway_positive_rate_H20 is zero; headway head cannot be meaningfully trained/evaluated.")
    for rare_name in ("collision", "offroad"):
        if float(label_stats.get(f"positive_rate_H20_{rare_name}", 0.0) or 0.0) <= 0.0:
            quality_warnings.append(f"{rare_name} support is zero at H20; report rare-event metrics as unsupported.")
    label_distribution["quality_warnings"] = quality_warnings
    label_path = export_json(label_distribution, os.path.join(output_dir, "fse_label_distribution.json"))
    source_path = export_json(fse_source_distribution(dataset, labels_binary, valid_mask), os.path.join(output_dir, "fse_source_distribution.json"))
    merge_report = {
        "raw_files": [os.path.abspath(p) for p in raw_paths],
        "num_samples_per_file": [int(raw_i["tokens"].shape[0]) for raw_i in raw_list],
        "schema_check_passed": True,
        "warning_count": 0,
    }
    merge_path = export_json(merge_report, os.path.join(output_dir, "fse_merge_report.json"))
    debug_path = os.path.join(output_dir, "fse_label_debug_sample.jsonl")
    with open(debug_path, "w", encoding="utf-8") as f:
        for sample in debug_samples:
            f.write(json.dumps(_jsonable_plain(sample), ensure_ascii=False) + "\n")
    return {
        "status": "ok",
        "fse_dataset_npz_path": os.path.abspath(dataset_path),
        "fse_dataset_meta_json_path": meta_path,
        "fse_label_distribution_json_path": label_path,
        "fse_source_distribution_json_path": source_path,
        "fse_split_json_path": split_path,
        "fse_merge_report_json_path": merge_path,
        "fse_label_debug_sample_jsonl_path": os.path.abspath(debug_path),
    }


def _z_artifact_requested(target: str, artifact_name: str) -> bool:
    target_norm = str(target).lower().strip()
    return target_norm == "both" or target_norm == str(artifact_name).lower().strip()


def _select_fse_z_artifact_indices(splits: Dict[str, np.ndarray], source_split: str) -> np.ndarray:
    split_norm = str(source_split).lower().strip().replace("-", "_").replace(" ", "_")
    if split_norm in FSE_FORBIDDEN_FORMAL_Z_SOURCE_SPLITS:
        raise ValueError(f"FSE z artifacts cannot be sourced from split='{source_split}'.")
    if split_norm == "train":
        selected = np.asarray(splits.get("train", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    elif split_norm == "val":
        selected = np.asarray(splits.get("val", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    elif split_norm == "train_val":
        selected = np.concatenate(
            [
                np.asarray(splits.get("train", np.zeros((0,), dtype=np.int64)), dtype=np.int64),
                np.asarray(splits.get("val", np.zeros((0,), dtype=np.int64)), dtype=np.int64),
            ],
            axis=0,
        )
    else:
        raise ValueError(f"Unsupported fse_z_artifact_source_split='{source_split}'. Choose from {FSE_Z_ARTIFACT_SOURCE_SPLITS}.")
    return np.unique(selected.astype(np.int64))


def _subsample_fse_z_artifact_indices(indices: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    cap = max(0, int(max_samples))
    if cap <= 0 or selected.shape[0] <= cap:
        return selected
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(selected, size=cap, replace=False).astype(np.int64)
    return np.sort(picked)


def _sha256_numpy_array(arr: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(value.dtype).encode("utf-8"))
    h.update(json.dumps(list(value.shape), sort_keys=True).encode("utf-8"))
    h.update(value.tobytes())
    return h.hexdigest()


def collect_fse_z_for_indices(
    model: FSEBottleneckTransformer,
    data: Dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    fse_cfg: FSEConfig,
) -> Dict[str, np.ndarray]:
    source_index = np.asarray(indices, dtype=np.int64).reshape(-1)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    z_chunks: List[np.ndarray] = []
    batch_size = max(1, int(getattr(fse_cfg, "batch_size", 128)))
    with torch.no_grad():
        for start in range(0, int(source_index.shape[0]), batch_size):
            idx = source_index[start: start + batch_size]
            batch = make_fse_batch(data, idx, device=device, action_conditioned=False)
            out = _fse_forward_from_batch(model, batch, action_conditioned=False)
            z = out.z_fse.detach().cpu().numpy().astype(np.float32)
            if z.ndim != 2 or z.shape[1] != int(fse_cfg.z_dim):
                raise ValueError(f"Collected z has shape {z.shape}; expected [B,{int(fse_cfg.z_dim)}].")
            if np.any(np.isnan(z)):
                raise ValueError("Collected FSE z contains NaN.")
            if np.any(np.isinf(z)):
                raise ValueError("Collected FSE z contains Inf.")
            z_chunks.append(z)
    if z_chunks:
        z_all = np.concatenate(z_chunks, axis=0).astype(np.float32)
    else:
        z_all = np.zeros((0, int(fse_cfg.z_dim)), dtype=np.float32)
    ep_key = "episode_uid_id" if "episode_uid_id" in data else "episode_id"
    return {
        "z": z_all,
        "source_index": source_index.astype(np.int64),
        "episode_id": np.asarray(data[ep_key][source_index], dtype=np.int64).reshape(-1),
        "step_id": np.asarray(data["step_id"][source_index], dtype=np.int64).reshape(-1),
    }


def _z_norm_summary(z: np.ndarray) -> Dict[str, float]:
    z_arr = np.asarray(z, dtype=np.float32)
    if z_arr.shape[0] <= 0:
        return {
            "z_norm_mean": float("nan"),
            "z_norm_std": float("nan"),
            "z_norm_p10": float("nan"),
            "z_norm_p50": float("nan"),
            "z_norm_p90": float("nan"),
            "z_batch_variance_mean": float("nan"),
        }
    norms = np.linalg.norm(z_arr, axis=1)
    return {
        "z_norm_mean": float(np.mean(norms)),
        "z_norm_std": float(np.std(norms)),
        "z_norm_p10": float(np.percentile(norms, 10)),
        "z_norm_p50": float(np.percentile(norms, 50)),
        "z_norm_p90": float(np.percentile(norms, 90)),
        "z_batch_variance_mean": float(np.var(z_arr, axis=0).mean()) if z_arr.shape[0] > 1 else 0.0,
    }


def _standard_z_artifact_fields(metadata: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "num_samples",
        "source_dataset",
        "source_split",
        "checkpoint_sha256",
        "fse_checkpoint_sha256",
        "checkpoint_hash",
        "scenario_builder_config_hash",
        "builder_hash",
        "traffic_density",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _write_fse_z_global_stats_json(z: np.ndarray, metadata: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    z_arr = np.asarray(z, dtype=np.float32)
    if z_arr.ndim != 2 or z_arr.shape[0] <= 0:
        raise ValueError("Cannot write FSE z stats from an empty z matrix.")
    mu_z = np.mean(z_arr, axis=0).astype(np.float32)
    std_z = np.std(z_arr, axis=0).astype(np.float32)
    if mu_z.shape != std_z.shape:
        raise ValueError("mu_z/std_z shape mismatch.")
    summary = {
        "raw_std_min": float(np.min(std_z)),
        "raw_std_max": float(np.max(std_z)),
        "raw_std_mean": float(np.mean(std_z)),
        "num_std_below_default_floor_1e-4": int(np.sum(std_z < 1e-4)),
        "recommended_std_floor": float(1e-4),
    }
    stats_metadata = copy.deepcopy(metadata)
    stats_metadata.update(summary)
    stats_metadata["artifact_type"] = "fse_z_global_stats"
    payload: Dict[str, Any] = {
        "artifact_type": "fse_z_global_stats",
        "artifact_version": str(stats_metadata.get("artifact_version", "fse_z_artifact_v1")),
        "mu_z": mu_z.tolist(),
        "std_z": std_z.tolist(),
        "std_z_is_raw_unclipped": True,
        "metadata": stats_metadata,
        **_standard_z_artifact_fields(stats_metadata),
        **summary,
    }
    path = export_json(payload, output_path)
    return {
        "path": path,
        "sha256": sha256_file(path),
        "metadata": stats_metadata,
        "mu_z_shape": list(mu_z.shape),
        "std_z_shape": list(std_z.shape),
        **summary,
    }


def _write_fse_shuffle_pool_npz(
    z_payload: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    output_path: str,
) -> Dict[str, Any]:
    z = np.asarray(z_payload["z"], dtype=np.float32)
    if z.ndim != 2 or z.shape[0] <= 0:
        raise ValueError("Cannot write FSE shuffle pool from an empty z matrix.")
    pool_metadata = copy.deepcopy(metadata)
    pool_metadata.update(_z_norm_summary(z))
    pool_metadata["artifact_type"] = "fse_shuffle_pool"
    metadata_json = json.dumps(_jsonable_plain(pool_metadata), sort_keys=True, ensure_ascii=True)
    arrays: Dict[str, Any] = {
        "z": z.astype(np.float32),
        "episode_id": np.asarray(z_payload["episode_id"], dtype=np.int64).reshape(-1),
        "step_id": np.asarray(z_payload["step_id"], dtype=np.int64).reshape(-1),
        "source_index": np.asarray(z_payload["source_index"], dtype=np.int64).reshape(-1),
        "metadata_json": np.asarray(metadata_json),
    }
    for key, value in pool_metadata.items():
        if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            arrays[str(key)] = np.asarray(value)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    abs_path = os.path.abspath(output_path)
    return {
        "path": abs_path,
        "sha256": sha256_file(abs_path),
        "metadata": pool_metadata,
        "z_shape": list(z.shape),
        "metadata_json_bytes": int(len(metadata_json.encode("utf-8"))),
        **{k: pool_metadata[k] for k in ("z_norm_mean", "z_norm_std", "z_norm_p10", "z_norm_p50", "z_norm_p90", "z_batch_variance_mean")},
    }


def _assert_z_artifact_metadata_matches(actual: Dict[str, Any], expected: Dict[str, Any], artifact_name: str) -> None:
    checks = ("source_split", "checkpoint_sha256", "scenario_builder_config_hash", "traffic_density")
    for key in checks:
        lhs = str(actual.get(key, "")).strip()
        rhs = str(expected.get(key, "")).strip()
        if key == "traffic_density":
            lhs = lhs.lower()
            rhs = rhs.lower()
        if lhs != rhs:
            raise ValueError(f"{artifact_name} self-check metadata mismatch for {key}: got '{lhs}', expected '{rhs}'.")
    actual_n = int(float(actual.get("num_samples", -1)))
    expected_n = int(float(expected.get("num_samples", -2)))
    if actual_n != expected_n:
        raise ValueError(f"{artifact_name} self-check num_samples mismatch: got {actual_n}, expected {expected_n}.")
    actual_dataset = str(actual.get("source_dataset_abs_path", actual.get("source_dataset", ""))).strip()
    expected_dataset = str(expected.get("source_dataset_abs_path", expected.get("source_dataset", ""))).strip()
    if actual_dataset and expected_dataset and os.path.abspath(actual_dataset) != os.path.abspath(expected_dataset):
        raise ValueError(f"{artifact_name} self-check source_dataset mismatch.")


def validate_built_z_artifacts(
    cfg: Config,
    stats_path: str,
    pool_path: str,
    expected_metadata: Dict[str, Any],
    check_stats: bool,
    check_pool: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "random_formal": {"status": "skipped"},
        "shuffled_formal": {"status": "skipped"},
    }
    if check_stats:
        if not str(stats_path).strip() or not os.path.exists(stats_path):
            raise FileNotFoundError(f"Cannot self-check random formal stats; missing file: {stats_path}")
        random_cfg = copy.deepcopy(cfg)
        setattr(random_cfg.fse, "_user_explicit_fusion_mode", False)
        setattr(random_cfg.fse, "_user_explicit_z_mode", False)
        configure_mode_behavior(random_cfg, "paper_csac_fse_z_gated_random")
        random_cfg.fse.run_tier = "formal"
        random_cfg.fse.random_z_distribution = "global_empirical_matched"
        random_cfg.fse.z_global_stats_path = os.path.abspath(stats_path)
        random_cfg.fse.shuffle_pool_path = ""
        random_cfg.fse.formal_min_z_artifact_samples = max(1, int(getattr(cfg.fse, "z_artifact_min_samples", 32)))
        runtime = FrozenFSEBottleneckRuntime(random_cfg, raw_state_dim=int(getattr(cfg.fse, "raw_state_dim", 0)), output_dir=str(cfg.fse.output_dir))
        expected_hash = sha256_file(stats_path)
        if str(runtime.global_stats_hash) != str(expected_hash):
            raise ValueError("Random formal self-check global stats hash mismatch.")
        _assert_z_artifact_metadata_matches(runtime.global_stats_metadata, expected_metadata, "global z stats")
        result["random_formal"] = {
            "status": "ok",
            "global_stats_sha256": runtime.global_stats_hash,
            "metadata": _jsonable_plain(runtime.global_stats_metadata),
        }
    if check_pool:
        if not str(pool_path).strip() or not os.path.exists(pool_path):
            raise FileNotFoundError(f"Cannot self-check shuffled formal pool; missing file: {pool_path}")
        shuffled_cfg = copy.deepcopy(cfg)
        setattr(shuffled_cfg.fse, "_user_explicit_fusion_mode", False)
        setattr(shuffled_cfg.fse, "_user_explicit_z_mode", False)
        configure_mode_behavior(shuffled_cfg, "paper_csac_fse_z_gated_shuffled")
        shuffled_cfg.fse.run_tier = "formal"
        shuffled_cfg.fse.random_z_distribution = "global_empirical_matched"
        shuffled_cfg.fse.z_global_stats_path = os.path.abspath(stats_path) if str(stats_path).strip() and os.path.exists(stats_path) else ""
        shuffled_cfg.fse.shuffle_pool_path = os.path.abspath(pool_path)
        shuffled_cfg.fse.formal_min_z_artifact_samples = max(1, int(getattr(cfg.fse, "z_artifact_min_samples", 32)))
        runtime = FrozenFSEBottleneckRuntime(shuffled_cfg, raw_state_dim=int(getattr(cfg.fse, "raw_state_dim", 0)), output_dir=str(cfg.fse.output_dir))
        expected_hash = sha256_file(pool_path)
        if str(runtime.shuffle_pool_hash) != str(expected_hash):
            raise ValueError("Shuffled formal self-check shuffle pool hash mismatch.")
        _assert_z_artifact_metadata_matches(runtime.shuffle_pool_metadata, expected_metadata, "shuffle pool")
        payload = {
            "status": "ok",
            "shuffle_pool_sha256": runtime.shuffle_pool_hash,
            "metadata": _jsonable_plain(runtime.shuffle_pool_metadata),
        }
        if str(stats_path).strip() and os.path.exists(stats_path):
            payload["global_stats_sha256"] = runtime.global_stats_hash
        result["shuffled_formal"] = payload
    return result


def run_fse_build_z_artifacts(cfg: Config) -> Dict[str, Any]:
    output_dir = str(cfg.fse.output_dir).strip() or "results/fse"
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = str(cfg.fse.dataset_path).strip()
    checkpoint_path = str(cfg.fse.checkpoint_path).strip()
    if not dataset_path:
        raise ValueError("--fse-dataset-path is required for --fse-task build-z-artifacts.")
    if not checkpoint_path:
        raise ValueError("--fse-checkpoint-path is required for --fse-task build-z-artifacts.")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"FSE dataset not found: {dataset_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"FSE checkpoint not found: {checkpoint_path}")

    target = str(cfg.fse.z_artifacts).lower().strip()
    source_split = str(cfg.fse.z_artifact_source_split).lower().strip().replace("-", "_").replace(" ", "_")
    if target not in FSE_Z_ARTIFACT_TARGETS:
        raise ValueError(f"Unsupported fse_z_artifacts='{target}'. Choose from {FSE_Z_ARTIFACT_TARGETS}.")
    if source_split not in FSE_Z_ARTIFACT_SOURCE_SPLITS:
        raise ValueError(f"Unsupported fse_z_artifact_source_split='{source_split}'. Choose from {FSE_Z_ARTIFACT_SOURCE_SPLITS}.")
    if source_split in FSE_FORBIDDEN_FORMAL_Z_SOURCE_SPLITS:
        raise ValueError(f"FSE z artifacts cannot be sourced from split='{source_split}'.")

    stats_path = str(cfg.fse.z_stats_output_path).strip() or os.path.join(output_dir, "fse_z_global_stats.json")
    pool_path = str(cfg.fse.shuffle_pool_output_path).strip() or os.path.join(output_dir, "fse_shuffle_pool.npz")
    warnings: List[str] = []
    if str(cfg.fse.run_tier).lower().strip() == "formal" and source_split != "train_val":
        warnings.append("formal_non_train_val_source_split_warning")

    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg.fse.z_mode = "real"
    runtime_cfg.fse.fusion_mode = "concat"
    runtime = FrozenFSEBottleneckRuntime(runtime_cfg, raw_state_dim=0, output_dir=output_dir)
    fse_cfg = copy.deepcopy(runtime.fse_cfg)
    fse_cfg.dataset_path = dataset_path
    data, load_warnings = load_fse_npz_dataset(dataset_path, fse_cfg)
    splits = split_fse_by_episode(data, seed=int(cfg.train.seed), fse_cfg=fse_cfg)
    pre_subsample_indices = _select_fse_z_artifact_indices(splits, source_split)
    selected_indices = _subsample_fse_z_artifact_indices(
        pre_subsample_indices,
        max_samples=int(getattr(cfg.fse, "z_artifact_max_samples", 0)),
        seed=int(cfg.train.seed),
    )
    min_samples = max(1, int(getattr(cfg.fse, "z_artifact_min_samples", 32)))
    if selected_indices.shape[0] < min_samples:
        raise ValueError(
            f"Selected z artifact sample count {int(selected_indices.shape[0])} is below minimum {min_samples} "
            f"for source_split='{source_split}'."
        )

    z_payload = collect_fse_z_for_indices(runtime.model, data, selected_indices, device=runtime.device, fse_cfg=fse_cfg)
    z = np.asarray(z_payload["z"], dtype=np.float32)
    if z.shape[0] < min_samples:
        raise ValueError(f"Collected z sample count {int(z.shape[0])} is below minimum {min_samples}.")
    if z.shape[1] != 64:
        raise ValueError(f"Formal FSE z artifacts require z_dim=64; got {z.shape[1]}.")

    episode_ids = np.asarray(z_payload["episode_id"], dtype=np.int64)
    step_ids = np.asarray(z_payload["step_id"], dtype=np.int64)
    source_indices = np.asarray(z_payload["source_index"], dtype=np.int64)
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    dataset_abs = os.path.abspath(dataset_path)
    checkpoint_abs = os.path.abspath(checkpoint_path)
    selected_summary = {
        "selected_index_count": int(source_indices.shape[0]),
        "selected_episode_count": int(np.unique(episode_ids).shape[0]),
        "selected_step_min": int(np.min(step_ids)),
        "selected_step_max": int(np.max(step_ids)),
        "source_index_sha256": _sha256_numpy_array(source_indices),
        "episode_id_sha256": _sha256_numpy_array(episode_ids),
    }
    base_metadata: Dict[str, Any] = {
        "artifact_version": "fse_z_artifact_v1",
        "num_samples": int(z.shape[0]),
        "source_dataset": dataset_abs,
        "source_dataset_abs_path": dataset_abs,
        "source_dataset_sha256": sha256_file(dataset_abs),
        "source_dataset_exists_at_build": bool(os.path.exists(dataset_abs)),
        "source_split": source_split,
        "checkpoint_path_abs": checkpoint_abs,
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "fse_checkpoint_sha256": runtime.checkpoint_sha256,
        "checkpoint_hash": runtime.checkpoint_sha256,
        "scenario_builder_config_hash": runtime.scenario_builder_config_hash,
        "builder_hash": runtime.scenario_builder_config_hash,
        "traffic_density": str(cfg.env.traffic_density),
        "scenario_frame_schema_version": SCENARIO_SCHEMA_VERSION,
        "z_dim": int(runtime.z_dim),
        "token_dim": int(SCENARIO_TOKEN_DIM),
        "n_tokens": int(SCENARIO_TOKEN_COUNT),
        "horizons": [int(v) for v in runtime.checkpoint.get("horizons", [])],
        "risk_names": list(runtime.checkpoint.get("risk_names", [])),
        "action_conditioned": bool(runtime.checkpoint.get("action_conditioned", False)),
        "seed": int(cfg.train.seed),
        "split_seed": int(cfg.train.seed),
        "split_stratify_by": str(getattr(fse_cfg, "split_stratify_by", "")),
        "generated_at_utc": generated_at_utc,
        "fse_model_config_hash": runtime.model_config_hash,
        "normalization_config_legacy_warning": bool(runtime.normalization_legacy_warning),
        "warnings": list(warnings),
        **selected_summary,
    }

    stats_result: Dict[str, Any] = {"status": "skipped"}
    pool_result: Dict[str, Any] = {"status": "skipped"}
    check_stats = _z_artifact_requested(target, "stats")
    check_pool = _z_artifact_requested(target, "pool")
    written_stats_path = ""
    written_pool_path = ""
    if check_stats:
        stats_result = _write_fse_z_global_stats_json(z, base_metadata, stats_path)
        written_stats_path = str(stats_result["path"])
    if check_pool:
        pool_result = _write_fse_shuffle_pool_npz(z_payload, base_metadata, pool_path)
        written_pool_path = str(pool_result["path"])

    stats_for_self_check = written_stats_path
    if not stats_for_self_check and os.path.exists(stats_path):
        stats_for_self_check = os.path.abspath(stats_path)
        warnings.append("using_existing_z_stats_for_pool_self_check")
    self_check = validate_built_z_artifacts(
        cfg,
        stats_path=stats_for_self_check,
        pool_path=written_pool_path,
        expected_metadata=base_metadata,
        check_stats=bool(check_stats),
        check_pool=bool(check_pool),
    )

    ep_key = "episode_uid_id" if "episode_uid_id" in data else "episode_id"
    result_path = os.path.join(output_dir, "build_z_artifacts_result.json")
    result: Dict[str, Any] = {
        "status": "ok",
        "task": "build-z-artifacts",
        "z_artifacts": target,
        "source_split": source_split,
        "warnings": list(warnings),
        "stats": stats_result,
        "pool": pool_result,
        "stats_path": written_stats_path,
        "stats_sha256": sha256_file(written_stats_path),
        "pool_path": written_pool_path,
        "pool_sha256": sha256_file(written_pool_path),
        "sample_count": int(z.shape[0]),
        "pre_subsample_index_count": int(pre_subsample_indices.shape[0]),
        "max_samples": int(getattr(cfg.fse, "z_artifact_max_samples", 0)),
        **selected_summary,
        "dataset_path": dataset_abs,
        "dataset_sha256": sha256_file(dataset_abs),
        "checkpoint_path": checkpoint_abs,
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "scenario_builder_config_hash": runtime.scenario_builder_config_hash,
        "traffic_density": str(cfg.env.traffic_density),
        "split_sizes": {name: int(np.asarray(idx).shape[0]) for name, idx in splits.items()},
        "split_episode_counts": {
            name: int(np.unique(data[ep_key][np.asarray(idx, dtype=np.int64)]).shape[0])
            for name, idx in splits.items()
        },
        "dataset_load_warnings": load_warnings,
        "artifact_metadata": base_metadata,
        "self_check": self_check,
        "build_z_artifacts_result_json_path": os.path.abspath(result_path),
    }
    export_json(result, result_path)
    return result


def run_fse_full(cfg: Config, modes: Optional[List[str]] = None, seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    if "," in str(cfg.fse.raw_path) or bool(getattr(cfg.fse, "source_balanced_sampler", False)):
        raise ValueError("--fse-task full is reserved for single-source smoke/simple runs; formal three-source CSAC runs must be staged.")
    root = str(cfg.fse.output_dir).strip() or "results/fse/full"
    collect_cfg = copy.deepcopy(cfg)
    collect_cfg.fse.output_dir = os.path.join(root, "collect")
    collect_result = run_fse_collect(collect_cfg, modes=modes, seeds=seeds)
    build_cfg = copy.deepcopy(cfg)
    build_cfg.fse.raw_path = collect_result["fse_raw_trajectories_npz_path"]
    build_cfg.fse.output_dir = os.path.join(root, "dataset")
    build_result = run_fse_build_dataset(build_cfg)
    train_cfg = copy.deepcopy(cfg)
    train_cfg.fse.dataset_path = build_result["fse_dataset_npz_path"]
    train_cfg.fse.output_dir = os.path.join(root, "train")
    train_result = run_fse_train(train_cfg)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.fse.dataset_path = build_result["fse_dataset_npz_path"]
    eval_cfg.fse.checkpoint_path = train_result["fse_checkpoint_path"]
    eval_cfg.fse.output_dir = os.path.join(root, "eval")
    eval_result = run_fse_eval(eval_cfg)
    return {"status": "ok", "collect": collect_result, "build_dataset": build_result, "train": train_result, "eval": eval_result}


def run_fse_task(cfg: Config, modes: Optional[List[str]] = None, seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    task = str(cfg.fse.task).lower().strip()
    if task == "none":
        return {}
    if task == "collect":
        return run_fse_collect(cfg, modes=modes, seeds=seeds)
    if task == "build-dataset":
        return run_fse_build_dataset(cfg)
    if task == "build-z-artifacts":
        return run_fse_build_z_artifacts(cfg)
    if task == "smoke":
        return run_fse_smoke(cfg)
    if task == "train":
        return run_fse_train(cfg)
    if task == "eval":
        return run_fse_eval(cfg)
    if task == "full":
        return run_fse_full(cfg, modes=modes, seeds=seeds)
    raise ValueError(f"Unsupported FSE task: {task}. Choose from {FSE_TASKS}.")


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
        self._reset_counter = 0

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is None:
            episode_seed = int(stable_int_hash(self.cfg.train.seed, "env_reset_fallback", self._reset_counter) % (2**31 - 1))
            self._reset_counter += 1
        else:
            episode_seed = int(seed)
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
            return self._episode_rng.uniform(self.action_low, self.action_high).astype(np.float32)

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
        role_vehicle_ids = {
            "front_same": self._vehicle_identity(front_same),
            "rear_same": self._vehicle_identity(rear_same),
            "left_front": self._vehicle_identity(front_left),
            "left_rear": self._vehicle_identity(rear_left),
            "right_front": self._vehicle_identity(front_right),
            "right_rear": self._vehicle_identity(rear_right),
        }

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
            "role_vehicle_ids": dict(role_vehicle_ids),
            "front_same_vehicle_id": role_vehicle_ids["front_same"],
            "rear_same_vehicle_id": role_vehicle_ids["rear_same"],
            "left_front_vehicle_id": role_vehicle_ids["left_front"],
            "left_rear_vehicle_id": role_vehicle_ids["left_rear"],
            "right_front_vehicle_id": role_vehicle_ids["right_front"],
            "right_rear_vehicle_id": role_vehicle_ids["right_rear"],
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

    def _vehicle_identity(self, vehicle) -> Optional[str]:
        if vehicle is None:
            return None
        for attr in ("uid", "id", "track_id"):
            value = getattr(vehicle, attr, None)
            if value is not None:
                return str(value)
        return f"episode_object:{id(vehicle)}"

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
    reward_type = str(getattr(cfg.reward, "reward_type", "paper_formula")).lower().strip()
    if reward_type != "paper_formula":
        raise ValueError(
            f"Unsupported reward_type='{reward_type}' in paper-only script. "
            "Use mode in {'paper_sac_pure','paper_csac_lagrangian'}."
        )
    return compute_paper_formula_reward(cfg, scene, next_scene, action, env_reward, timeout_failure=timeout_failure)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device,
        raw_state_dim: int = 0,
        z_dim: int = 0,
        replay_seed: Optional[int] = None,
        store_fse_frame_tokens: bool = False,
    ):
        self.capacity = int(capacity)
        self.device = device
        self.state_dim = int(state_dim)
        self.raw_state_dim = int(raw_state_dim)
        self.z_dim = int(z_dim)
        self.fse_enabled = bool(self.raw_state_dim > 0 and self.z_dim > 0)
        self.store_fse_frame_tokens = bool(store_fse_frame_tokens)
        if self.store_fse_frame_tokens and not self.fse_enabled:
            raise ValueError("ReplayBuffer can store FSE frame tokens only when FSE-RL is enabled.")
        seed_value = int(0 if replay_seed is None else replay_seed) % (2**32)
        self.rng = np.random.default_rng(seed_value)
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
        self.raw_state = np.zeros((capacity, self.raw_state_dim), dtype=np.float32) if self.fse_enabled else None
        self.next_raw_state = np.zeros((capacity, self.raw_state_dim), dtype=np.float32) if self.fse_enabled else None
        self.z_real = np.full((capacity, self.z_dim), np.nan, dtype=np.float32) if self.fse_enabled else None
        self.next_z_real = np.full((capacity, self.z_dim), np.nan, dtype=np.float32) if self.fse_enabled else None
        self.z_used = np.zeros((capacity, self.z_dim), dtype=np.float32) if self.fse_enabled else None
        self.next_z_used = np.zeros((capacity, self.z_dim), dtype=np.float32) if self.fse_enabled else None
        self.z_real_valid = np.zeros((capacity, 1), dtype=np.float32) if self.fse_enabled else None
        self.next_z_real_valid = np.zeros((capacity, 1), dtype=np.float32) if self.fse_enabled else None
        self.z_used_valid = np.zeros((capacity, 1), dtype=np.float32) if self.fse_enabled else None
        self.next_z_used_valid = np.zeros((capacity, 1), dtype=np.float32) if self.fse_enabled else None
        self.fse_forward_executed = np.zeros((capacity, 1), dtype=np.float32) if self.fse_enabled else None
        self.z_source = np.full((capacity,), "", dtype="<U48") if self.fse_enabled else None
        self.next_z_source = np.full((capacity,), "", dtype="<U48") if self.fse_enabled else None
        self.shuffled_source_episode_id = np.full((capacity,), -1, dtype=np.int64) if self.fse_enabled else None
        self.shuffled_source_step_id = np.full((capacity,), -1, dtype=np.int64) if self.fse_enabled else None
        self.shuffled_fallback_flag = np.zeros((capacity, 1), dtype=np.float32) if self.fse_enabled else None
        self.frame_tokens = np.zeros((capacity, SCENARIO_TOKEN_COUNT, SCENARIO_TOKEN_DIM), dtype=np.float32) if self.store_fse_frame_tokens else None
        self.frame_token_mask = np.zeros((capacity, SCENARIO_TOKEN_COUNT), dtype=np.float32) if self.store_fse_frame_tokens else None
        self.frame_entity_valid_mask = np.zeros((capacity, SCENARIO_TOKEN_COUNT), dtype=np.float32) if self.store_fse_frame_tokens else None
        self.frame_token_type_ids = np.zeros((capacity, SCENARIO_TOKEN_COUNT), dtype=np.int64) if self.store_fse_frame_tokens else None
        self.frame_token_role_ids = np.zeros((capacity, SCENARIO_TOKEN_COUNT), dtype=np.int64) if self.store_fse_frame_tokens else None
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
        if np.any(np.isnan(self.state[self.ptr])) or np.any(np.isinf(self.state[self.ptr])):
            raise ValueError("state_aug contains NaN/Inf before replay insertion.")
        if np.any(np.isnan(self.next_state[self.ptr])) or np.any(np.isinf(self.next_state[self.ptr])):
            raise ValueError("next_state_aug contains NaN/Inf before replay insertion.")
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
        if self.fse_enabled:
            if bool(transition_meta.get("z_used_valid", False)) is not True or bool(transition_meta.get("next_z_used_valid", False)) is not True:
                raise ValueError("FSE-RL replay insertion requires z_used_valid and next_z_used_valid.")
            assert self.raw_state is not None and self.next_raw_state is not None
            assert self.z_real is not None and self.next_z_real is not None and self.z_used is not None and self.next_z_used is not None
            self.raw_state[self.ptr] = np.asarray(transition_meta.get("raw_state"), dtype=np.float32).reshape(self.raw_state_dim)
            self.next_raw_state[self.ptr] = np.asarray(transition_meta.get("next_raw_state"), dtype=np.float32).reshape(self.raw_state_dim)
            self.z_real[self.ptr] = np.asarray(transition_meta.get("z_real"), dtype=np.float32).reshape(self.z_dim)
            self.next_z_real[self.ptr] = np.asarray(transition_meta.get("next_z_real"), dtype=np.float32).reshape(self.z_dim)
            self.z_used[self.ptr] = np.asarray(transition_meta.get("z_used"), dtype=np.float32).reshape(self.z_dim)
            self.next_z_used[self.ptr] = np.asarray(transition_meta.get("next_z_used"), dtype=np.float32).reshape(self.z_dim)
            self.z_real_valid[self.ptr] = float(bool(transition_meta.get("z_real_valid", False)))
            self.next_z_real_valid[self.ptr] = float(bool(transition_meta.get("next_z_real_valid", False)))
            self.z_used_valid[self.ptr] = float(bool(transition_meta.get("z_used_valid", False)))
            self.next_z_used_valid[self.ptr] = float(bool(transition_meta.get("next_z_used_valid", False)))
            self.fse_forward_executed[self.ptr] = float(bool(transition_meta.get("fse_forward_executed", False)))
            self.z_source[self.ptr] = str(transition_meta.get("z_source", ""))[:47]
            self.next_z_source[self.ptr] = str(transition_meta.get("next_z_source", ""))[:47]
            self.shuffled_source_episode_id[self.ptr] = int(transition_meta.get("shuffled_source_episode_id", -1))
            self.shuffled_source_step_id[self.ptr] = int(transition_meta.get("shuffled_source_step_id", -1))
            self.shuffled_fallback_flag[self.ptr] = float(bool(transition_meta.get("shuffled_fallback_flag", False)))
            if self.store_fse_frame_tokens:
                assert self.frame_tokens is not None and self.frame_token_mask is not None
                assert self.frame_entity_valid_mask is not None and self.frame_token_type_ids is not None and self.frame_token_role_ids is not None
                for key in ("frame_tokens", "frame_token_mask", "frame_entity_valid_mask", "frame_token_type_ids", "frame_token_role_ids"):
                    if key not in transition_meta:
                        raise ValueError(f"Action-risk replay insertion requires {key}.")
                self.frame_tokens[self.ptr] = np.asarray(transition_meta["frame_tokens"], dtype=np.float32).reshape(SCENARIO_TOKEN_COUNT, SCENARIO_TOKEN_DIM)
                self.frame_token_mask[self.ptr] = np.asarray(transition_meta["frame_token_mask"], dtype=np.float32).reshape(SCENARIO_TOKEN_COUNT)
                self.frame_entity_valid_mask[self.ptr] = np.asarray(transition_meta["frame_entity_valid_mask"], dtype=np.float32).reshape(SCENARIO_TOKEN_COUNT)
                self.frame_token_type_ids[self.ptr] = np.asarray(transition_meta["frame_token_type_ids"], dtype=np.int64).reshape(SCENARIO_TOKEN_COUNT)
                self.frame_token_role_ids[self.ptr] = np.asarray(transition_meta["frame_token_role_ids"], dtype=np.int64).reshape(SCENARIO_TOKEN_COUNT)
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
        arr = np.asarray(sorted(pool), dtype=np.int64)
        idx = self.rng.choice(arr, size=int(n), replace=bool(len(arr) < int(n)))
        return [int(v) for v in idx.tolist()]

    def _sample_indices(self, batch_size: int, priority_cfg: Optional[dict]) -> Tuple[np.ndarray, Dict[str, float]]:
        valid = np.asarray(sorted(self.active_indices), dtype=np.int64)
        if valid.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        near_pool = self.label_index_sets[1]
        danger_pool = self.label_index_sets[2]
        if not priority_cfg or not bool(priority_cfg.get("enabled", False)):
            idx = self.rng.choice(valid, size=int(batch_size), replace=bool(valid.size < int(batch_size)))
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
            sampled = [int(v) for v in self.rng.choice(valid, size=int(batch_size), replace=True).tolist()]
        idx = np.asarray(sampled, dtype=np.int64)
        if idx.size > int(batch_size):
            idx = idx[: int(batch_size)]
        self.rng.shuffle(idx)
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
        batch = {
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
        if self.fse_enabled:
            batch.update({
                "raw_state": torch.as_tensor(self.raw_state[idx], device=self.device),
                "next_raw_state": torch.as_tensor(self.next_raw_state[idx], device=self.device),
                "z_real": torch.as_tensor(self.z_real[idx], device=self.device),
                "next_z_real": torch.as_tensor(self.next_z_real[idx], device=self.device),
                "z_used": torch.as_tensor(self.z_used[idx], device=self.device),
                "next_z_used": torch.as_tensor(self.next_z_used[idx], device=self.device),
                "z_real_valid": torch.as_tensor(self.z_real_valid[idx], device=self.device),
                "next_z_real_valid": torch.as_tensor(self.next_z_real_valid[idx], device=self.device),
                "z_used_valid": torch.as_tensor(self.z_used_valid[idx], device=self.device),
                "next_z_used_valid": torch.as_tensor(self.next_z_used_valid[idx], device=self.device),
            })
        if self.store_fse_frame_tokens:
            batch.update({
                "frame_tokens": torch.as_tensor(self.frame_tokens[idx], dtype=torch.float32, device=self.device),
                "frame_token_mask": torch.as_tensor(self.frame_token_mask[idx], dtype=torch.float32, device=self.device),
                "frame_entity_valid_mask": torch.as_tensor(self.frame_entity_valid_mask[idx], dtype=torch.float32, device=self.device),
                "frame_token_type_ids": torch.as_tensor(self.frame_token_type_ids[idx], dtype=torch.long, device=self.device),
                "frame_token_role_ids": torch.as_tensor(self.frame_token_role_ids[idx], dtype=torch.long, device=self.device),
            })
        return batch


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

    def _sample_full(self, state):
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
        return action, log_prob, mu_action, unit_action, torch.tanh(mu)

    def sample(self, state):
        action, log_prob, mu_action, _, _ = self._sample_full(state)
        return action, log_prob, mu_action

    def sample_with_normalized(self, state):
        return self._sample_full(state)


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


class FSEFusionEncoder(nn.Module):
    def __init__(self, state_aug_dim: int, raw_state_dim: int, z_dim: int, hidden_dim: int, fusion_mode: str):
        super().__init__()
        self.state_aug_dim = int(state_aug_dim)
        self.raw_state_dim = int(raw_state_dim)
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.fusion_mode = str(fusion_mode).lower().strip()
        self.last_gate_mean = float("nan")
        self.last_gate_std = float("nan")
        if self.raw_state_dim <= 0 or self.z_dim <= 0 or self.state_aug_dim != self.raw_state_dim + self.z_dim:
            raise ValueError("FSEFusionEncoder requires state_aug_dim = raw_state_dim + z_dim.")
        if self.fusion_mode == "concat":
            self.concat = nn.Sequential(
                nn.LayerNorm(self.state_aug_dim),
                nn.Linear(self.state_aug_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
        elif self.fusion_mode == "gated":
            self.state_mlp = nn.Sequential(
                nn.Linear(self.raw_state_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.z_mlp = nn.Sequential(
                nn.Linear(self.z_dim, self.hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid(),
            )
            self.out_norm = nn.LayerNorm(self.hidden_dim)
        else:
            raise ValueError(f"Unsupported fse fusion mode: {self.fusion_mode}")

    def forward(self, state_aug: torch.Tensor) -> torch.Tensor:
        if state_aug.shape[-1] != self.state_aug_dim:
            raise ValueError(f"state_aug last dim must be {self.state_aug_dim}, got {state_aug.shape[-1]}.")
        if self.fusion_mode == "concat":
            self.last_gate_mean = float("nan")
            self.last_gate_std = float("nan")
            return self.concat(state_aug)
        raw_state = state_aug[..., : self.raw_state_dim]
        z_used = state_aug[..., self.raw_state_dim :]
        s_emb = self.state_mlp(raw_state)
        z_emb = self.z_mlp(z_used)
        gate = self.gate_mlp(torch.cat([s_emb, z_emb], dim=-1))
        with torch.no_grad():
            self.last_gate_mean = float(gate.mean().detach().cpu().item())
            self.last_gate_std = float(gate.std(unbiased=False).detach().cpu().item())
        return self.out_norm(s_emb + gate * z_emb)


class FSEActor(nn.Module):
    def __init__(self, state_aug_dim: int, raw_state_dim: int, z_dim: int, action_dim: int, hidden_dim: int, action_low, action_high, fusion_mode: str):
        super().__init__()
        self.encoder = FSEFusionEncoder(state_aug_dim, raw_state_dim, z_dim, hidden_dim, fusion_mode)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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

    def forward(self, state_aug: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.encoder(state_aug))
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def _sample_full(self, state_aug: torch.Tensor):
        mu, log_std = self(state_aug)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        unit_action = torch.tanh(z)
        action = self.action_bias + self.action_scale * unit_action
        log_prob = dist.log_prob(z) - torch.log(1.0 - unit_action.pow(2) + 1e-6)
        log_prob = log_prob - torch.log(self.action_scale + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mu_action = self.action_bias + self.action_scale * torch.tanh(mu)
        return action, log_prob, mu_action, unit_action, torch.tanh(mu)

    def sample(self, state_aug: torch.Tensor):
        action, log_prob, mu_action, _, _ = self._sample_full(state_aug)
        return action, log_prob, mu_action

    def sample_with_normalized(self, state_aug: torch.Tensor):
        return self._sample_full(state_aug)


class FSECritic(nn.Module):
    def __init__(self, state_aug_dim: int, raw_state_dim: int, z_dim: int, action_dim: int, hidden_dim: int, fusion_mode: str):
        super().__init__()
        self.encoder = FSEFusionEncoder(state_aug_dim, raw_state_dim, z_dim, hidden_dim, fusion_mode)
        self.q = MLP(hidden_dim + action_dim, 1, hidden_dim)

    def forward(self, state_aug: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        fused = self.encoder(state_aug)
        return self.q(torch.cat([fused, action], dim=-1))


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
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.gamma = float(cfg.sac.gamma)
        self.tau = float(cfg.sac.tau)
        self.batch_size = int(cfg.sac.batch_size)
        self.target_entropy = float(cfg.sac.target_entropy)
        self.update_counter = 0
        self.action_fse_runtime: Optional[FrozenActionFSEPenaltyRuntime] = None
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
        fse_raw_state_dim = int(getattr(cfg.fse, "raw_state_dim", 0)) if is_fse_rl_mode(cfg.train.mode) else 0
        fse_z_dim = int(getattr(cfg.fse, "z_dim", 0)) if is_fse_rl_mode(cfg.train.mode) else 0
        replay_seed = stable_int_hash(cfg.train.seed, cfg.train.mode, "replay_rng") % (2**32)
        self.replay = ReplayBuffer(
            state_dim,
            action_dim,
            cfg.sac.buffer_size,
            self.device,
            raw_state_dim=fse_raw_state_dim,
            z_dim=fse_z_dim,
            replay_seed=replay_seed,
            store_fse_frame_tokens=bool(is_fse_action_risk_mode(cfg.train.mode) and fse_raw_state_dim > 0 and fse_z_dim > 0),
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def reset_episode_context(self) -> None:
        return None

    def attach_action_fse_runtime(self, runtime: Optional[FrozenActionFSEPenaltyRuntime]) -> None:
        self.action_fse_runtime = runtime

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
        self.update_counter += 1
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
        self.fse_rl_enabled = bool(is_fse_rl_mode(cfg.train.mode))
        if self.fse_rl_enabled:
            raw_state_dim = int(getattr(cfg.fse, "raw_state_dim", 0))
            z_dim = int(getattr(cfg.fse, "z_dim", 0))
            if raw_state_dim <= 0 or z_dim <= 0 or int(state_dim) != raw_state_dim + z_dim:
                raise ValueError("FSE-RL agent requires state_dim = raw_state_dim + z_dim.")
            self.actor = FSEActor(
                state_dim,
                raw_state_dim,
                z_dim,
                action_dim,
                cfg.sac.hidden_dim,
                action_low,
                action_high,
                cfg.fse.fusion_mode,
            ).to(self.device)
            self.q1 = FSECritic(state_dim, raw_state_dim, z_dim, action_dim, cfg.sac.hidden_dim, cfg.fse.fusion_mode).to(self.device)
            self.q2 = FSECritic(state_dim, raw_state_dim, z_dim, action_dim, cfg.sac.hidden_dim, cfg.fse.fusion_mode).to(self.device)
            self.q1_target = FSECritic(state_dim, raw_state_dim, z_dim, action_dim, cfg.sac.hidden_dim, cfg.fse.fusion_mode).to(self.device)
            self.q2_target = FSECritic(state_dim, raw_state_dim, z_dim, action_dim, cfg.sac.hidden_dim, cfg.fse.fusion_mode).to(self.device)
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.sac.actor_lr)
            self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.sac.critic_lr)
            self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.sac.critic_lr)
        def make_cost_critic() -> nn.Module:
            if self.fse_rl_enabled:
                return FSECritic(
                    state_dim,
                    int(cfg.fse.raw_state_dim),
                    int(cfg.fse.z_dim),
                    action_dim,
                    cfg.sac.hidden_dim,
                    cfg.fse.fusion_mode,
                )
            return Critic(state_dim, action_dim, cfg.sac.hidden_dim)
        self.cost_critics = nn.ModuleDict({
            "safety": make_cost_critic(),
            "boundary": make_cost_critic(),
            "speed": make_cost_critic(),
        }).to(self.device)
        self.cost_targets = nn.ModuleDict({
            name: make_cost_critic()
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

    @staticmethod
    def _module_gate_stats(module: nn.Module) -> Tuple[float, float]:
        enc = getattr(module, "encoder", None)
        if enc is None:
            return float("nan"), float("nan")
        return float(getattr(enc, "last_gate_mean", float("nan"))), float(getattr(enc, "last_gate_std", float("nan")))

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
        self.update_counter += 1
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

        action_risk_enabled = bool(is_fse_action_risk_mode(self.cfg.train.mode))
        if action_risk_enabled:
            if self.action_fse_runtime is None:
                raise ValueError("Action-risk mode requires an attached FrozenActionFSEPenaltyRuntime.")
            new_action_env, log_prob, _, new_action_norm, _ = self.actor.sample_with_normalized(state)
            action_for_q = new_action_env
            action_for_env = new_action_env
            action_for_fse = new_action_norm
        else:
            new_action_env, log_prob, _ = self.actor.sample(state)
            action_for_q = new_action_env
            action_for_env = new_action_env
            action_for_fse = None
        q_new = torch.min(self.q1(state, action_for_q), self.q2(state, action_for_q))
        cost_q = {
            name: self._average_cost_q(self.cost_critics[name](state, action_for_q))
            for name in self.cost_critics.keys()
        }
        lagrangian_penalty = sum(self.lambdas[name].detach() * cost_q[name] for name in self.cost_critics.keys())
        base_actor_loss_terms = self.alpha.detach() * log_prob - q_new + lagrangian_penalty
        base_actor_loss = base_actor_loss_terms.mean()
        actor_loss = base_actor_loss
        action_risk_info: Dict[str, float] = {}
        if action_risk_enabled:
            assert action_for_fse is not None and self.action_fse_runtime is not None
            required = ("frame_tokens", "frame_token_mask", "frame_entity_valid_mask", "frame_token_type_ids", "frame_token_role_ids")
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(f"Action-risk replay batch missing required frame token fields: {missing}")
            tokens_for_action_fse = batch["frame_tokens"]
            token_mask_for_action_fse = batch["frame_token_mask"]
            entity_valid_mask_for_action_fse = batch["frame_entity_valid_mask"]
            token_type_ids_for_action_fse = batch["frame_token_type_ids"]
            token_role_ids_for_action_fse = batch["frame_token_role_ids"]
            if is_fse_action_risk_shuffled_mode(self.cfg.train.mode):
                perm = torch.randperm(int(tokens_for_action_fse.shape[0]), device=tokens_for_action_fse.device)
                tokens_for_action_fse = tokens_for_action_fse[perm]
                token_mask_for_action_fse = token_mask_for_action_fse[perm]
                entity_valid_mask_for_action_fse = entity_valid_mask_for_action_fse[perm]
                token_type_ids_for_action_fse = token_type_ids_for_action_fse[perm]
                token_role_ids_for_action_fse = token_role_ids_for_action_fse[perm]

            action_min = float(action_for_fse.detach().min().item())
            action_max = float(action_for_fse.detach().max().item())
            if action_min < -1.0001 or action_max > 1.0001:
                raise ValueError(f"action_for_fse must stay in [-1,1], got min={action_min:.6f}, max={action_max:.6f}.")
            risk_out = self.action_fse_runtime.compute_penalty(
                tokens=tokens_for_action_fse,
                token_mask=token_mask_for_action_fse,
                entity_valid_mask=entity_valid_mask_for_action_fse,
                token_type_ids=token_type_ids_for_action_fse,
                token_role_ids=token_role_ids_for_action_fse,
                action=action_for_fse,
                beta_risk=float(self.cfg.fse.action_risk_beta),
                kappa=float(self.cfg.fse.action_risk_uncertainty_kappa),
            )
            risk_grad = torch.autograd.grad(
                risk_out.risk_norm.mean(),
                action_for_fse,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            risk_grad_norm = float(risk_grad.detach().reshape(risk_grad.shape[0], -1).norm(dim=-1).mean().item())
            grad_base_norm = float("nan")
            cos_grad = float("nan")
            grad_ratio = float("nan")
            diag_interval = max(1, int(self.cfg.fse.action_risk_grad_diag_interval))
            if self.update_counter % diag_interval == 0:
                grad_base = torch.autograd.grad(
                    base_actor_loss,
                    action_for_fse,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                grad_base_flat = grad_base.detach().reshape(grad_base.shape[0], -1)
                grad_risk_flat = risk_grad.detach().reshape(risk_grad.shape[0], -1)
                grad_base_norm = float(grad_base_flat.norm(dim=-1).mean().item())
                risk_grad_norm_for_ratio = float(grad_risk_flat.norm(dim=-1).mean().item())
                cos_grad = float(F.cosine_similarity(grad_base_flat, grad_risk_flat, dim=-1, eps=1e-8).mean().item())
                grad_ratio = float(risk_grad_norm_for_ratio / max(grad_base_norm, 1e-8))
            actor_loss = base_actor_loss + risk_out.risk_penalty.mean()
            risk_active_threshold = float(self.cfg.fse.action_risk_active_threshold)
            gate_active_threshold = float(self.cfg.fse.action_risk_gate_active_threshold)
            risk_norm_detached = risk_out.risk_norm.detach()
            gate_detached = risk_out.uncertainty_gate.detach()
            risk_high = risk_norm_detached > risk_active_threshold
            gate_suppressed = gate_detached <= gate_active_threshold
            uncertainty = risk_out.uncertainty.detach()
            action_risk_info = {
                "fse_action_risk_raw_mean": float(risk_out.risk_raw.detach().mean().item()),
                "fse_action_risk_norm_mean": float(risk_norm_detached.mean().item()),
                "fse_action_uncertainty_h10_mean": float(uncertainty[:, 0, 0].mean().item()) if uncertainty.shape[1] > 0 else float("nan"),
                "fse_action_uncertainty_h20_mean": float(uncertainty[:, 1, 0].mean().item()) if uncertainty.shape[1] > 1 else float("nan"),
                "fse_action_uncertainty_h40_mean": float(uncertainty[:, 2, 0].mean().item()) if uncertainty.shape[1] > 2 else float("nan"),
                "fse_action_uncertainty_mean": float(risk_out.uncertainty_scalar.detach().mean().item()),
                "uncertainty_gate_mean": float(gate_detached.mean().item()),
                "uncertainty_gate_min": float(gate_detached.min().item()),
                "uncertainty_gate_max": float(gate_detached.max().item()),
                "risk_penalty_mean": float(risk_out.risk_penalty.detach().mean().item()),
                "risk_high_rate": float(risk_high.float().mean().item()),
                "uncertainty_gate_suppressed_rate": float(gate_suppressed.float().mean().item()),
                "risk_penalty_active_rate": float((risk_high & (~gate_suppressed)).float().mean().item()),
                "action_for_fse_min": action_min,
                "action_for_fse_max": action_max,
                "action_for_fse_abs_mean": float(action_for_fse.detach().abs().mean().item()),
                "risk_grad_norm": risk_grad_norm,
                "cos_grad_actor_base_risk": cos_grad,
                "grad_base_norm": grad_base_norm,
                "grad_risk_norm": risk_grad_norm,
                "risk_to_base_grad_norm_ratio": grad_ratio,
                "ood_action_rate": float("nan"),
            }
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
        if action_risk_info:
            action_risk_info.update(self.action_fse_runtime.freeze_fields() if self.action_fse_runtime is not None else {})
            out.update(action_risk_info)
        actor_gate_mean, actor_gate_std = self._module_gate_stats(self.actor)
        q1_gate_mean, q1_gate_std = self._module_gate_stats(self.q1)
        q2_gate_mean, q2_gate_std = self._module_gate_stats(self.q2)
        out.update({
            "fse_actor_gate_mean": actor_gate_mean,
            "fse_actor_gate_std": actor_gate_std,
            "fse_reward_critic1_gate_mean": q1_gate_mean,
            "fse_reward_critic1_gate_std": q1_gate_std,
            "fse_reward_critic2_gate_mean": q2_gate_mean,
            "fse_reward_critic2_gate_std": q2_gate_std,
        })
        for cost_name in ("safety", "boundary", "speed"):
            if cost_name in self.cost_critics:
                gm, gs = self._module_gate_stats(self.cost_critics[cost_name])
            else:
                gm, gs = float("nan"), float("nan")
            out[f"fse_cost_critic_{cost_name}_gate_mean"] = gm
            out[f"fse_cost_critic_{cost_name}_gate_std"] = gs
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
    if is_fse_rl_mode(cfg.train.mode):
        return LagrangianConstrainedSACAgent(state_dim, action_dim, cfg, action_low, action_high)
    if cfg.train.mode == "paper_csac_lagrangian":
        return LagrangianConstrainedSACAgent(state_dim, action_dim, cfg, action_low, action_high)
    if cfg.train.mode == "paper_sac_pure":
        return SACAgent(state_dim, action_dim, cfg, action_low, action_high)
    raise ValueError(f"Unsupported mode: {cfg.train.mode}")


def _script_sha256() -> str:
    try:
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return ""


def _module_state_dict_or_none(obj: Any) -> Optional[dict]:
    return obj.state_dict() if isinstance(obj, nn.Module) else None


def agent_checkpoint_payload(
    agent,
    cfg: Config,
    state_dim: int,
    action_dim: int,
    action_low,
    action_high,
    fse_runtime: Optional[FrozenFSEBottleneckRuntime] = None,
) -> dict:
    payload = {
        "checkpoint_type": "baseline_policy",
        "mode": str(cfg.train.mode),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "action_space_low": np.asarray(action_low, dtype=np.float32).reshape(-1).tolist(),
        "action_space_high": np.asarray(action_high, dtype=np.float32).reshape(-1).tolist(),
        "seed": int(cfg.train.seed),
        "episodes": int(cfg.train.episodes),
        "max_steps": int(cfg.train.max_steps_per_episode),
        "traffic_density": str(cfg.env.traffic_density),
        "config": asdict(cfg),
        "script_sha256": _script_sha256(),
        "python": sys.version,
        "torch": torch.__version__,
        "timestamp_utc": _utc_now_iso(),
        "observation_normalization": {"env_normalize_obs": bool(cfg.env.normalize_obs), "absolute_obs": bool(cfg.env.absolute_obs)},
        "reward_normalization": {"reward_type": str(cfg.reward.reward_type), "env_reward_scale": float(cfg.reward.env_reward_scale)},
        "action_scaling": {
            "action_low": np.asarray(action_low, dtype=np.float32).reshape(-1).tolist(),
            "action_high": np.asarray(action_high, dtype=np.float32).reshape(-1).tolist(),
        },
        "actor_state_dict": agent.actor.state_dict() if hasattr(agent, "actor") else None,
        "q1_state_dict": _module_state_dict_or_none(getattr(agent, "q1", None)),
        "q2_state_dict": _module_state_dict_or_none(getattr(agent, "q2", None)),
        "q1_target_state_dict": _module_state_dict_or_none(getattr(agent, "q1_target", None)),
        "q2_target_state_dict": _module_state_dict_or_none(getattr(agent, "q2_target", None)),
        "log_alpha": float(getattr(agent, "log_alpha", torch.tensor(float("nan"))).detach().cpu().item()) if hasattr(agent, "log_alpha") else float("nan"),
    }
    if hasattr(agent, "cost_critics") and isinstance(agent.cost_critics, nn.ModuleDict):
        payload["cost_critics_state_dict"] = agent.cost_critics.state_dict()
    if hasattr(agent, "cost_targets") and isinstance(agent.cost_targets, nn.ModuleDict):
        payload["cost_targets_state_dict"] = agent.cost_targets.state_dict()
    for name in ("lambdas", "budgets"):
        if hasattr(agent, name):
            value = getattr(agent, name)
            if isinstance(value, dict):
                payload[name] = {
                    str(k): (float(v.detach().cpu().item()) if torch.is_tensor(v) else _jsonable_plain(v))
                    for k, v in value.items()
                }
    for name in ("lambda_collision", "lambda_headway"):
        if hasattr(agent, name):
            value = getattr(agent, name)
            payload[name] = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    if is_fse_rl_mode(cfg.train.mode):
        runtime_meta = fse_runtime.audit_metadata() if fse_runtime is not None else {}
        payload["checkpoint_type"] = "fse_rl_policy"
        payload["fse_rl"] = {
            "enabled": True,
            "mode": str(cfg.train.mode),
            "fusion_mode": str(cfg.fse.fusion_mode),
            "z_mode": str(cfg.fse.z_mode),
            "random_z_distribution": str(cfg.fse.random_z_distribution),
            "random_z_std_floor": float(cfg.fse.random_z_std_floor),
            "run_tier": str(cfg.fse.run_tier),
            "dims": {
                "raw_state_dim": int(cfg.fse.raw_state_dim),
                "z_dim": int(cfg.fse.z_dim),
                "state_aug_dim": int(cfg.fse.state_aug_dim),
            },
            "fse_checkpoint_path": os.path.abspath(str(cfg.fse.checkpoint_path)) if str(cfg.fse.checkpoint_path).strip() else "",
            "fse_checkpoint_sha256": str(runtime_meta.get("fse_checkpoint_sha256", "")),
            "scenario_builder_config_hash": str(runtime_meta.get("scenario_builder_config_hash", "")),
            "global_z_stats_sha256": str(runtime_meta.get("global_z_stats_sha256", "")),
            "offline_shuffle_pool_sha256": str(runtime_meta.get("offline_shuffle_pool_sha256", "")),
            "global_z_stats_metadata": _jsonable_plain(runtime_meta.get("global_z_stats_metadata", {})),
            "offline_shuffle_pool_metadata": _jsonable_plain(runtime_meta.get("offline_shuffle_pool_metadata", {})),
            "allow_legacy_normalization": bool(cfg.fse.allow_legacy_normalization),
            "normalization_config_legacy_warning": bool(runtime_meta.get("normalization_config_legacy_warning", False)),
        }
        action_runtime = getattr(agent, "action_fse_runtime", None)
        if action_runtime is not None:
            action_meta = action_runtime.audit_metadata()
            payload["fse_rl"]["action_risk"] = {
                "enabled": True,
                "checkpoint_path": os.path.abspath(str(cfg.fse.action_risk_checkpoint_path)),
                "checkpoint_sha256": str(action_meta.get("fse_action_checkpoint_sha256", "")),
                "acceptance_gate_path": os.path.abspath(str(cfg.fse.action_risk_acceptance_json)),
                "acceptance_gate_sha256": str(action_meta.get("fse_action_acceptance_gate_sha256", "")),
                "beta": float(cfg.fse.action_risk_beta),
                "uncertainty_kappa": float(cfg.fse.action_risk_uncertainty_kappa),
                "action_metadata": _jsonable_plain(action_meta.get("action_metadata", {})),
            }
    return payload


def save_agent_checkpoint(
    agent,
    cfg: Config,
    state_dim: int,
    action_dim: int,
    action_low,
    action_high,
    save_path: str,
    fse_runtime: Optional[FrozenFSEBottleneckRuntime] = None,
) -> str:
    if not str(save_path).strip():
        return ""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(agent_checkpoint_payload(agent, cfg, state_dim, action_dim, action_low, action_high, fse_runtime=fse_runtime), save_path)
    return os.path.abspath(save_path)


def load_agent_checkpoint_for_eval(agent, cfg: Config, checkpoint_path: str, state_dim: int, action_dim: int, action_low, action_high) -> dict:
    try:
        payload = torch.load(str(checkpoint_path), map_location=torch.device(cfg.train.device), weights_only=False)
    except TypeError:
        payload = torch.load(str(checkpoint_path), map_location=torch.device(cfg.train.device))
    if str(payload.get("mode", "")) and str(payload.get("mode")) != str(cfg.train.mode):
        raise ValueError(f"Checkpoint mode {payload.get('mode')} does not match current mode {cfg.train.mode}.")
    if int(payload.get("state_dim", state_dim)) != int(state_dim):
        raise ValueError(f"Checkpoint state_dim={payload.get('state_dim')} does not match current state_dim={state_dim}.")
    if int(payload.get("action_dim", action_dim)) != int(action_dim):
        raise ValueError(f"Checkpoint action_dim={payload.get('action_dim')} does not match current action_dim={action_dim}.")
    if is_fse_rl_mode(cfg.train.mode):
        fse_rl = dict(payload.get("fse_rl") or {})
        if not bool(fse_rl.get("enabled", False)):
            raise ValueError("FSE-RL eval requires checkpoint metadata field fse_rl.enabled=true.")
        dims = dict(fse_rl.get("dims") or {})
        expected_dims = {
            "raw_state_dim": int(cfg.fse.raw_state_dim),
            "z_dim": int(cfg.fse.z_dim),
            "state_aug_dim": int(cfg.fse.state_aug_dim),
        }
        for key, expected in expected_dims.items():
            if int(dims.get(key, expected)) != int(expected):
                raise ValueError(f"Checkpoint fse_rl.dims.{key}={dims.get(key)} does not match current {key}={expected}.")
        if str(fse_rl.get("fusion_mode", "")) != str(cfg.fse.fusion_mode):
            raise ValueError("Checkpoint fse_rl.fusion_mode does not match current FSE fusion mode.")
        if str(fse_rl.get("z_mode", "")) != str(cfg.fse.z_mode):
            raise ValueError("Checkpoint fse_rl.z_mode does not match current FSE z mode.")
        current_fse_hash = sha256_file(str(cfg.fse.checkpoint_path)) if str(cfg.fse.checkpoint_path).strip() else ""
        stored_fse_hash = str(fse_rl.get("fse_checkpoint_sha256", ""))
        if current_fse_hash and stored_fse_hash and current_fse_hash != stored_fse_hash:
            raise ValueError("Checkpoint fse_rl.fse_checkpoint_sha256 does not match current FSE checkpoint.")
        current_builder_hash = scenario_builder_config_hash(cfg)
        stored_builder_hash = str(fse_rl.get("scenario_builder_config_hash", ""))
        if stored_builder_hash and stored_builder_hash != current_builder_hash:
            raise ValueError("Checkpoint fse_rl.scenario_builder_config_hash does not match current builder config.")
        current_global_hash = sha256_file(str(cfg.fse.z_global_stats_path)) if str(cfg.fse.z_global_stats_path).strip() else ""
        stored_global_hash = str(fse_rl.get("global_z_stats_sha256", ""))
        if (current_global_hash or stored_global_hash) and current_global_hash != stored_global_hash:
            raise ValueError("Checkpoint fse_rl.global_z_stats_sha256 does not match current global z stats path.")
        current_pool_hash = sha256_file(str(cfg.fse.shuffle_pool_path)) if str(cfg.fse.shuffle_pool_path).strip() else ""
        stored_pool_hash = str(fse_rl.get("offline_shuffle_pool_sha256", ""))
        if (current_pool_hash or stored_pool_hash) and current_pool_hash != stored_pool_hash:
            raise ValueError("Checkpoint fse_rl.offline_shuffle_pool_sha256 does not match current shuffle pool path.")
        if bool(fse_rl.get("allow_legacy_normalization", False)) != bool(cfg.fse.allow_legacy_normalization):
            raise ValueError("Checkpoint fse_rl.allow_legacy_normalization does not match current flag.")
        if current_fse_hash:
            try:
                fse_checkpoint = torch.load(str(cfg.fse.checkpoint_path), map_location=torch.device(cfg.train.device), weights_only=False)
            except TypeError:
                fse_checkpoint = torch.load(str(cfg.fse.checkpoint_path), map_location=torch.device(cfg.train.device))
            current_norm_legacy = not isinstance(fse_structured_normalization_payload(fse_checkpoint), dict)
            if bool(fse_rl.get("normalization_config_legacy_warning", current_norm_legacy)) != bool(current_norm_legacy):
                raise ValueError("Checkpoint fse_rl.normalization_config_legacy_warning does not match current FSE checkpoint.")
    ck_low = np.asarray(payload.get("action_space_low", action_low), dtype=np.float32).reshape(-1)
    ck_high = np.asarray(payload.get("action_space_high", action_high), dtype=np.float32).reshape(-1)
    if not np.allclose(ck_low, np.asarray(action_low, dtype=np.float32).reshape(-1), atol=1e-5):
        raise ValueError("Checkpoint action_space_low does not match current environment.")
    if not np.allclose(ck_high, np.asarray(action_high, dtype=np.float32).reshape(-1), atol=1e-5):
        raise ValueError("Checkpoint action_space_high does not match current environment.")
    actor_state = payload.get("actor_state_dict")
    if actor_state is None:
        raise ValueError("Checkpoint is missing actor_state_dict.")
    agent.actor.load_state_dict(actor_state)
    for module_name, key in (
        ("q1", "q1_state_dict"),
        ("q2", "q2_state_dict"),
        ("q1_target", "q1_target_state_dict"),
        ("q2_target", "q2_target_state_dict"),
    ):
        state = payload.get(key)
        module = getattr(agent, module_name, None)
        if state is not None and isinstance(module, nn.Module):
            module.load_state_dict(state)
    if isinstance(getattr(agent, "cost_critics", None), nn.ModuleDict) and payload.get("cost_critics_state_dict") is not None:
        agent.cost_critics.load_state_dict(payload["cost_critics_state_dict"])
    if isinstance(getattr(agent, "cost_targets", None), nn.ModuleDict) and payload.get("cost_targets_state_dict") is not None:
        agent.cost_targets.load_state_dict(payload["cost_targets_state_dict"])
    if "log_alpha" in payload and hasattr(agent, "log_alpha"):
        with torch.no_grad():
            agent.log_alpha.copy_(torch.as_tensor(float(payload["log_alpha"]), dtype=torch.float32, device=agent.device))
    if hasattr(agent, "lambdas") and isinstance(getattr(agent, "lambdas"), dict) and isinstance(payload.get("lambdas"), dict):
        for key, value in payload["lambdas"].items():
            if key in agent.lambdas:
                agent.lambdas[key] = torch.tensor(float(value), dtype=torch.float32, device=agent.device)
    for key in ("lambda_collision", "lambda_headway"):
        if hasattr(agent, key) and key in payload:
            setattr(agent, key, torch.tensor(float(payload[key]), dtype=torch.float32, device=agent.device))
    return payload


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
    "scenario_frame_schema_warning_count",
    "scenario_frame_missing_optional_count",
    "scenario_frame_memory_key_bin_count",
    "scenario_frame_identity_switch_rate",
]

FSE_TRAIN_DIAGNOSTIC_FIELDS = [
    "fse_z_real_norm_mean",
    "fse_z_used_norm_mean",
    "fse_z_used_abs_mean",
    "fse_z_real_abs_mean",
    "fse_z_real_dim_variance_mean",
    "fse_z_used_dim_variance_mean",
    "fse_z_real_batch_variance_mean",
    "fse_z_used_batch_variance_mean",
    "fse_z_real_temporal_delta_mean",
    "fse_z_used_temporal_delta_mean",
    "fse_state_aug_norm_ratio_p10",
    "fse_state_aug_norm_ratio_p50",
    "fse_state_aug_norm_ratio_p90",
    "shuffled_cross_episode_rate",
    "shuffled_same_episode_far_rate",
    "shuffled_random_fallback_rate",
    "shuffled_self_sample_count",
    "shuffled_nearby_episode_sample_count",
    "state_aug_has_nan",
    "state_aug_has_inf",
    "next_state_aug_has_nan",
    "next_state_aug_has_inf",
    "fse_actor_gate_mean",
    "fse_actor_gate_std",
    "fse_reward_critic1_gate_mean",
    "fse_reward_critic1_gate_std",
    "fse_reward_critic2_gate_mean",
    "fse_reward_critic2_gate_std",
    "fse_cost_critic_safety_gate_mean",
    "fse_cost_critic_safety_gate_std",
    "fse_cost_critic_boundary_gate_mean",
    "fse_cost_critic_boundary_gate_std",
    "fse_cost_critic_speed_gate_mean",
    "fse_cost_critic_speed_gate_std",
    "fse_param_count_frozen",
    "trainable_fse_param_count",
    "fse_optimizer_param_count",
    "fse_nonzero_grad_param_count",
    "fse_grad_norm",
    "fse_weight_delta_norm",
    "fse_forward_count",
    "fse_random_std_clipped_dims",
    "fse_action_risk_raw_mean",
    "fse_action_risk_norm_mean",
    "fse_action_uncertainty_h10_mean",
    "fse_action_uncertainty_h20_mean",
    "fse_action_uncertainty_h40_mean",
    "fse_action_uncertainty_mean",
    "uncertainty_gate_mean",
    "uncertainty_gate_min",
    "uncertainty_gate_max",
    "risk_penalty_mean",
    "risk_high_rate",
    "uncertainty_gate_suppressed_rate",
    "risk_penalty_active_rate",
    "action_for_fse_min",
    "action_for_fse_max",
    "action_for_fse_abs_mean",
    "risk_grad_norm",
    "cos_grad_actor_base_risk",
    "grad_base_norm",
    "grad_risk_norm",
    "risk_to_base_grad_norm_ratio",
    "ood_action_rate",
    "fse_action_param_count",
    "fse_action_trainable_param_count",
    "fse_action_nonzero_grad_param_count",
    "fse_action_hash_unchanged",
]

for _field in FSE_TRAIN_DIAGNOSTIC_FIELDS:
    if _field not in TRAIN_DIAGNOSTIC_FIELDS:
        TRAIN_DIAGNOSTIC_FIELDS.append(_field)


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
    timestamp = _timestamp()
    run_name = f"seed_{cfg.train.seed}_{timestamp}"
    predicted_event_path = os.path.abspath(os.path.join(path, run_name, f"events.out.tfevents.{timestamp}.placeholder"))
    if os.name == "nt" and len(predicted_event_path) >= 240:
        mode_hash = stable_int_hash(cfg.train.mode) % 1000000
        run_name = f"s{cfg.train.seed}_{mode_hash:06d}_{timestamp}"
    run_dir = os.path.join(path, run_name)
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


def _result_dir_from_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return os.path.abspath("results")
    parent = os.path.dirname(raw)
    return os.path.abspath(parent or ".")


def make_fse_state_aug(raw_state: np.ndarray, z_info: FSEStepZ, raw_state_dim: int, z_dim: int) -> np.ndarray:
    if not bool(z_info.z_used_valid):
        raise ValueError("FSE state augmentation requires z_used_valid=true.")
    raw = np.asarray(raw_state, dtype=np.float32).reshape(int(raw_state_dim))
    z_used = np.asarray(z_info.z_used, dtype=np.float32).reshape(int(z_dim))
    state_aug = np.concatenate([raw, z_used], axis=0).astype(np.float32)
    if np.any(np.isnan(state_aug)):
        raise ValueError("state_aug contains NaN before policy/replay use.")
    if np.any(np.isinf(state_aug)):
        raise ValueError("state_aug contains Inf before policy/replay use.")
    return state_aug


def fse_build_frame(
    builder: Optional[ScenarioFrameBuilder],
    obs: np.ndarray,
    scene: dict,
    last_action: np.ndarray,
    episode_id: int,
    step_id: int,
    traffic_density: str,
) -> Optional[ScenarioFrame]:
    if builder is None:
        return None
    return builder.build(
        obs=obs,
        scene=scene,
        last_action=last_action,
        episode_id=int(episode_id),
        step_id=int(step_id),
        traffic_density=traffic_density,
    )


def fse_trace_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    base = [
        "episode_id",
        "step_id",
        "mode",
        "z_source",
        "z_real_valid",
        "z_used_valid",
        "fse_forward_executed",
        "z_real_norm",
        "z_used_norm",
        "z_real_dim_variance",
        "z_used_dim_variance",
        "next_z_source",
        "next_z_real_valid",
        "next_z_used_valid",
        "next_z_real_norm",
        "next_z_used_norm",
        "fse_state_aug_norm_ratio",
        "uncertainty_mean",
        "risk_mean",
        "shuffled_source_episode_id",
        "shuffled_source_step_id",
        "shuffled_fallback_flag",
        "shuffled_self_sample_flag",
        "shuffled_nearby_episode_sample_flag",
    ]
    for row in rows:
        for key in row.keys():
            if key not in base:
                base.append(str(key))
    return base


def export_fse_z_trace(diag: Optional[FSEZDiagnostics], save_path: str) -> str:
    if diag is None:
        return ""
    rows = list(diag.trace_rows)
    return export_csv(rows, save_path, fse_trace_fieldnames(rows))


def fse_resolved_config_payload(
    cfg: Config,
    runtime: FrozenFSEBottleneckRuntime,
    raw_state_dim: int,
    action_dim: int,
    action_runtime: Optional[FrozenActionFSEPenaltyRuntime] = None,
) -> Dict[str, Any]:
    global_stats_path = str(cfg.fse.z_global_stats_path).strip()
    shuffle_pool_path = str(cfg.fse.shuffle_pool_path).strip()
    return {
        "mode": str(cfg.train.mode),
        "fusion_mode": str(cfg.fse.fusion_mode),
        "z_mode": str(cfg.fse.z_mode),
        "random_z_distribution": str(cfg.fse.random_z_distribution),
        "random_z_std_floor": float(cfg.fse.random_z_std_floor),
        "run_tier": str(cfg.fse.run_tier),
        "allow_legacy_fse_normalization": bool(cfg.fse.allow_legacy_normalization),
        "fse_checkpoint_path": os.path.abspath(str(cfg.fse.checkpoint_path)),
        "fse_checkpoint_sha256": str(runtime.checkpoint_sha256),
        "global_z_stats_path": os.path.abspath(global_stats_path) if global_stats_path else "",
        "global_z_stats_sha256": str(runtime.global_stats_hash),
        "shuffle_pool_path": os.path.abspath(shuffle_pool_path) if shuffle_pool_path else "",
        "shuffle_pool_sha256": str(runtime.shuffle_pool_hash),
        "global_z_stats_metadata": _jsonable_plain(runtime.global_stats_metadata),
        "shuffle_pool_metadata": _jsonable_plain(runtime.shuffle_pool_metadata),
        "raw_state_dim": int(raw_state_dim),
        "z_dim": int(runtime.z_dim),
        "state_aug_dim": int(runtime.state_aug_dim),
        "action_dim": int(action_dim),
        "seed": int(cfg.train.seed),
        "traffic_density": str(cfg.env.traffic_density),
        "episodes": int(cfg.train.episodes),
        "max_steps": int(cfg.train.max_steps_per_episode),
        "eval_episodes": int(cfg.eval.episodes),
        "scenario_builder_config_hash": str(runtime.scenario_builder_config_hash),
        "script_git_sha_or_file_hash": str(runtime.script_file_hash),
        "warnings": list(runtime.warning_messages),
        "action_risk_enabled": bool(action_runtime is not None),
        "fse_action_risk_checkpoint_path": os.path.abspath(str(cfg.fse.action_risk_checkpoint_path)) if str(cfg.fse.action_risk_checkpoint_path).strip() else "",
        "fse_action_risk_checkpoint_sha256": str(action_runtime.checkpoint_sha256) if action_runtime is not None else "",
        "fse_action_risk_acceptance_json": os.path.abspath(str(cfg.fse.action_risk_acceptance_json)) if str(cfg.fse.action_risk_acceptance_json).strip() else "",
        "fse_action_risk_acceptance_sha256": str(action_runtime.acceptance_gate_sha256) if action_runtime is not None else "",
        "fse_action_risk_beta": float(cfg.fse.action_risk_beta),
        "fse_action_risk_uncertainty_kappa": float(cfg.fse.action_risk_uncertainty_kappa),
        "fse_action_risk_grad_diag_interval": int(cfg.fse.action_risk_grad_diag_interval),
        "fse_action_risk_audit": action_runtime.audit_metadata() if action_runtime is not None else {},
    }


def fse_assert_update_finite(update_info: Dict[str, Any]) -> None:
    action_risk_required_finite = {
        "fse_action_risk_raw_mean",
        "fse_action_risk_norm_mean",
        "fse_action_uncertainty_mean",
        "uncertainty_gate_mean",
        "risk_penalty_mean",
        "risk_penalty_active_rate",
        "risk_grad_norm",
        "action_for_fse_min",
        "action_for_fse_max",
        "action_for_fse_abs_mean",
    }
    for key, value in update_info.items():
        if not (key.endswith("_loss") or key in ("q1_loss", "q2_loss", "actor_loss", "alpha_loss") or key in action_risk_required_finite):
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if not np.isfinite(numeric):
            raise ValueError(f"Non-finite FSE-RL training metric detected: {key}={numeric}")


def default_result_path(cfg: Config) -> str:
    return os.path.join("results", f"{_timestamp()}_{cfg.train.mode}_seed{cfg.train.seed}_run_result.json")


def scenario_frame_probe_path_from_result(result_path: str) -> str:
    base = str(result_path).strip()
    if not base:
        return ""
    root, ext = os.path.splitext(base)
    suffix = ext if ext else ".json"
    if root.endswith("_run_result"):
        root = root[: -len("_run_result")]
    return f"{root}_scenario_frame_probe{suffix}"


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


def _seeded_file_path(path: str, seed: int, append_when_missing: bool = True) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    if "{seed}" in raw:
        return raw.replace("{seed}", str(seed))
    if not append_when_missing:
        return raw
    stem, ext = os.path.splitext(raw)
    seed_tag = f"_seed{seed}"
    if re.search(r"_seed-?\d+$", stem):
        stem = re.sub(r"_seed-?\d+$", seed_tag, stem)
    else:
        stem = f"{stem}{seed_tag}"
    return f"{stem}{ext}"


def _seeded_dir_path(path: str, seed: int, append_when_missing: bool = True) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    if "{seed}" in raw:
        return raw.replace("{seed}", str(seed))
    if not append_when_missing:
        return raw
    if re.search(r"(?:^|[\\/])seed[_-]?-?\d+$", raw):
        return re.sub(r"seed[_-]?-?\d+$", f"seed_{seed}", raw)
    return os.path.join(raw, f"seed_{seed}")


def _apply_seed_specific_paths(cfg: Config, seed: int, append_when_missing: bool = True) -> None:
    cfg.train.seed = int(seed)
    if cfg.diagnostics.export_train_log:
        cfg.diagnostics.train_log_path = _seeded_file_path(cfg.diagnostics.train_log_path, seed, append_when_missing)
    if cfg.diagnostics.save_train_json:
        cfg.diagnostics.train_json_path = _seeded_file_path(cfg.diagnostics.train_json_path, seed, append_when_missing)
    if str(cfg.diagnostics.run_result_json_path).strip():
        cfg.diagnostics.run_result_json_path = _seeded_file_path(cfg.diagnostics.run_result_json_path, seed, append_when_missing)
    if str(getattr(cfg.diagnostics, "checkpoint_save_path", "")).strip():
        cfg.diagnostics.checkpoint_save_path = _seeded_file_path(cfg.diagnostics.checkpoint_save_path, seed, append_when_missing)
    if str(cfg.diagnostics.tensorboard_log_dir).strip():
        cfg.diagnostics.tensorboard_log_dir = _seeded_dir_path(cfg.diagnostics.tensorboard_log_dir, seed, append_when_missing)
    if str(cfg.eval.video_dir).strip():
        cfg.eval.video_dir = _seeded_dir_path(cfg.eval.video_dir, seed, append_when_missing)


def _mode_tagged_file_path(path: str, mode: str, append_when_missing: bool = True) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    if "{mode}" in raw:
        return raw.replace("{mode}", str(mode))
    if not append_when_missing:
        return raw
    stem, ext = os.path.splitext(raw)
    if re.search(r"_(paper_sac_pure|paper_csac_lagrangian)$", stem):
        stem = re.sub(r"_(paper_sac_pure|paper_csac_lagrangian)$", f"_{mode}", stem)
    else:
        stem = f"{stem}_{mode}"
    return f"{stem}{ext}"


def _mode_tagged_dir_path(path: str, mode: str, append_when_missing: bool = True) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    if "{mode}" in raw:
        return raw.replace("{mode}", str(mode))
    if not append_when_missing:
        return raw
    return os.path.join(raw, str(mode))


def _apply_mode_specific_paths(cfg: Config, mode: str, append_when_missing: bool = True) -> None:
    if cfg.diagnostics.export_train_log:
        cfg.diagnostics.train_log_path = _mode_tagged_file_path(cfg.diagnostics.train_log_path, mode, append_when_missing)
    if cfg.diagnostics.save_train_json:
        cfg.diagnostics.train_json_path = _mode_tagged_file_path(cfg.diagnostics.train_json_path, mode, append_when_missing)
    if str(cfg.diagnostics.run_result_json_path).strip():
        cfg.diagnostics.run_result_json_path = _mode_tagged_file_path(cfg.diagnostics.run_result_json_path, mode, append_when_missing)
    if str(getattr(cfg.diagnostics, "checkpoint_save_path", "")).strip():
        cfg.diagnostics.checkpoint_save_path = _mode_tagged_file_path(cfg.diagnostics.checkpoint_save_path, mode, append_when_missing)
    if str(cfg.diagnostics.tensorboard_log_dir).strip():
        cfg.diagnostics.tensorboard_log_dir = _mode_tagged_dir_path(cfg.diagnostics.tensorboard_log_dir, mode, append_when_missing)
    if str(cfg.eval.video_dir).strip():
        cfg.eval.video_dir = _mode_tagged_dir_path(cfg.eval.video_dir, mode, append_when_missing)


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
        parent = os.path.dirname(base)
        if os.path.basename(parent) in ("{seed}", "seed_{seed}", "seed-{seed}"):
            return os.path.join(os.path.dirname(parent), "multi_seed_summary.json")
        if "{seed}" in base:
            base = base.replace("{seed}", "all")
        if "{mode}" in base:
            base = base.replace("{mode}", str(cfg.train.mode))
        stem, ext = os.path.splitext(base)
        final_ext = ext or ".json"
        return f"{stem}_multi_seed_summary{final_ext}"
    return os.path.join(PUBLIC_STAGE0_BASELINE_ROOT, str(cfg.train.mode), f"{_timestamp()}_multi_seed_summary.json")


def _resolve_summary_template(path: str, mode: str) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    return raw.replace("{mode}", str(mode)).replace("{seed}", "all")


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
    output_path = _resolve_summary_template(summary_path, base_cfg.train.mode) or _default_multi_seed_summary_path(base_cfg)
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
    output_path = _resolve_summary_template(summary_path, "all_modes") or os.path.join(
        PUBLIC_STAGE0_BASELINE_ROOT, f"{_timestamp()}_paper_mode_comparison_summary.json"
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
    if "risk_penalty_mean" in update_info:
        msg += (
            f" risk_penalty={update_info.get('risk_penalty_mean', 0.0):.4f}"
            f" risk_grad={update_info.get('risk_grad_norm', 0.0):.4f}"
            f" gate={update_info.get('uncertainty_gate_mean', 0.0):.4f}"
        )
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


def evaluate_policy(cfg: Config, agent, raw_state_dim: int = 0, output_dir: str = "") -> dict:
    print("\n================ EVAL RAW POLICY START ================\n")
    eval_results = []
    saved_videos: List[str] = []
    video_events: List[dict] = []
    fse_enabled = bool(is_fse_rl_mode(cfg.train.mode))
    scenario_builder = ScenarioFrameBuilder(cfg) if bool(cfg.train.enable_scenario_frame) else None
    if fse_enabled and scenario_builder is None:
        raise ValueError("FSE-RL eval requires scenario frames.")
    if scenario_builder is not None:
        print_scenario_frame_schema_probe(scenario_builder, prefix="EVAL")

    eval_runtime: Optional[FrozenFSEBottleneckRuntime] = None
    eval_diag: Optional[FSEZDiagnostics] = None
    if fse_enabled:
        if int(raw_state_dim) <= 0:
            raise ValueError("FSE-RL eval requires raw_state_dim from the training environment.")
        eval_runtime = FrozenFSEBottleneckRuntime(cfg, raw_state_dim=int(raw_state_dim), output_dir=output_dir)
        cfg.fse.z_dim = int(eval_runtime.z_dim)
        cfg.fse.raw_state_dim = int(raw_state_dim)
        cfg.fse.state_aug_dim = int(eval_runtime.state_aug_dim)
        eval_diag = FSEZDiagnostics(
            int(eval_runtime.z_dim),
            trace_stride=int(cfg.fse.z_trace_stride),
            trace_max_rows=int(cfg.fse.z_trace_max_rows),
        )

    for ep in range(cfg.eval.episodes):
        eval_seed = int(cfg.train.seed + cfg.eval.seed_offset + ep)
        env = HighwayEnvWrapper(cfg, render=cfg.eval.render, record_video=cfg.eval.save_video)
        frames: List[np.ndarray] = []
        scenario_frame_logged = False
        try:
            raw_state, _ = env.reset(seed=eval_seed)
            state_for_agent = np.asarray(raw_state, dtype=np.float32)
            current_z: Optional[FSEStepZ] = None
            current_frame: Optional[ScenarioFrame] = None
            if fse_enabled:
                assert eval_runtime is not None
                scene0 = env.get_scene_dict()
                current_frame = fse_build_frame(
                    scenario_builder,
                    raw_state,
                    scene0,
                    env.last_action,
                    episode_id=int(ep),
                    step_id=0,
                    traffic_density=cfg.env.traffic_density,
                )
                current_z = eval_runtime.compute(current_frame, episode_id=int(ep), step_id=0)
                state_for_agent = make_fse_state_aug(raw_state, current_z, int(raw_state_dim), int(eval_runtime.z_dim))
            agent.reset_episode_context()
            if eval_diag is not None:
                eval_diag.reset_episode()
            if cfg.eval.save_video:
                first_frame = env.render_frame()
                if first_frame is not None:
                    frames.append(first_frame)
            metrics = EpisodeMetrics(cfg)
            for t in range(cfg.train.max_steps_per_episode):
                scene = env.get_scene_dict()
                if scenario_builder is not None:
                    if fse_enabled:
                        scenario_frame = current_frame
                    else:
                        scenario_frame = fse_build_frame(
                            scenario_builder,
                            raw_state,
                            scene,
                            env.last_action,
                            episode_id=int(ep),
                            step_id=int(t),
                            traffic_density=cfg.env.traffic_density,
                        )
                    interval = max(0, int(getattr(cfg.train, "scenario_frame_probe_interval", 0)))
                    detailed_probe = bool(getattr(cfg.train, "scenario_frame_debug", False)) or (interval > 0 and t % interval == 0)
                    if scenario_frame is not None and (not scenario_frame_logged or detailed_probe):
                        print_scenario_frame_probe(scenario_frame, scene, detailed=detailed_probe)
                        scenario_frame_logged = True
                action = env.clip_action(agent.select_action(state_for_agent, evaluate=True))
                next_raw_state, env_reward, terminated, truncated, _ = env.step(action)
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
                next_state_for_agent = np.asarray(next_raw_state, dtype=np.float32)
                next_z: Optional[FSEStepZ] = None
                next_frame: Optional[ScenarioFrame] = None
                if fse_enabled:
                    assert eval_runtime is not None and current_z is not None
                    if current_z.z_real_valid:
                        eval_runtime.append_shuffle_buffer(int(ep), int(t), current_z)
                    next_frame = fse_build_frame(
                        scenario_builder,
                        next_raw_state,
                        next_scene,
                        env.last_action,
                        episode_id=int(ep),
                        step_id=int(t + 1),
                        traffic_density=cfg.env.traffic_density,
                    )
                    next_z = eval_runtime.compute(next_frame, episode_id=int(ep), step_id=int(t + 1))
                    next_state_for_agent = make_fse_state_aug(next_raw_state, next_z, int(raw_state_dim), int(eval_runtime.z_dim))
                    assert eval_diag is not None
                    eval_diag.record(
                        episode_id=int(ep),
                        step_id=int(t),
                        raw_state=np.asarray(raw_state, dtype=np.float32),
                        current=current_z,
                        next_info=next_z,
                        state_aug=np.asarray(state_for_agent, dtype=np.float32),
                        next_state_aug=np.asarray(next_state_for_agent, dtype=np.float32),
                        mode=str(cfg.train.mode),
                    )
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
                raw_state = next_raw_state
                state_for_agent = next_state_for_agent
                current_z = next_z
                current_frame = next_frame
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
    eval_payload = {
        "raw": {"episodes": eval_results, "summary": summary},
        "primary": summary,
        "primary_report": "raw",
        "saved_videos": saved_videos,
        "video_events": video_events,
    }
    if scenario_builder is not None:
        eval_payload["scenario_frame_probe"] = scenario_builder.probe_summary()
    if eval_runtime is not None and eval_diag is not None:
        eval_payload["fse_rl"] = {
            "audit": eval_runtime.audit_metadata(),
            "freeze": eval_runtime.freeze_fields(),
            "diagnostics": eval_diag.fields(),
            "fse_forward_count": int(eval_runtime.fse_forward_count),
            "error_type": str(eval_runtime.error_type),
        }
        eval_trace_path = os.path.join(output_dir or ".", "fse_eval_z_trace.csv")
        eval_payload["fse_z_trace_csv_path"] = export_fse_z_trace(eval_diag, eval_trace_path)
    return eval_payload


def run(cfg: Config) -> dict:
    if cfg.train.mode not in ALL_MODES:
        raise ValueError(f"Unsupported mode: {cfg.train.mode}")
    set_seed(cfg.train.seed)
    fse_enabled = bool(is_fse_rl_mode(cfg.train.mode))
    result_path = str(cfg.diagnostics.run_result_json_path).strip() or default_result_path(cfg)
    result_dir = _result_dir_from_path(result_path)
    print(
        f"[RUN-CONFIG] mode={cfg.train.mode} reward_type={cfg.reward.reward_type} "
        f"goal_modes_disabled=True traffic_density={cfg.env.traffic_density} "
        f"scenario_frame_enabled={bool(cfg.train.enable_scenario_frame)} "
        f"seed={cfg.train.seed} device={describe_runtime_device(cfg.train.device)} "
        f"scenario=4lanes_30mps_20s_absobs_goal500 "
        f"penalty_impl={cfg.sac.penalty_impl} stage={cfg.sac.upgrade_stage}"
    )
    env = HighwayEnvWrapper(cfg, render=cfg.train.render)
    raw_state, _ = env.reset(seed=cfg.train.seed)
    raw_state_dim = int(raw_state.shape[0])
    action_dim = env.action_dim()
    scenario_builder = ScenarioFrameBuilder(cfg) if bool(cfg.train.enable_scenario_frame) else None
    if fse_enabled and scenario_builder is None:
        raise ValueError("FSE-RL modes require scenario frames.")
    fse_runtime: Optional[FrozenFSEBottleneckRuntime] = None
    action_fse_runtime: Optional[FrozenActionFSEPenaltyRuntime] = None
    fse_diag: Optional[FSEZDiagnostics] = None
    resolved_config_path = ""
    state_dim = int(raw_state_dim)
    if fse_enabled:
        fse_runtime = FrozenFSEBottleneckRuntime(cfg, raw_state_dim=raw_state_dim, output_dir=result_dir)
        cfg.fse.raw_state_dim = int(raw_state_dim)
        cfg.fse.z_dim = int(fse_runtime.z_dim)
        cfg.fse.state_aug_dim = int(fse_runtime.state_aug_dim)
        state_dim = int(fse_runtime.state_aug_dim)
        fse_diag = FSEZDiagnostics(
            int(fse_runtime.z_dim),
            trace_stride=int(cfg.fse.z_trace_stride),
            trace_max_rows=int(cfg.fse.z_trace_max_rows),
        )
        if is_fse_action_risk_mode(cfg.train.mode):
            action_fse_runtime = FrozenActionFSEPenaltyRuntime(cfg, action_dim=action_dim, output_dir=result_dir)
        resolved_config_path = export_json(
            fse_resolved_config_payload(
                cfg,
                fse_runtime,
                raw_state_dim=raw_state_dim,
                action_dim=action_dim,
                action_runtime=action_fse_runtime,
            ),
            os.path.join(result_dir, "resolved_config.json"),
        )
    agent = build_agent(state_dim, action_dim, cfg, env.action_low, env.action_high)
    if action_fse_runtime is not None:
        agent.attach_action_fse_runtime(action_fse_runtime)
    tb_writer, tb_run_dir = create_tensorboard_writer(cfg)
    if tb_run_dir:
        print(f"[DIAG] tensorboard run dir: {tb_run_dir}")

    total_steps = 0
    train_episode_results: List[dict] = []
    train_diagnostic_rows: List[dict] = []
    start_time = time.time()
    if scenario_builder is not None:
        print_scenario_frame_schema_probe(scenario_builder, prefix="TRAIN")
    try:
        for ep in range(cfg.train.episodes):
            raw_state, _ = env.reset(seed=cfg.train.seed + ep)
            agent.reset_episode_context()
            if fse_diag is not None:
                fse_diag.reset_episode()
            metrics = EpisodeMetrics(cfg)
            last_update_info: Dict[str, float] = {}
            scenario_frame_logged = False
            state_for_agent = np.asarray(raw_state, dtype=np.float32)
            current_z: Optional[FSEStepZ] = None
            current_frame: Optional[ScenarioFrame] = None
            if fse_enabled:
                assert fse_runtime is not None
                scene0 = env.get_scene_dict()
                current_frame = fse_build_frame(
                    scenario_builder,
                    raw_state,
                    scene0,
                    env.last_action,
                    episode_id=int(ep),
                    step_id=0,
                    traffic_density=cfg.env.traffic_density,
                )
                current_z = fse_runtime.compute(current_frame, episode_id=int(ep), step_id=0)
                state_for_agent = make_fse_state_aug(raw_state, current_z, raw_state_dim, int(fse_runtime.z_dim))
            for t in range(cfg.train.max_steps_per_episode):
                scene = env.get_scene_dict()
                if scenario_builder is not None:
                    if fse_enabled:
                        scenario_frame = current_frame
                    else:
                        scenario_frame = fse_build_frame(
                            scenario_builder,
                            raw_state,
                            scene,
                            env.last_action,
                            episode_id=int(ep),
                            step_id=int(t),
                            traffic_density=cfg.env.traffic_density,
                        )
                    interval = max(0, int(getattr(cfg.train, "scenario_frame_probe_interval", 0)))
                    detailed_probe = bool(getattr(cfg.train, "scenario_frame_debug", False)) or (interval > 0 and total_steps % interval == 0)
                    if scenario_frame is not None and (not scenario_frame_logged or detailed_probe):
                        print_scenario_frame_probe(scenario_frame, scene, detailed=detailed_probe)
                        scenario_frame_logged = True
                if total_steps < cfg.sac.start_steps:
                    action = env.sample_random_action()
                else:
                    action = env.clip_action(agent.select_action(state_for_agent, evaluate=False))

                next_raw_state, env_reward, terminated, truncated, _ = env.step(action)
                next_scene = env.get_scene_dict()
                next_is_success = terminal_success(cfg, next_scene)
                next_is_failure = terminal_failure(cfg, next_scene)
                timeout_failure = bool((truncated or (t + 1 >= cfg.train.max_steps_per_episode)) and not next_is_success and not next_is_failure)
                reward, reward_terms = compute_reward(cfg, scene, next_scene, action, env_reward, timeout_failure=timeout_failure)
                cost_dict = compute_costs(cfg, scene, next_scene, action)
                done = bool(terminated or truncated or next_is_failure or next_is_success or timeout_failure)
                next_state_for_agent = np.asarray(next_raw_state, dtype=np.float32)
                next_z: Optional[FSEStepZ] = None
                next_frame: Optional[ScenarioFrame] = None
                if fse_enabled:
                    assert fse_runtime is not None and fse_diag is not None and current_z is not None
                    if current_z.z_real_valid:
                        fse_runtime.append_shuffle_buffer(int(ep), int(t), current_z)
                    next_frame = fse_build_frame(
                        scenario_builder,
                        next_raw_state,
                        next_scene,
                        env.last_action,
                        episode_id=int(ep),
                        step_id=int(t + 1),
                        traffic_density=cfg.env.traffic_density,
                    )
                    next_z = fse_runtime.compute(next_frame, episode_id=int(ep), step_id=int(t + 1))
                    next_state_for_agent = make_fse_state_aug(next_raw_state, next_z, raw_state_dim, int(fse_runtime.z_dim))
                    fse_diag.record(
                        episode_id=int(ep),
                        step_id=int(t),
                        raw_state=np.asarray(raw_state, dtype=np.float32),
                        current=current_z,
                        next_info=next_z,
                        state_aug=np.asarray(state_for_agent, dtype=np.float32),
                        next_state_aug=np.asarray(next_state_for_agent, dtype=np.float32),
                        mode=str(cfg.train.mode),
                    )
                # Danger tags are training-data annotations only. They do not alter online actions.
                danger_info = classify_transition_danger(cfg, scene, next_scene, cost_dict)
                transition_meta = {
                    "episode_id": int(ep),
                    "step_in_episode": int(t),
                    "danger_label": int(danger_info["danger_label"]),
                    "is_collision": bool(danger_info["is_collision"]),
                    "is_near_danger": bool(danger_info["is_near_danger"]),
                }
                if fse_enabled:
                    assert current_z is not None and next_z is not None
                    transition_meta.update({
                        "raw_state": np.asarray(raw_state, dtype=np.float32),
                        "next_raw_state": np.asarray(next_raw_state, dtype=np.float32),
                        "z_real": np.asarray(current_z.z_real, dtype=np.float32),
                        "z_real_valid": bool(current_z.z_real_valid),
                        "z_used": np.asarray(current_z.z_used, dtype=np.float32),
                        "z_used_valid": bool(current_z.z_used_valid),
                        "z_source": str(current_z.z_source),
                        "next_z_real": np.asarray(next_z.z_real, dtype=np.float32),
                        "next_z_real_valid": bool(next_z.z_real_valid),
                        "next_z_used": np.asarray(next_z.z_used, dtype=np.float32),
                        "next_z_used_valid": bool(next_z.z_used_valid),
                        "next_z_source": str(next_z.z_source),
                        "fse_forward_executed": bool(current_z.fse_forward_executed),
                        "shuffled_source_episode_id": int(current_z.shuffled_source_episode_id),
                        "shuffled_source_step_id": int(current_z.shuffled_source_step_id),
                        "shuffled_fallback_flag": bool(current_z.shuffled_fallback_flag),
                    })
                    if is_fse_action_risk_mode(cfg.train.mode):
                        if current_frame is None:
                            raise ValueError("Action-risk replay insertion requires current ScenarioFrame.")
                        transition_meta.update({
                            "frame_tokens": np.asarray(current_frame.tokens, dtype=np.float32),
                            "frame_token_mask": np.asarray(current_frame.token_mask, dtype=np.float32),
                            "frame_entity_valid_mask": np.asarray(current_frame.entity_valid_mask, dtype=np.float32),
                            "frame_token_type_ids": np.asarray(current_frame.token_type_ids, dtype=np.int64),
                            "frame_token_role_ids": np.asarray(current_frame.token_role_ids, dtype=np.int64),
                        })
                agent.replay.add(
                    state_for_agent,
                    action,
                    reward,
                    next_state_for_agent,
                    float(done),
                    cost_dict=cost_dict,
                    transition_meta=transition_meta,
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
                    if fse_enabled and update_info:
                        fse_assert_update_finite(update_info)
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

                raw_state = next_raw_state
                state_for_agent = next_state_for_agent
                current_z = next_z
                current_frame = next_frame
                total_steps += 1
                if done:
                    break

            ep_summary = metrics.summary()
            agent.replay.mark_episode_success(int(ep), bool(ep_summary.get("success", 0)))
            train_episode_results.append(ep_summary)
            train_diag_row = build_train_diagnostic_row(ep, ep_summary, last_update_info)
            if scenario_builder is not None:
                train_diag_row.update(scenario_builder.train_log_fields())
            if fse_runtime is not None and fse_diag is not None:
                train_diag_row.update(fse_diag.fields())
                train_diag_row.update(fse_runtime.freeze_fields())
                train_diag_row["fse_forward_count"] = float(fse_runtime.fse_forward_count)
                train_diag_row["fse_random_std_clipped_dims"] = float(fse_runtime.random_std_clipped_dims)
            if action_fse_runtime is not None:
                train_diag_row.update(action_fse_runtime.freeze_fields())
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
    fse_trace_path = export_fse_z_trace(fse_diag, os.path.join(result_dir, "fse_z_trace.csv")) if fse_diag is not None else ""
    effective_features = _effective_feature_flags(cfg)
    config_payload = asdict(cfg)
    config_payload["effective_features"] = dict(effective_features)
    scenario_frame_train_probe = scenario_builder.probe_summary() if scenario_builder is not None else None
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
            "scenario_frame_schema_version": SCENARIO_SCHEMA_VERSION if scenario_builder is not None else "",
            "scenario_frame": scenario_frame_train_probe,
        },
        "train_diagnostics": train_diagnostic_rows,
        "train_diagnostics_csv_path": train_log_path,
        "train_diagnostics_json_path": train_json_path,
        "train_tensorboard_dir": tb_run_dir,
        "scenario_frame_probe_json_path": "",
        "run_result_json_path": "",
        "baseline_checkpoint_path": "",
        "fse_z_trace_csv_path": fse_trace_path,
        "resolved_config_json_path": resolved_config_path,
    }
    if fse_runtime is not None and fse_diag is not None:
        result["metadata"]["fse_rl"] = {
            "audit": fse_runtime.audit_metadata(),
            "freeze": fse_runtime.freeze_fields(),
            "diagnostics": fse_diag.fields(),
            "fse_forward_count": int(fse_runtime.fse_forward_count),
            "error_type": str(fse_runtime.error_type),
        }
        if action_fse_runtime is not None:
            result["metadata"]["fse_rl"]["action_risk"] = {
                "audit": action_fse_runtime.audit_metadata(),
                "freeze": action_fse_runtime.freeze_fields(),
                "forward_count": int(action_fse_runtime.forward_count),
                "error_type": str(action_fse_runtime.error_type),
            }
        result["metadata"]["resolved_config_json_path"] = resolved_config_path
        result["metadata"]["fse_z_trace_csv_path"] = fse_trace_path
    if str(getattr(cfg.diagnostics, "checkpoint_save_path", "")).strip():
        result["baseline_checkpoint_path"] = save_agent_checkpoint(
            agent,
            cfg,
            state_dim,
            action_dim,
            env.action_low,
            env.action_high,
            str(cfg.diagnostics.checkpoint_save_path).strip(),
            fse_runtime=fse_runtime,
        )
        result["metadata"]["baseline_checkpoint_path"] = result["baseline_checkpoint_path"]
    if cfg.eval.enabled:
        result["eval"] = evaluate_policy(cfg, agent, raw_state_dim=raw_state_dim if fse_enabled else 0, output_dir=result_dir)
    if scenario_builder is not None:
        scenario_probe_payload = {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "train": scenario_frame_train_probe,
            "eval": result.get("eval", {}).get("scenario_frame_probe") if isinstance(result.get("eval"), dict) else None,
        }
        scenario_probe_path = scenario_frame_probe_path_from_result(result_path)
        result["scenario_frame_probe_json_path"] = export_json(scenario_probe_payload, scenario_probe_path)
        result["metadata"]["scenario_frame_probe_json_path"] = result["scenario_frame_probe_json_path"]
    result["run_result_json_path"] = export_json(result, result_path)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-mode highway SAC / constrained SAC baseline experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Windows launch examples:\n"
            "  Single line:\n"
            "    python fix10E_fse_guided_paper_csac.py --modes paper_sac_pure,paper_csac_lagrangian --traffic-density sparse --episodes 500 --max-steps 400\n"
            "  cmd.exe multiline (caret ^):\n"
            "    python fix10E_fse_guided_paper_csac.py ^\n"
            "      --modes paper_sac_pure,paper_csac_lagrangian ^\n"
            "      --penalty-impl stage4_constrained_recurrent_v1 ^\n"
            "      --upgrade-stage m1m2 ^\n"
            "      --episodes 500 ^\n"
            "      --max-steps 200\n"
            "  PowerShell multiline (backtick `):\n"
            "    python fix10E_fse_guided_paper_csac.py `\n"
            "      --modes paper_sac_pure,paper_csac_lagrangian `\n"
            "      --penalty-impl stage4_constrained_recurrent_v1 `\n"
            "      --upgrade-stage m1m2 `\n"
            "      --episodes 500 `\n"
            "      --max-steps 200\n"
            "  Formal 5-seed CSAC baseline (uses public default paths):\n"
            "    python fix10E_fse_guided_paper_csac.py ^\n"
            "      --mode paper_csac_lagrangian ^\n"
            "      --traffic-density sparse ^\n"
            "      --seeds 40-44 ^\n"
            "      --episodes 500 ^\n"
            "      --max-steps 400 ^\n"
            "      --eval-episodes 20 ^\n"
            "      --device cuda:0"
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
    parser.add_argument("--mode", type=str, default="paper_csac_lagrangian", choices=ALL_MODES)
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
    scenario_frame_group = parser.add_mutually_exclusive_group()
    scenario_frame_group.add_argument("--enable-scenario-frame", dest="enable_scenario_frame", action="store_true", default=None)
    scenario_frame_group.add_argument("--disable-scenario-frame", dest="enable_scenario_frame", action="store_false")
    parser.add_argument("--scenario-frame-debug", action="store_true",
                        help="Print detailed ScenarioFrame probes for debugging.")
    parser.add_argument("--scenario-frame-probe-interval", type=int, default=0,
                        help="Print detailed ScenarioFrame probes every N environment steps when N > 0.")
    parser.add_argument("--train-log-path", type=str, default=PUBLIC_BASELINE_TRAIN_LOG_PATH)
    parser.add_argument("--train-json-path", type=str, default=PUBLIC_BASELINE_TRAIN_JSON_PATH)
    parser.add_argument("--run-result-json-path", type=str, default=PUBLIC_BASELINE_RUN_RESULT_JSON_PATH)
    parser.add_argument("--checkpoint-save-path", type=str, default=PUBLIC_BASELINE_CHECKPOINT_PATH,
                        help="Save a baseline SAC/CSAC checkpoint after RL training.")
    parser.add_argument("--multi-seed-summary-path", type=str, default="")
    parser.add_argument("--tensorboard-dir", type=str, default=PUBLIC_BASELINE_TENSORBOARD_DIR)
    parser.add_argument("--tensorboard-flush-secs", type=int, default=10)
    parser.add_argument("--fse-task", type=str, default="none", choices=FSE_TASKS,
                        help="Run the module-2 FSE workflow instead of RL when not 'none'.")
    parser.add_argument("--fse-collect-policy", type=str, default="online_train", choices=FSE_COLLECT_POLICIES,
                        help="FSE raw trajectory collection policy.")
    parser.add_argument("--policy-checkpoint-path", type=str, default="",
                        help="Fixed baseline checkpoint path for eval_policy/noisy_eval collection.")
    parser.add_argument("--policy-checkpoint-path-template", type=str, default="",
                        help="Baseline checkpoint path template; supports {seed}.")
    parser.add_argument("--policy-checkpoint-map", type=str, default="",
                        help="Comma separated env-seed to checkpoint map, e.g. seed140=path40.pt,seed141=path41.pt.")
    parser.add_argument("--allow-untrained-eval-policy", action="store_true",
                        help="Debug only: allow eval/noisy collection without a trained checkpoint outside csac_main.")
    parser.add_argument("--fse-source-ratio", type=str, default="online_train=0.65,eval_policy=0.25,noisy_eval=0.10",
                        help="Target training batch source ratio.")
    parser.add_argument("--fse-source-balanced-sampler", action="store_true",
                        help="Use source-aware sampling during FSE training.")
    parser.add_argument("--fse-noise-std-acc", type=float, default=0.15)
    parser.add_argument("--fse-noise-std-steer", type=float, default=0.05)
    parser.add_argument("--fse-noise-clip", type=float, default=0.30)
    parser.add_argument("--fse-split-stratify-by", type=str, default="collect_policy,traffic_density,mode")
    parser.add_argument("--fse-raw-path", type=str, default="",
                        help="Comma separated raw trajectory .npz path(s) for build-dataset.")
    parser.add_argument("--fse-dataset-path", type=str, default="",
                        help="Path to the Step-2 FSE labeled .npz dataset.")
    parser.add_argument("--fse-output-dir", type=str, default="results/fse",
                        help="Directory for FSE logs, metrics, smoke output, and default checkpoint.")
    parser.add_argument("--fse-checkpoint-path", type=str, default="",
                        help="Checkpoint path for FSE eval, or explicit save path for FSE train.")
    parser.add_argument("--fse-action-conditioned", action="store_true",
                        help="Train/evaluate an action-conditioned FSE model when dataset actions are present.")
    parser.add_argument("--fse-action-risk-checkpoint-path", type=str, default="",
                        help="Action-conditioned FSE checkpoint for Module-6 actor risk regularization.")
    parser.add_argument("--fse-action-risk-acceptance-json", type=str, default="",
                        help="Acceptance gate JSON for the action-conditioned FSE checkpoint.")
    parser.add_argument("--strict-fse-action-gate", action="store_true",
                        help="Treat action-FSE gate warnings as blocking errors.")
    parser.add_argument("--fse-action-risk-beta", type=float, default=0.05)
    parser.add_argument("--fse-action-risk-uncertainty-kappa", type=float, default=2.0)
    parser.add_argument("--fse-action-risk-grad-diag-interval", type=int, default=50)
    parser.add_argument("--fse-action-risk-active-threshold", type=float, default=0.05)
    parser.add_argument("--fse-action-risk-gate-active-threshold", type=float, default=0.10)
    parser.add_argument("--fse-fusion-mode", type=str, default=None, choices=FSE_FUSION_MODES)
    parser.add_argument("--fse-z-mode", type=str, default=None, choices=FSE_Z_MODES)
    parser.add_argument("--fse-random-z-distribution", type=str, default="global_empirical_matched", choices=FSE_RANDOM_Z_DISTRIBUTIONS)
    parser.add_argument("--fse-random-z-std-floor", type=float, default=1e-4)
    parser.add_argument("--fse-run-tier", type=str, default="smoke", choices=FSE_RUN_TIERS)
    parser.add_argument("--allow-legacy-fse-normalization", action="store_true")
    parser.add_argument("--fse-z-global-stats-path", type=str, default="")
    parser.add_argument("--fse-shuffle-pool-path", type=str, default="")
    parser.add_argument("--fse-z-artifacts", type=str, default="both", choices=FSE_Z_ARTIFACT_TARGETS,
                        help="Artifact target for --fse-task build-z-artifacts.")
    parser.add_argument("--fse-z-artifact-source-split", type=str, default="train_val", choices=FSE_Z_ARTIFACT_SOURCE_SPLITS,
                        help="Episode-level split used to build formal z artifacts; test/final splits are intentionally unavailable.")
    parser.add_argument("--fse-z-stats-output-path", type=str, default="",
                        help="Output path for fse_z_global_stats.json; defaults to fse_output_dir/fse_z_global_stats.json.")
    parser.add_argument("--fse-shuffle-pool-output-path", type=str, default="",
                        help="Output path for fse_shuffle_pool.npz; defaults to fse_output_dir/fse_shuffle_pool.npz.")
    parser.add_argument("--fse-z-artifact-max-samples", type=int, default=0,
                        help="Deterministic cap for build-z-artifacts; 0 uses all selected rows.")
    parser.add_argument("--fse-z-artifact-min-samples", type=int, default=32,
                        help="Minimum selected rows required for build-z-artifacts.")
    parser.add_argument("--fse-z-trace-stride", type=int, default=1)
    parser.add_argument("--fse-z-trace-max-rows", type=int, default=200000)
    parser.add_argument("--fse-epochs", type=int, default=None)
    parser.add_argument("--fse-batch-size", type=int, default=None)
    parser.add_argument("--fse-lr", type=float, default=None)
    parser.add_argument("--fse-weight-decay", type=float, default=None)
    parser.add_argument("--fse-train-split", type=float, default=None)
    parser.add_argument("--fse-val-split", type=float, default=None)
    parser.add_argument("--fse-test-split", type=float, default=None)
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
    if getattr(args, "enable_scenario_frame", None) is not None:
        cfg.train.enable_scenario_frame = bool(args.enable_scenario_frame)
    cfg.train.scenario_frame_debug = bool(getattr(args, "scenario_frame_debug", False))
    cfg.train.scenario_frame_probe_interval = max(0, int(getattr(args, "scenario_frame_probe_interval", 0)))
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
    cfg.diagnostics.checkpoint_save_path = str(getattr(args, "checkpoint_save_path", "")).strip()
    cfg.diagnostics.tensorboard_log_dir = str(args.tensorboard_dir).strip()
    cfg.diagnostics.tensorboard_flush_secs = max(1, int(args.tensorboard_flush_secs))
    cfg.fse.task = str(getattr(args, "fse_task", "none")).lower().strip()
    cfg.fse.collect_policy = str(getattr(args, "fse_collect_policy", "online_train")).lower().strip()
    cfg.fse.raw_path = str(getattr(args, "fse_raw_path", "")).strip()
    cfg.fse.dataset_path = str(getattr(args, "fse_dataset_path", "")).strip()
    cfg.fse.output_dir = str(getattr(args, "fse_output_dir", "results/fse")).strip() or "results/fse"
    cfg.fse.checkpoint_path = str(getattr(args, "fse_checkpoint_path", "")).strip()
    cfg.fse.action_conditioned = bool(getattr(args, "fse_action_conditioned", False))
    cfg.fse.action_risk_checkpoint_path = str(getattr(args, "fse_action_risk_checkpoint_path", "")).strip()
    cfg.fse.action_risk_acceptance_json = str(getattr(args, "fse_action_risk_acceptance_json", "")).strip()
    cfg.fse.strict_action_gate = bool(getattr(args, "strict_fse_action_gate", False))
    cfg.fse.action_risk_beta = max(0.0, float(getattr(args, "fse_action_risk_beta", cfg.fse.action_risk_beta)))
    cfg.fse.action_risk_uncertainty_kappa = max(0.0, float(getattr(args, "fse_action_risk_uncertainty_kappa", cfg.fse.action_risk_uncertainty_kappa)))
    cfg.fse.action_risk_grad_diag_interval = max(1, int(getattr(args, "fse_action_risk_grad_diag_interval", cfg.fse.action_risk_grad_diag_interval)))
    cfg.fse.action_risk_active_threshold = float(np.clip(getattr(args, "fse_action_risk_active_threshold", cfg.fse.action_risk_active_threshold), 0.0, 1.0))
    cfg.fse.action_risk_gate_active_threshold = float(np.clip(getattr(args, "fse_action_risk_gate_active_threshold", cfg.fse.action_risk_gate_active_threshold), 0.0, 1.0))
    explicit_fusion_mode = getattr(args, "fse_fusion_mode", None)
    explicit_z_mode = getattr(args, "fse_z_mode", None)
    setattr(cfg.fse, "_user_explicit_fusion_mode", explicit_fusion_mode is not None)
    setattr(cfg.fse, "_user_explicit_z_mode", explicit_z_mode is not None)
    cfg.fse.fusion_mode = str(explicit_fusion_mode if explicit_fusion_mode is not None else cfg.fse.fusion_mode).lower().strip()
    cfg.fse.z_mode = str(explicit_z_mode if explicit_z_mode is not None else cfg.fse.z_mode).lower().strip()
    cfg.fse.random_z_distribution = str(getattr(args, "fse_random_z_distribution", cfg.fse.random_z_distribution)).lower().strip()
    cfg.fse.random_z_std_floor = max(0.0, float(getattr(args, "fse_random_z_std_floor", cfg.fse.random_z_std_floor)))
    cfg.fse.run_tier = str(getattr(args, "fse_run_tier", cfg.fse.run_tier)).lower().strip()
    cfg.fse.allow_legacy_normalization = bool(getattr(args, "allow_legacy_fse_normalization", False))
    cfg.fse.z_global_stats_path = str(getattr(args, "fse_z_global_stats_path", "")).strip()
    cfg.fse.shuffle_pool_path = str(getattr(args, "fse_shuffle_pool_path", "")).strip()
    cfg.fse.z_artifacts = str(getattr(args, "fse_z_artifacts", cfg.fse.z_artifacts)).lower().strip()
    cfg.fse.z_artifact_source_split = str(getattr(args, "fse_z_artifact_source_split", cfg.fse.z_artifact_source_split)).lower().strip()
    cfg.fse.z_stats_output_path = str(getattr(args, "fse_z_stats_output_path", "")).strip()
    cfg.fse.shuffle_pool_output_path = str(getattr(args, "fse_shuffle_pool_output_path", "")).strip()
    cfg.fse.z_artifact_max_samples = max(0, int(getattr(args, "fse_z_artifact_max_samples", cfg.fse.z_artifact_max_samples)))
    cfg.fse.z_artifact_min_samples = max(1, int(getattr(args, "fse_z_artifact_min_samples", cfg.fse.z_artifact_min_samples)))
    cfg.fse.formal_min_z_artifact_samples = int(cfg.fse.z_artifact_min_samples)
    cfg.fse.z_trace_stride = max(1, int(getattr(args, "fse_z_trace_stride", cfg.fse.z_trace_stride)))
    cfg.fse.z_trace_max_rows = max(0, int(getattr(args, "fse_z_trace_max_rows", cfg.fse.z_trace_max_rows)))
    cfg.fse.policy_checkpoint_path = str(getattr(args, "policy_checkpoint_path", "")).strip()
    cfg.fse.policy_checkpoint_path_template = str(getattr(args, "policy_checkpoint_path_template", "")).strip()
    cfg.fse.policy_checkpoint_map = str(getattr(args, "policy_checkpoint_map", "")).strip()
    cfg.fse.allow_untrained_eval_policy = bool(getattr(args, "allow_untrained_eval_policy", False))
    cfg.fse.source_ratio = str(getattr(args, "fse_source_ratio", cfg.fse.source_ratio)).strip()
    cfg.fse.source_balanced_sampler = bool(getattr(args, "fse_source_balanced_sampler", False))
    cfg.fse.noise_std_acc = max(0.0, float(getattr(args, "fse_noise_std_acc", cfg.fse.noise_std_acc)))
    cfg.fse.noise_std_steer = max(0.0, float(getattr(args, "fse_noise_std_steer", cfg.fse.noise_std_steer)))
    cfg.fse.noise_clip = max(0.0, float(getattr(args, "fse_noise_clip", cfg.fse.noise_clip)))
    cfg.fse.split_stratify_by = str(getattr(args, "fse_split_stratify_by", cfg.fse.split_stratify_by)).strip()
    if getattr(args, "fse_epochs", None) is not None:
        cfg.fse.epochs = max(1, int(args.fse_epochs))
    if getattr(args, "fse_batch_size", None) is not None:
        cfg.fse.batch_size = max(1, int(args.fse_batch_size))
    if getattr(args, "fse_lr", None) is not None:
        cfg.fse.lr = max(0.0, float(args.fse_lr))
    if getattr(args, "fse_weight_decay", None) is not None:
        cfg.fse.weight_decay = max(0.0, float(args.fse_weight_decay))
    if getattr(args, "fse_train_split", None) is not None:
        cfg.fse.train_split = float(np.clip(args.fse_train_split, 0.0, 1.0))
    if getattr(args, "fse_val_split", None) is not None:
        cfg.fse.val_split = float(np.clip(args.fse_val_split, 0.0, 1.0))
    if getattr(args, "fse_test_split", None) is not None:
        cfg.fse.test_split = float(np.clip(args.fse_test_split, 0.0, 1.0))
    if args.batch_size is not None:
        cfg.sac.batch_size = max(1, int(args.batch_size))
        cfg.fse.batch_size = max(1, int(args.batch_size))
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
    if cfg.fse.task not in FSE_TASKS:
        raise ValueError(f"Unsupported fse_task='{cfg.fse.task}'. Choose from {FSE_TASKS}.")
    if cfg.fse.collect_policy not in FSE_COLLECT_POLICIES:
        raise ValueError(f"Unsupported fse_collect_policy='{cfg.fse.collect_policy}'. Choose from {FSE_COLLECT_POLICIES}.")
    if cfg.fse.z_artifacts not in FSE_Z_ARTIFACT_TARGETS:
        raise ValueError(f"Unsupported fse_z_artifacts='{cfg.fse.z_artifacts}'. Choose from {FSE_Z_ARTIFACT_TARGETS}.")
    if cfg.fse.z_artifact_source_split not in FSE_Z_ARTIFACT_SOURCE_SPLITS:
        raise ValueError(
            f"Unsupported fse_z_artifact_source_split='{cfg.fse.z_artifact_source_split}'. "
            f"Choose from {FSE_Z_ARTIFACT_SOURCE_SPLITS}."
        )
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
    if str(getattr(parsed_args, "modes", "")).strip() and str(getattr(parsed_args, "mode", "")).strip():
        if str(parsed_args.mode).lower().strip() not in requested_modes:
            print(f"[WARN] --modes={requested_modes} overrides --mode={parsed_args.mode}.")
    if str(config.fse.task).lower().strip() != "none":
        result = run_fse_task(config, modes=requested_modes, seeds=requested_seeds)
        print(json.dumps(_jsonable_plain(result), indent=2, ensure_ascii=False))
        sys.exit(0)
    if len(requested_modes) == 1:
        selected_mode = requested_modes[0]
        configure_mode_behavior(config, selected_mode)
        _apply_mode_specific_paths(config, selected_mode, append_when_missing=False)
    if len(requested_modes) == 1 and len(requested_seeds) == 1:
        _apply_seed_specific_paths(config, requested_seeds[0], append_when_missing=False)
        run(config)
    elif len(requested_modes) == 1:
        print(f"[MULTI-SEED] requested mode: {requested_modes[0]}")
        print(f"[MULTI-SEED] requested seeds: {requested_seeds}")
        run_multi_seed(
            base_cfg=config,
            seeds=requested_seeds,
            summary_path=str(getattr(parsed_args, "multi_seed_summary_path", "")).strip(),
        )
    else:
        print(f"[MULTI-MODE] requested modes: {requested_modes}")
        print(f"[MULTI-MODE] requested seeds: {requested_seeds}")
        run_multi_mode_seed(
            base_cfg=config,
            modes=requested_modes,
            seeds=requested_seeds,
            summary_path=str(getattr(parsed_args, "multi_seed_summary_path", "")).strip(),
        )
