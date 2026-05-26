# FSE 离线混合数据集训练闭环计划：CSAC 主线最终实施版

## Summary

- 主线数据集固定为 **CSAC-only baseline**：
  ```text
  CSAC online_train + CSAC eval_policy + CSAC noisy_eval
  ```
  训练 batch source ratio 为 `online_train=0.65, eval_policy=0.25, noisy_eval=0.10`。
- `random` 和 SAC 数据只用于辅助消融，必须输出到独立目录，不进入 `csac_main` 主数据集。
- Stage 0 单独训练 baseline CSAC checkpoint；online_train collect 重新训练并采集在线训练分布。二者关系写入 metadata。
- 模块二边界不变：FSE 不影响在线 action，不接入 CSAC actor/critic，不修改 reward/cost/done/env.step，不做 joint fine-tuning。

## Key Changes

- Add/extend CLI:
  ```text
  --checkpoint-save-path
  --fse-task {none,collect,build-dataset,train,eval,full,smoke}
  --fse-collect-policy {online_train,eval_policy,noisy_eval,random}
  --policy-checkpoint-path
  --policy-checkpoint-path-template
  --policy-checkpoint-map
  --allow-untrained-eval-policy
  --fse-source-ratio online_train=0.65,eval_policy=0.25,noisy_eval=0.10
  --fse-source-balanced-sampler
  --fse-noise-std-acc
  --fse-noise-std-steer
  --fse-noise-clip
  --fse-split-stratify-by collect_policy,traffic_density,mode
  --fse-raw-path / --fse-dataset-path / --fse-output-dir / --fse-checkpoint-path
  --fse-epochs / --fse-batch-size / --fse-lr / --fse-weight-decay
  ```

- Stage 0 baseline checkpoint:
  - Train unmodified baseline CSAC and save per seed:
    ```text
    results/baseline_csac/{seed}/checkpoint.pt
    results/baseline_csac/{seed}/run_result.json
    results/baseline_csac/{seed}/train_log.csv
    ```
  - These checkpoints are used only for `eval_policy/noisy_eval`.
  - `online_train` FSE collect is a separate online-training data run; metadata records:
    ```text
    online_train_training_run_id
    eval_noisy_checkpoint_run_id
    ```

- `full` task:
  - `--fse-task full` is only for single-source smoke/simple runs.
  - Formal three-source CSAC datasets must be run in stages.
  - If `full` is asked to combine multiple collect sources, it must fail with a clear error.

## Raw/Dataset Contract

- Required raw arrays:
  ```text
  tokens, token_mask, entity_valid_mask, token_type_ids, token_role_ids
  episode_uid_id, episode_id_local, step_id, mode_id, seed_id, raw_source_id
  collect_policy_id, source_group_id, policy_checkpoint_uid_id
  action, reward, done, done_reason_id, terminated, truncated, timeout, goal_reached, success
  ego_x, ego_y, ego_speed, ego_lane_id, lane_offset, road_boundary_margin, lane_width
  front_vehicle_valid, front_distance, front_rel_speed, front_ttc
  collision, crashed, offroad
  cost_collision, cost_headway, cost_lane, cost_speed, cost_total, cost_safety
  ```

- Source group ids:
  ```text
  MAIN_CSAC = 0
  AUX_RANDOM_OOD = 1
  AUX_SAC_RISK = 2
  MIXED_ABLATION = 3
  ```

- Checkpoint uid:
  ```text
  policy_checkpoint_uid_id = 0 reserved for NO_CHECKPOINT
  real checkpoints start at 1
  ```
  Metadata stores `policy_checkpoint_uid_mapping`.

- `cost_safety` definition:
  ```text
  cost_safety = max(cost_collision, cost_headway, cost_lane)
  ```
  `cost_speed` and comfort are excluded from FSE safety regression. Merge rejects mismatched definitions.

- Noisy action:
  - `action = [acceleration, steering]`.
  - Noise is added in normalized agent action space and clipped to `action_space.low/high`.
  - Save `action_space_low`, `action_space_high`, `noise_space="normalized_action"`.

## Labeling Metadata And Terminal Rules

- Label thresholds and horizon metadata must be saved to `fse_dataset_meta.json`, `fse_label_distribution.json`, and `fse_checkpoint.pt`:
  ```text
  horizons = [10, 20, 40]
  env_dt = 0.05
  horizon_seconds = [0.5, 1.0, 2.0]
  unsafe_headway_tau = 1.2
  low_ttc_tau = 4.0
  lane_boundary_margin_threshold = 0.3
  lane_offset_threshold = 0.45 * lane_width
  ```

- Terminal rule:
  - Failure if `collision/crashed/offroad=True` or `done_reason_id in {DONE_COLLISION,DONE_OFFROAD}`.
  - Normal truncation if `done_reason_id in {DONE_TIMEOUT,DONE_GOAL,DONE_OTHER}` and no failure flags.
  - `DONE_OTHER` statistics are mandatory:
    ```text
    done_other_count
    done_other_with_failure_flags_count
    done_other_without_failure_flags_count
    ```
  - Warning if `done_other_count / done_count > 0.10`.

- Dense/sparse scope:
  - First main closure runs `sparse`.
  - `dense` is second-round generalization or mixed-density extension.
  - Sparse/dense must not be silently mixed; `traffic_density_id` and density distribution are reported in source/split metrics.

## Build, Split, And Quality Gates

- `build-dataset`:
  - Merges raw files only after schema/name/order/action/cost/checkpoint/source-group checks.
  - Rejects auxiliary random/SAC raw files when output path/source group is `csac_main`, unless an explicit ablation flag is used.
  - Does not accept `--fse-source-balanced-sampler`; that flag is train-only.

- Split:
  - Merge first, then stratified split by `collect_policy_id`; if available also by `traffic_density_id` and `mode_id`.
  - Split uses `episode_uid_id`, never local episode id.
  - Recommended env seed ranges:
    ```text
    online_train: 40-44
    eval_policy: 140-144
    noisy_eval: 240-244
    ```

- Data quality gates:
  ```text
  num_episodes >= 50
  valid_rate_H20 >= 0.70
  valid_rate_H40 >= 0.50
  low_ttc_positive_rate_H20 > 0
  unsafe_headway_positive_rate_H20 > 0
  ```
  Source-level warnings:
  - online_train must contain low_ttc or unsafe_headway support.
  - noisy_eval should increase major-risk support over eval_policy.
  - eval_policy collision/offroad/low_ttc support may be 0; warning only.

## Training, Sampling, And Eval

- Train uses existing FSE bottleneck model/loss.
- Source ratio semantics:
  - `target_source_ratio` is **training batch sample-level ratio**.
  - `actual_source_ratio` is reported by samples and episodes.
- With `--fse-source-balanced-sampler`:
  - sample source by target ratio;
  - within each source, use risk-priority sampling;
  - if source major-risk support is insufficient, sample with replacement inside source;
  - if source total support is insufficient, fallback to global source pool;
  - log `source_fallback_count` and `risk_fallback_count`.
- `random` source, if used, is capped at `<=5%` at sampler level.

- Eval metrics:
  ```text
  metrics_all_mixed
  metrics_by_collect_policy.online_train
  metrics_by_collect_policy.eval_policy
  metrics_by_collect_policy.noisy_eval
  metrics_eval_policy_only
  train_source_ratio_actual
  train_batch_source_ratio_mean
  eval_source_ratio_natural
  ```
  Val/test use natural distribution only.

- Best checkpoint uses validation composite score:
  ```text
  0.35 * macro_F1_low_ttc_headway
  + 0.30 * mean_PR_AUC_collision_offroad
  + 0.15 * recall_collision_offroad
  + 0.10 * (1 - clip(ECE, 0, 1))
  + 0.10 * (1 - clip(monotonic_violation_rate, 0, 1))
  ```
  NaN rare-event components are skipped; all-NaN component becomes 0. Tie-breaker: lower ECE, lower Brier, lower val loss.

## Output Layout

```text
results/fse_bottleneck/csac_main/
results/fse_bottleneck/sac_aux_risk/
results/fse_bottleneck/mixed_ablation/
```

Main artifacts:
```text
fse_raw_trajectories.npz
fse_raw_episodes.jsonl
scenario_frame_probe.json
fse_dataset.npz
fse_dataset_meta.json
fse_label_distribution.json
fse_source_distribution.json
fse_split.json
fse_merge_report.json
fse_label_debug_sample.jsonl
fse_checkpoint.pt
fse_train_log.csv
fse_eval_metrics.json
fse_z_stats.csv
```

## Test Plan

- Stage 0 saves per-seed checkpoint at `results/baseline_csac/{seed}/checkpoint.pt`.
- `eval_policy/noisy_eval` fail if checkpoint map/path is missing.
- Build command rejects `--fse-source-balanced-sampler`.
- `DONE_OTHER` stats and warning are produced.
- Label metadata thresholds/horizon seconds are present in dataset meta and checkpoint.
- Source-balanced sampler logs source and risk fallback counts.
- Eval metrics are reported for mixed and per-source subsets.
- Formal three-source experiment cannot use `--fse-task full`.
- `--fse-task none` preserves existing RL behavior.

## Recommended Commands

Stage 0, baseline CSAC checkpoints:
```bat
python fix10E_fse_guided_paper_csac.py ^
  --mode paper_csac_lagrangian ^
  --seeds 40-44 ^
  --traffic-density sparse ^
  --episodes 300 ^
  --max-steps 400 ^
  --device cuda:0 ^
  --train-log-path results/baseline_csac/{seed}/train_log.csv ^
  --run-result-json-path results/baseline_csac/{seed}/run_result.json ^
  --checkpoint-save-path results/baseline_csac/{seed}/checkpoint.pt ^
  --multi-seed-summary-path results/baseline_csac/multi_seed_summary.json
```

CSAC online_train:
```bat
python fix10E_fse_guided_paper_csac.py ^
  --fse-task collect ^
  --fse-collect-policy online_train ^
  --modes paper_csac_lagrangian ^
  --seeds 40-44 ^
  --traffic-density sparse ^
  --episodes 300 ^
  --max-steps 400 ^
  --device cuda:0 ^
  --enable-scenario-frame ^
  --fse-output-dir results/fse_bottleneck/csac_main/collect_online_sparse
```

CSAC eval_policy:
```bat
python fix10E_fse_guided_paper_csac.py ^
  --fse-task collect ^
  --fse-collect-policy eval_policy ^
  --policy-checkpoint-map seed140=results/baseline_csac/40/checkpoint.pt,seed141=results/baseline_csac/41/checkpoint.pt,seed142=results/baseline_csac/42/checkpoint.pt,seed143=results/baseline_csac/43/checkpoint.pt,seed144=results/baseline_csac/44/checkpoint.pt ^
  --modes paper_csac_lagrangian ^
  --seeds 140-144 ^
  --traffic-density sparse ^
  --episodes 100 ^
  --max-steps 400 ^
  --device cuda:0 ^
  --enable-scenario-frame ^
  --fse-output-dir results/fse_bottleneck/csac_main/collect_eval_sparse
```

CSAC noisy_eval:
```bat
python fix10E_fse_guided_paper_csac.py ^
  --fse-task collect ^
  --fse-collect-policy noisy_eval ^
  --policy-checkpoint-map seed240=results/baseline_csac/40/checkpoint.pt,seed241=results/baseline_csac/41/checkpoint.pt,seed242=results/baseline_csac/42/checkpoint.pt,seed243=results/baseline_csac/43/checkpoint.pt,seed244=results/baseline_csac/44/checkpoint.pt ^
  --fse-noise-std-acc 0.15 ^
  --fse-noise-std-steer 0.05 ^
  --fse-noise-clip 0.30 ^
  --modes paper_csac_lagrangian ^
  --seeds 240-244 ^
  --traffic-density sparse ^
  --episodes 60 ^
  --max-steps 400 ^
  --device cuda:0 ^
  --enable-scenario-frame ^
  --fse-output-dir results/fse_bottleneck/csac_main/collect_noisy_sparse
```

Build:
```bat
python fix10E_fse_guided_paper_csac.py ^
  --fse-task build-dataset ^
  --fse-raw-path results/fse_bottleneck/csac_main/collect_online_sparse/fse_raw_trajectories.npz,results/fse_bottleneck/csac_main/collect_eval_sparse/fse_raw_trajectories.npz,results/fse_bottleneck/csac_main/collect_noisy_sparse/fse_raw_trajectories.npz ^
  --fse-output-dir results/fse_bottleneck/csac_main/dataset_mixed ^
  --fse-source-ratio online_train=0.65,eval_policy=0.25,noisy_eval=0.10 ^
  --fse-split-stratify-by collect_policy,traffic_density,mode
```

Train:
```bat
python fix10E_fse_guided_paper_csac.py ^
  --fse-task train ^
  --fse-dataset-path results/fse_bottleneck/csac_main/dataset_mixed/fse_dataset.npz ^
  --fse-output-dir results/fse_bottleneck/csac_main/train_mixed ^
  --fse-epochs 80 ^
  --fse-batch-size 256 ^
  --fse-lr 3e-4 ^
  --fse-weight-decay 1e-4 ^
  --fse-source-ratio online_train=0.65,eval_policy=0.25,noisy_eval=0.10 ^
  --fse-source-balanced-sampler ^
  --device cuda:0
```

Eval:
```bat
python fix10E_fse_guided_paper_csac.py ^
  --fse-task eval ^
  --fse-dataset-path results/fse_bottleneck/csac_main/dataset_mixed/fse_dataset.npz ^
  --fse-checkpoint-path results/fse_bottleneck/csac_main/train_mixed/fse_checkpoint.pt ^
  --fse-output-dir results/fse_bottleneck/csac_main/eval_mixed ^
  --device cuda:0
```

## Assumptions

- Current code must add explicit baseline checkpoint save/load before `eval_policy/noisy_eval` can be meaningful.
- Stage 0 checkpoint run and online_train collect run are intentionally separate for reproducibility.
- First official closure is sparse; dense is second-round generalization or mixed-density extension.
- Module two claims remain limited to offline future-risk prediction and `z_fse` representation readiness.
