# Build Formal Z Artifacts For Stage G.1b

## Summary
新增 `--fse-task build-z-artifacts`，用已有 FSE checkpoint 与 labeled FSE dataset 生成 formal random/shuffled 消融所需 artifact：

- `fse_z_global_stats.json`：供 `paper_csac_fse_z_gated_random --fse-run-tier formal` 使用。
- `fse_shuffle_pool.npz`：供 `paper_csac_fse_z_gated_shuffled --fse-run-tier formal` 使用。

默认使用 episode-level `train_val` split，不接触 test/final mixed test；生成后立即用 `FrozenFSEBottleneckRuntime` 做 formal 自检。

## Key Changes
- 扩展 CLI / config / task dispatch：
  - `FSE_TASKS` 增加 `build-z-artifacts`。
  - `FSEConfig` 增加 `z_artifacts`、`z_artifact_source_split`、`z_stats_output_path`、`shuffle_pool_output_path`、`z_artifact_max_samples`、`z_artifact_min_samples`。
  - `parse_args()` 增加对应参数，`apply_args()` 写入 `cfg.fse`。
  - `run_fse_task()` 增加 `build-z-artifacts -> run_fse_build_z_artifacts(cfg)`。
- 新增参数：
  - `--fse-z-artifacts {both,stats,pool}`，默认 `both`。
  - `--fse-z-artifact-source-split {train,val,train_val}`，默认 `train_val`。
  - `--fse-z-stats-output-path`，默认 `$fse_output_dir/fse_z_global_stats.json`。
  - `--fse-shuffle-pool-output-path`，默认 `$fse_output_dir/fse_shuffle_pool.npz`。
  - `--fse-z-artifact-max-samples`，默认 `0` 表示全量。
  - `--fse-z-artifact-min-samples`，默认 `32`；Stage G.1b 正式建议传 `1000`。
- split 策略：
  - formal 默认和推荐只使用 `train_val`。
  - `train/val` 作为 debug 选项；formal 下使用非 `train_val` 时写入 warning，并在 result JSON 中显式标记。
  - 不提供 `test` 选项。
- 新增核心函数：
  - `run_fse_build_z_artifacts(cfg)`：读取 dataset/checkpoint、split、收集 z、保存 artifact、自检、输出 `build_z_artifacts_result.json`。
  - `collect_fse_z_for_indices(...)`：batch frozen forward，返回 `z/source_index/episode_id/step_id`。
  - `validate_built_z_artifacts(cfg, stats_path, pool_path)`：deep-copy cfg，不污染原配置；设置临时 mode 后必须调用 `configure_mode_behavior()` 同步 `cfg.fse.z_mode/fusion_mode`。
- checkpoint 校验至少覆盖：
  - `scenario_frame_schema_version` 匹配。
  - `z_dim=64`、`token_dim=24`、`n_tokens=12`。
  - `horizons=[10,20,40]`。
  - `risk_names == FSE_RISK_NAMES`。
  - `action_conditioned=False`。
- frozen forward：
  - `model.eval()`。
  - 所有 FSE 参数 `requires_grad=False`。
  - 全程 `torch.no_grad()`。
  - 若 `max_samples > 0`，使用 `np.random.default_rng(cfg.train.seed)` deterministic subsample。
  - `source_index` 必须保存原始 dataset row index，不是 pool 内局部编号。
- `fse_z_global_stats.json` 写入：
  - 原始 `mu_z/std_z`，shape `[64]`；不在 artifact 中覆盖为 clipped std。
  - 标准 metadata：`num_samples/source_dataset/source_split/checkpoint_sha256/scenario_builder_config_hash/traffic_density`。
  - 审计 metadata：`source_dataset_abs_path/source_dataset_sha256/source_dataset_exists_at_build/z_dim/token_dim/n_tokens/horizons/risk_names/seed/split_seed/split_stratify_by/generated_at_utc/artifact_version`。
  - std 摘要：`raw_std_min/raw_std_max/raw_std_mean/num_std_below_default_floor_1e-4/recommended_std_floor`。
- `fse_shuffle_pool.npz` 写入：
  - `z` shape `[N,64]`。
  - `episode_id`：优先 `episode_uid_id`，否则 `episode_id`。
  - `step_id/source_index`。
  - `metadata_json` 为完整 metadata JSON string。
  - 同时双写 `num_samples/source_dataset/source_split/checkpoint_sha256/scenario_builder_config_hash/traffic_density` 等单独字段。
  - metadata 增加 z 分布摘要：`z_norm_mean/z_norm_std/z_norm_p10/z_norm_p50/z_norm_p90/z_batch_variance_mean`。
- formal 自检：
  - random 自检：deep-copy cfg，设 `mode=paper_csac_fse_z_gated_random`，调用 `configure_mode_behavior()`，设 `run_tier=formal`、`z_global_stats_path=stats_path`，实例化 runtime。
  - shuffled 自检：deep-copy cfg，设 `mode=paper_csac_fse_z_gated_shuffled`，调用 `configure_mode_behavior()`，设 `run_tier=formal`、`z_global_stats_path=stats_path`、`shuffle_pool_path=pool_path`，实例化 runtime。
  - 自检后显式检查 runtime 回读 hash/metadata 与生成值一致：artifact hash、`source_split`、`checkpoint_sha256`、`scenario_builder_config_hash`、`traffic_density`。
- `build_z_artifacts_result.json` 记录：
  - stats/pool path、hash、sample count、自检状态。
  - `selected_index_count/selected_episode_count/selected_step_min/selected_step_max`。
  - `source_index_sha256/episode_id_sha256`。
  - dataset path、dataset hash、checkpoint hash、builder hash、warnings。

## Commands
生成 artifacts：

```bat
python %SCRIPT% ^
  --fse-task build-z-artifacts ^
  --traffic-density %DENSITY% ^
  --seed 42 ^
  --device %DEVICE% ^
  --max-steps 400 ^
  --fse-dataset-path "%FSE_DATASET%" ^
  --fse-checkpoint-path "%FSE_CKPT%" ^
  --fse-run-tier formal ^
  --fse-z-artifacts both ^
  --fse-z-artifact-source-split train_val ^
  --fse-z-artifact-min-samples 1000 ^
  --fse-z-stats-output-path "%STAGEG_ROOT%/stage_g1_single/artifacts/42/fse_z_global_stats.json" ^
  --fse-shuffle-pool-output-path "%STAGEG_ROOT%/stage_g1_single/artifacts/42/fse_shuffle_pool.npz" ^
  --fse-output-dir "%STAGEG_ROOT%/stage_g1_single/artifacts/42"
```

Random formal：

```bat
python %SCRIPT% ^
  --mode paper_csac_fse_z_gated_random ^
  --traffic-density %DENSITY% ^
  --seed 42 ^
  --device %DEVICE% ^
  --episodes 300 ^
  --max-steps 400 ^
  --eval-episodes 10 ^
  --fse-checkpoint-path "%FSE_CKPT%" ^
  --fse-run-tier formal ^
  --fse-random-z-distribution global_empirical_matched ^
  --fse-z-global-stats-path "%STAGEG_ROOT%/stage_g1_single/artifacts/42/fse_z_global_stats.json" ^
  --fse-z-trace-stride 5 ^
  --fse-z-trace-max-rows 100000 ^
  --train-log-path "%STAGEG_ROOT%/stage_g1_single/z_gated_random/42/train_log.csv" ^
  --train-json-path "%STAGEG_ROOT%/stage_g1_single/z_gated_random/42/train_log.json" ^
  --run-result-json-path "%STAGEG_ROOT%/stage_g1_single/z_gated_random/42/run_result.json" ^
  --checkpoint-save-path "%STAGEG_ROOT%/stage_g1_single/z_gated_random/42/checkpoint.pt" ^
  --tensorboard-dir "%STAGEG_ROOT%/stage_g1_single/z_gated_random/42/tensorboard"
```

Shuffled formal：

```bat
python %SCRIPT% ^
  --mode paper_csac_fse_z_gated_shuffled ^
  --traffic-density %DENSITY% ^
  --seed 42 ^
  --device %DEVICE% ^
  --episodes 300 ^
  --max-steps 400 ^
  --eval-episodes 10 ^
  --fse-checkpoint-path "%FSE_CKPT%" ^
  --fse-run-tier formal ^
  --fse-random-z-distribution global_empirical_matched ^
  --fse-z-global-stats-path "%STAGEG_ROOT%/stage_g1_single/artifacts/42/fse_z_global_stats.json" ^
  --fse-shuffle-pool-path "%STAGEG_ROOT%/stage_g1_single/artifacts/42/fse_shuffle_pool.npz" ^
  --fse-z-trace-stride 5 ^
  --fse-z-trace-max-rows 100000 ^
  --train-log-path "%STAGEG_ROOT%/stage_g1_single/z_gated_shuffled/42/train_log.csv" ^
  --train-json-path "%STAGEG_ROOT%/stage_g1_single/z_gated_shuffled/42/train_log.json" ^
  --run-result-json-path "%STAGEG_ROOT%/stage_g1_single/z_gated_shuffled/42/run_result.json" ^
  --checkpoint-save-path "%STAGEG_ROOT%/stage_g1_single/z_gated_shuffled/42/checkpoint.pt" ^
  --tensorboard-dir "%STAGEG_ROOT%/stage_g1_single/z_gated_shuffled/42/tensorboard"
```

## Test Plan
- Static check:
  - `python -m py_compile fix10E_fse_guided_paper_csac.py`
- Artifact build:
  - Run `--fse-task build-z-artifacts` with real `%FSE_DATASET%/%FSE_CKPT%`.
  - Confirm result JSON contains stats path, pool path, sample count, checkpoint hash, builder hash, selected-index hashes, self-check status.
- Positive validation:
  - Stats JSON has `mu_z/std_z` length 64, raw std summary, and standard metadata fields.
  - Pool NPZ has `z.shape == [N,64]`, original `source_index`, `episode_id/step_id`, parseable `metadata_json`, duplicated standalone metadata fields, and z-norm summary.
  - Formal random runtime loads stats.
  - Formal shuffled runtime loads pool and stats.
- Negative validation:
  - Missing metadata fails.
  - `source_split=test/final_mixed_test` fails.
  - checkpoint hash mismatch fails.
  - builder hash mismatch fails.
  - traffic density mismatch fails.
  - sample count below `min_samples` fails.
- Short formal runs:
  - random formal: `episodes=5 max_steps=50 eval_episodes=2` runs without missing-stats error.
  - shuffled formal: same short run with both stats and pool paths.
- Full Stage G.1b:
  - Run random formal 300 episodes.
  - Run shuffled formal 300 episodes.
  - Compare against real gated, concat, and baseline.

## Assumptions
- `%FSE_DATASET%` is the labeled `.npz` dataset used to train or calibrate `%FSE_CKPT%`.
- `%FSE_CKPT%` is state-conditioned, not action-conditioned.
- Stage G.1b formal artifact source split is `train_val`.
- Random formal only supports `global_empirical_matched`.
- Shuffled formal should pass both stats and pool paths for audit consistency.
- No heads-only, action-conditioned FSE, teacher prior, memory calibration, or parameter-matched state-only baseline will be added in this change.
