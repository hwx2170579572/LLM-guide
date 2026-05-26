# 模块三最终计划：FSE Bottleneck 接入 CSAC

## Summary
实现 frozen state-conditioned FSE 接入 CSAC。主模型为 `paper_csac_fse_z_gated`，对照为 `paper_csac_fse_z_concat`，消融为 `paper_csac_fse_z_gated_random / shuffled / zero`。RL 只使用 `state_aug = concat(raw_state, z_used)`；decoder heads 不进入 Actor/Critic。

## Key Changes
- 实验模式与配置：
  - 新增 modes：`paper_csac_fse_z_concat`、`paper_csac_fse_z_gated`、`paper_csac_fse_z_gated_random`、`paper_csac_fse_z_gated_shuffled`、`paper_csac_fse_z_gated_zero`。
  - 新增 CLI/config：`--fse-fusion-mode`、`--fse-z-mode`、`--fse-random-z-distribution`、`--fse-random-z-std-floor`、`--fse-run-tier`、`--allow-legacy-fse-normalization`、`--fse-z-global-stats-path`、`--fse-shuffle-pool-path`、`--fse-z-trace-stride`、`--fse-z-trace-max-rows`。
  - formal random 必须提供 `--fse-z-global-stats-path`；formal shuffled 必须提供 `--fse-shuffle-pool-path`。
  - smoke 可自动生成临时 stats/pool，但必须写 warning。

- z 字段语义：
  - replay/trace 存 `z_real`、`z_real_valid`、`z_used`、`z_used_valid`、`z_source`、`next_z_real`、`next_z_real_valid`、`next_z_used`、`next_z_used_valid`、`next_z_source`、`fse_forward_executed`。
  - 所有进入 replay 的 transition 必须 `z_used_valid=true` 且 `next_z_used_valid=true`。
  - `real`：`z_real_valid=true`，`z_used=z_real`。
  - `shuffled`：`z_real_valid=true`，`z_used=z_shuffled`。
  - `random` formal rollout：`z_real_valid=false`，`z_real=NaN` 或 omitted，`z_used=z_random`。
  - `zero`：`z_real_valid=false`，`z_real=NaN` 或 omitted，`z_used=0`。
  - 主诊断使用 `z_used`；真实 FSE 诊断单独使用 `z_real`。

- z 生成策略：
  - `real`：每 step 执行 FSE forward。
  - `shuffled`：每 step 执行 FSE forward 得到 `z_real`，当前 step 先从旧 buffer/offline pool 采样 `z_used`，store 后再 append 当前 `z_real`。
  - `random`：formal rollout 不执行当前 step FSE forward，只读固定 global z stats。
  - `zero`：不执行 FSE forward。
  - deterministic seed 使用 SHA256 稳定哈希，不使用 Python 内置 `hash()`。
  - global stats 来源为 FSE train/val split 或独立 unlabeled calibration pool，禁止使用 final mixed test split。
  - `std_z = max(std_z, fse_random_z_std_floor)`，默认 `1e-4`；记录 clipped 维度数。

- shuffled 规则：
  - offline shuffled pool 来源于 train/calibration split，不来自当前 eval episodes。
  - eval sample index 使用稳定 SHA256 对 `(seed, episode_id, step_id, mode)` 取模。
  - online shuffled 优先跨 episode；次选同 episode且步距大于 50；不足 fallback 到 global random。
  - formal 允许 warmup 排除；纳入正式统计后 `shuffled_random_fallback_rate <= 0.10`。eval 使用 offline pool 时 fallback 必须为 0。

- Frozen FSE 与审计：
  - 所有 FSE-RL modes 都要求 checkpoint；random/zero 也加载 checkpoint用于 metadata、z_dim、参数量、审计。
  - 校验 schema、`z_dim=64`、`token_dim=24`、`n_tokens=12`、`horizons=[10,20,40]`、risk names、`action_conditioned=False`。
  - FSE `eval()`、`requires_grad=False`、不进入 optimizer。
  - 记录 `fse_optimizer_param_count=0`、`trainable_fse_param_count=0`、`fse_nonzero_grad_param_count=0`、`fse_grad_norm=0`、`fse_weight_delta_norm=0`。
  - 保存 `fse_weight_sha256_before/after`。
  - 审计字段包括 `fse_model_class`、`fse_model_config_hash`、`fse_checkpoint_sha256`、`scenario_builder_config_hash`、`script_git_sha_or_file_hash`。
  - legacy normalization：smoke 可 warning 放行；formal 必须显式允许，否则报错。

- 网络与接口：
  - Agent 统一接收 `state_aug`；gated 网络内部 assert `state_aug.shape[-1] == raw_state_dim + z_dim` 并 split。
  - concat 网络直接消费 `state_aug`。
  - fusion encoder 只融合 raw state 与 z_used；action 不进入 gate。
  - all reward critics、target reward critics、all cost critics、target cost critics 使用相同 fusion interface；action 在 fusion 后 concat 进 Q MLP。
  - 固定 gate 日志列：
    - `fse_actor_gate_mean/std`
    - `fse_reward_critic1_gate_mean/std`
    - `fse_reward_critic2_gate_mean/std`
    - `fse_cost_critic_safety_gate_mean/std`
    - `fse_cost_critic_boundary_gate_mean/std`
    - `fse_cost_critic_speed_gate_mean/std`
  - 若后续实现不同 cost critic 集合，仍保留固定列；不存在的列填 NaN。
  - gated `z_mlp` 使用 `bias=False`；zero z 仅作 sanity check，不替代原始 CSAC baseline。
  - 预留 `paper_csac_fse_z_gated_state_extra` 作为 parameter-matched state-only control，本轮不实现。

- Replay、输入校验与 trace：
  - replay 存 `raw_state/z_real/z_real_valid/z_used/z_used_valid/state_aug/action/reward/cost/next_raw_state/next_z_real/next_z_real_valid/next_z_used/next_z_used_valid/next_state_aug/done/metadata`。
  - 训练只使用 `state_aug/next_state_aug`。
  - 每次入 replay 前检查：
    - `state_aug_has_nan`
    - `state_aug_has_inf`
    - `next_state_aug_has_nan`
    - `next_state_aug_has_inf`
  - train 发现 NaN/Inf 立即报错；eval 立即报错并写 error type。
  - 输出 `fse_z_trace.csv`，支持 stride/max rows；聚合指标每步统计。
  - 记录 `z_real_*` 与 `z_used_*` 两套 variance/temporal delta。
  - `fse_state_aug_norm_ratio = ||z_used||_2 / (||raw_state||_2 + 1e-6)`，记录 p10/p50/p90。

- 配置与 checkpoint：
  - 每个 run 输出 `resolved_config.json`，至少包含 mode、fusion/z mode、random distribution/std floor、checkpoint path/hash、global stats path/hash、shuffle pool path/hash、raw_state_dim、z_dim、state_aug_dim、seed、traffic_density、episodes、max_steps、eval_episodes。
  - agent checkpoint 写入 `fse_rl`：enabled、fusion_mode、z_mode、dims、FSE checkpoint hash、builder hash、global stats hash、shuffle pool hash、legacy normalization flag。
  - eval reload 校验 mode、dims、checkpoint hash、builder hash、global stats/shuffle pool hash 一致。

- RNG 隔离：
  - 拆分 `env_rng`、`agent_rng`、`replay_rng`、`fse_random_z_rng`、`shuffle_rng`、`eval_rng`。
  - baseline 模式不得初始化 FSE runtime、不得读取 FSE checkpoint、不得创建 FSE RNG、不得改变全局 RNG 调用顺序。
  - baseline 回归使用 fixed action trace 或独立 action RNG。

## Test Plan
- Baseline 回归：
  - 同 seed/同参数跑 `paper_csac_lagrangian` first-N-step trace。
  - 验收：action/reward/done 与修改前一致；未初始化/读取 FSE；未调用 FSE RNG；state_dim 不变。

- Real z smoke：
  - 跑 `paper_csac_fse_z_concat` 与 `paper_csac_fse_z_gated`。
  - hard checks：`z_used_batch_variance_mean > 1e-6`、`z_used_temporal_delta_mean > 1e-6`、loss 无 NaN/Inf、median gate 在 `[0.01,0.99]`、FSE frozen 证据全部满足、state_aug 无 NaN/Inf。

- 消融 smoke：
  - random：`z_real_valid=false` in formal rollout，`z_used_batch_variance_mean > 1e-6`，读取固定 global stats。
  - shuffled：`z_used_batch_variance_mean > 1e-6`，`shuffled_self_sample_count=0`，formal fallback rate 达标。
  - zero：`z_used_batch_variance_mean == 0`，`z_used_norm_mean == 0`。
  - 三者参数量与 real gated 一致。

- Eval smoke：
  - 跑 `paper_csac_fse_z_gated --episodes 1 --max-steps 30 --eval-episodes 2`。
  - 验收：eval 正常，`z_used_norm_mean > 0`，不依赖 optimizer/replay，`llm_calls=0`，state_aug 无 NaN/Inf。

- Stage G.1 single-seed readiness：
  - concat/gated 跑 300 episodes。
  - success 不显著低于 baseline；collision、unsafe_headway、low_ttc、mean_cost_safety 至少一个改善。
  - critic loss 不持续发散，lambda 不异常饱和，z/gate trace 稳定。

- Stage G.2 multi-seed readiness：
  - sparse 3 seeds，只作为是否进入下一阶段的趋势判断，不作为论文显著性结论。
  - real gated 相对 baseline 的 safety gain 至少比 random/shuffled/zero 中位 gain 高 20%，或在 collision、unsafe_headway、low_ttc、mean_cost_safety 中至少两个指标优于消融。
  - 正式论文结论需要 5 seeds 或 bootstrap CI。
  - 若 real gated 只提升 return、不改善 safety，不进入 action-conditioned FSE。

## Assumptions
- 本轮只实现模块三，不实现 action-conditioned risk penalty、teacher prior、scheduler、memory calibration。
- `heads_only` 与 `z_without_heads` 暂不实现，但保留接口扩展空间。
- formal random/shuffled 不使用 final mixed test split 生成 z stats 或 z pool。
- 当前 Stage E/F checkpoint 可用于 smoke；formal run 需要结构化 normalization，或显式允许 legacy normalization。
