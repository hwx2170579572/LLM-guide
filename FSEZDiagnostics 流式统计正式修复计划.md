# FSEZDiagnostics 流式统计正式修复计划

## Summary

修复 `FSEZDiagnostics` 在长训练中因保存全量 `z_used_values / z_real_values` 并在 `fields()` 中反复 `np.stack()` 导致的 `ArrayMemoryError`。本修复只改变诊断统计实现，不改变 FSE forward、`z_used/z_real` 生成、`state_aug`、ReplayBuffer、reward/cost、actor/critic update 或 formal random/shuffled artifact 机制。

## Key Changes

- 在 `fix10E_fse_guided_paper_csac.py` 中为诊断统计新增内部工具：
  - `RunningMean`：维护 `count/total`，忽略非有限标量。
  - `RunningVectorVariance`：Welford online variance，维护 `count/mean/m2`，shape 错误直接 `ValueError`，非有限向量跳过并记录内部 `skipped_nonfinite`。
- 改造 `FSEZDiagnostics`：
  - 删除 `z_used_values`、`z_real_values`、norm/abs/temporal-delta 全量列表。
  - `dim_variance_mean` 用每步 `np.var(z)` + `RunningMean`。
  - `batch_variance_mean` 用 `mean(m2 / count)`，严格保持 `np.var(..., ddof=0)` 语义。
  - norm、abs、temporal delta 全部改为 running mean。
  - `z_norm_ratios` 暂保留为 float list，用于 p10/p50/p90。
  - `trace_rows` 继续由 `trace_stride` 和 `trace_max_rows` 控制。
- 保持现有输出字段名不变：
  - `fse_z_used_dim_variance_mean`
  - `fse_z_used_batch_variance_mean`
  - `fse_z_real_dim_variance_mean`
  - `fse_z_real_batch_variance_mean`
  - `fse_z_used_temporal_delta_mean`
  - 以及其它现有 FSE 诊断字段。
- 保持 mode 统计语义：
  - `real`：统计真实 `z_used` 和真实 `z_real`。
  - `random`：统计 random `z_used`，`fse_z_real_* = NaN`。
  - `shuffled`：统计错配 `z_used`，同时统计当前真实 `z_real`。
  - `zero`：`z_used` norm/dim variance/batch variance 为 0，`fse_z_real_* = NaN`。
- `reset_episode()` 只重置 `_prev_z_used/_prev_z_real`，不重置 running statistics，保持 run-level 累计统计。
- 删除或停用 `FSEZDiagnostics` 内旧 `_batch_variance()` / `_dim_variance()` 全量 stack 路径；全文件其他合法 `np.stack` 不受影响。

## Test Plan

- 语法检查：

```bat
python -m py_compile fix10E_fse_guided_paper_csac.py
```

- deterministic 统计一致性检查：
  - 100 个固定随机 64 维 z：旧 `np.stack` 结果与新 streaming 结果误差 `< 1e-6` 或 `< 1e-5`。
  - 边界用例：
    - no sample → `NaN`
    - one sample → batch variance = 0
    - all-zero samples → dim variance = 0，batch variance = 0

- 危险路径搜索：

```bat
findstr /n "z_used_values z_real_values np.stack" fix10E_fse_guided_paper_csac.py
```

验收只针对 `FSEZDiagnostics`：其中不应再有 `z_used_values.append`、`z_real_values.append`、`fields()` 中的 `np.stack` 或全量 stack variance helper。

- 四种 mode 短测，random/shuffled formal 必须带 stats/pool：

```bat
--modes paper_csac_fse_z_gated,paper_csac_fse_z_gated_zero,paper_csac_fse_z_gated_random,paper_csac_fse_z_gated_shuffled
--fse-run-tier formal
--fse-z-global-stats-path "%Z_STATS%"
--fse-shuffle-pool-path "%Z_POOL%"
```

- 压力 sanity check：
  - 先跑 `paper_csac_fse_z_gated_zero` 30 episodes，验证不会再积累 64 维 z list。
  - 再短测 `paper_csac_fse_z_gated` 和 `paper_csac_fse_z_gated_shuffled`，覆盖 `z_real_valid=True` 和 shuffled used/real 双路径。

## Formal Run Defaults

- 后续 formal multi-mode 建议用逗号分隔 modes：

```bat
--modes paper_csac_fse_z_gated,paper_csac_fse_z_gated_zero,paper_csac_fse_z_gated_random,paper_csac_fse_z_gated_shuffled
```

- G.2/G.3 长跑默认建议：

```bat
--fse-z-trace-stride 10
--fse-z-trace-max-rows 80000
```

需要更密集 trace 时再降回 stride=5。避免 stride=1 + 100000 在 multi-mode formal 中造成 Python dict trace 内存压力。

## Assumptions

- 诊断字段只用于日志、论文分析和健康检查，不参与训练决策。
- 修复后的统计是 run-level cumulative，不是 episode-level。
- batch variance 使用 population variance，即 `ddof=0`。
- random/zero 下 real-z 缺失必须保持 `NaN`，不能写成 0。
- 已完成且未崩溃的 G.1 阶段结果不强制重跑；后续正式 multi-seed/multi-mode 结果统一使用修复后的代码。若 G.1 单 seed 进入论文表格或附录，建议用修复后代码补跑以统一日志口径。
