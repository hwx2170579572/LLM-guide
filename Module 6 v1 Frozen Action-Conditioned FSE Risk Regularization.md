# Module 6 v1: Frozen Action-Conditioned FSE Risk Regularization

## Summary

在 frozen SG-FSE-CSAC 上加入 one-step action-conditioned FSE risk regularization。`z_state = FSE_state(tokens)` 继续输入 Actor/Critic；`FSE_action(tokens, action)` 只在 actor update 中通过 decoder risk head 提供可微风险正则。

本轮不接 teacher prior、不改 reward/cost/env.step/action clip、不做 OOD 规则门控。`g_ood` 不进入 loss，`ood_action_rate` 仅记录为 `NaN`。

## Key Changes

- 新增模式：
  - `paper_csac_fse_z_gated_action_risk`
  - `paper_csac_fse_z_gated_action_risk_shuffled_tokens`
  - 均固定 `fusion_mode=gated`、`z_mode=real`；普通 `paper_csac_fse_z_gated` 不加载 action-FSE，行为保持不变。

- 新增 `FrozenActionFSEPenaltyRuntime`：
  - 只接受 `action_conditioned=True`、`action_dim=2` checkpoint。
  - 现有 state-FSE runtime 只接受 `action_conditioned=False` checkpoint。
  - action-FSE 参数 `requires_grad=False`、不进 optimizer，但 forward 不能使用 `torch.no_grad()`。

- acceptance gate：
  - CLI：`--fse-action-risk-checkpoint-path`、`--fse-action-risk-acceptance-json`、`--strict-fse-action-gate`。
  - hard gate：checkpoint hash、schema、`dataset_type=same_state_branch_rollout`、`action_space=policy_normalized_tanh`、`action_dim=2`、same-state pair AUC `>=0.65`、`z_variance > 1e-6`、`risk_grad_smoke_passed=true`。
  - dev profile 下 rare recall/ECE/Brier/monotonic violation 只 report/warn；formal profile 下这些指标按阈值 hard gate。

- action 坐标系强制拆分：
  - 扩展 Actor/FSEActor 的采样接口，保留旧 `sample()`，新增返回 normalized 与 env-scaled action 的方法。
  - 当前 critic/replay 路径使用 env-scaled action，因此 actor update 中固定命名：
    - `new_action_norm`
    - `new_action_env`
    - `action_for_q = new_action_env`
    - `action_for_env = new_action_env`
    - `action_for_fse = new_action_norm`
  - `action_for_fse` 必须满足 `[-1.0001, 1.0001]` 断言，并记录 min/max/abs mean。
  - action-FSE dataset/checkpoint 优先使用 `actions_policy_norm`，metadata 写入 `policy_normalized_tanh`、bounds、order。

- ReplayBuffer：
  - action-risk 模式保存 current frame 的 `tokens / token_mask / entity_valid_mask / token_type_ids / token_role_ids`。
  - 不改变 state-FSE current/next `z_state` 与 critic target 数据流。

## Actor Loss And Diagnostics

- actor loss：

```text
L_actor = L_CSAC + beta_risk * stopgrad(uncertainty_gate) * risk_norm
```

- risk/gate：
  - risk weights：`[1.0, 0.7, 0.6, 0.9, 0.4]`
  - horizon weights：`[0.50, 0.35, 0.15]`
  - `risk_norm = risk_raw / 3.6`
  - `uncertainty_gate = exp(-kappa * clamp(weighted_uncertainty, 0, 5)).detach()`
  - defaults：`beta_risk=0.05`、`kappa=2.0`

- action-FSE forward 必须保持 action 梯度：

```python
tokens = tokens.detach()
out = fse_action(tokens, token_mask, entity_valid_mask, token_type_ids, token_role_ids, action=action_for_fse)
```

不得写成 `with torch.no_grad(): out = fse_action(...)`。

- 梯度诊断每 `--fse-action-risk-grad-diag-interval` 步计算，默认 50；必须 `retain_graph=True`，随后仍执行完整 `actor_loss.backward()`：

```python
grad_risk = torch.autograd.grad(
    risk_norm.mean(),
    action_for_fse,
    retain_graph=True,
    create_graph=False,
    allow_unused=False,
)[0]
```

- 日志字段：
  - `fse_action_risk_raw_mean`
  - `fse_action_risk_norm_mean`
  - `fse_action_uncertainty_h10_mean / h20_mean / h40_mean`
  - `uncertainty_gate_mean / min / max`
  - `risk_penalty_mean`
  - `risk_high_rate`
  - `uncertainty_gate_suppressed_rate`
  - `risk_penalty_active_rate`
  - `action_for_fse_min / max / abs_mean`
  - `risk_grad_norm`
  - `cos_grad_actor_base_risk`
  - `grad_base_norm`
  - `grad_risk_norm`
  - `risk_to_base_grad_norm_ratio`
  - `ood_action_rate = NaN`

- freeze audit：
  - `fse_action_param_count`
  - `fse_action_trainable_param_count`
  - `fse_action_nonzero_grad_param_count`
  - `fse_action_hash_unchanged`

## Shuffled-Token Ablation

`paper_csac_fse_z_gated_action_risk_shuffled_tokens` 只打乱 action-risk 分支输入：

```python
perm = torch.randperm(batch_size, device=tokens.device)
tokens_for_action_fse = tokens[perm]
token_mask_for_action_fse = token_mask[perm]
entity_valid_mask_for_action_fse = entity_valid_mask[perm]
token_type_ids_for_action_fse = token_type_ids[perm]
token_role_ids_for_action_fse = token_role_ids[perm]
```

不得打乱 `state`、`z_state`、state-FSE tokens、`new_action_norm`、reward、cost、done、next_state、critic target、ReplayBuffer 顺序。

## Test Plan And Formal Interpretation

- Tests:
  - state runtime 拒绝 action checkpoint；action runtime 拒绝 state checkpoint。
  - action metadata、action space、acceptance gate 不匹配时 fail。
  - action-FSE forward 无 `torch.no_grad()`，`risk_grad_norm > 1e-6`。
  - action-FSE trainable params 为 0，非零 grad params 为 0，训练前后 hash 一致。
  - `action_for_fse` 始终在 `[-1.0001, 1.0001]`。
  - 普通 `paper_csac_fse_z_gated` 不加载 action checkpoint。

- Formal matrix:
  - baseline：`paper_csac_fse_z_gated`
  - main：action-risk beta `0.02 / 0.05 / 0.10`
  - ablation：shuffled-token beta `0.05`
  - sparse：`5 seeds × eval100`；dense 同样优先，资源不足先做 dense single-seed smoke。

- Interpretation:
  - improvement：collision/offroad/low_ttc/unsafe_headway 至少两个下降，且 success 不低于 baseline `-0.05`、mean speed 不低于 `-1.0 m/s`、return 不低于 `95%`、goal progress 不低于 `-5%`。
  - conservative shift：collision 下降，但 success/return/speed/progress 任意两个明显下降；只能表述为策略更保守。
