## Step 1（最终版）：ScenarioFrame 场景封装（可开关、只读）

### Summary
- 基于 `fix10E_fse_guided_paper_csac.py` 新增 `ScenarioFrameBuilder` 与 `ScenarioFrame` 双视图输出。
- 补齐你要求的两点：
  1. `ScenarioFrameBuilder` 初始化持有 `cfg`（统一归一化参数来源）。
  2. 增加 `enable/disable scenario frame` 开关（用于行为不漂移对照测试）。
- 保持 Step 1 只读，不改 reward/cost/actor/critic/replay 行为路径。

### Key Changes
- 在 [fix10E_fse_guided_paper_csac.py](D:/Program Files (x86)/paper/LLM-guide/fix10E_fse_guided_paper_csac.py) 新增：
  - `class ScenarioFrameBuilder:`
    - `__init__(self, cfg: Config)`：保存 `self.cfg`，所有归一化/裁剪阈值仅从 `cfg` 读取。
    - `build(self, scene_dict: dict, episode_id: int, step_in_episode: int) -> ScenarioFrame`
    - 构建器为纯函数式使用：不访问 env、不调用 `get_scene_dict()`、不修改入参字典。
  - `ScenarioFrame` 数据结构：
    - `semantic`
    - `tokens[12,9]`
    - `token_mask[12]`
    - `token_type_ids[12]`
    - `token_role_ids[12]`
    - `scalar_context[7]`
    - `meta`

- 新增配置与开关：
  - `TrainConfig.enable_scenario_frame: bool = True`
  - CLI 增加互斥布尔：
    - `--enable-scenario-frame`
    - `--disable-scenario-frame`
  - `run()` / `evaluate_policy()` 中按开关决定是否构建 ScenarioFrame。
  - `[RUN-CONFIG]` 增加 `scenario_frame_enabled=True/False`。

- Token 规则保持锁定：
  - lane token（含 left/right）`token_mask` 永远为 1，用 `lane_available` 表达可用性。
  - neighbor token 按实体有效性决定 mask。
  - `semantic` 使用 `entity_valid` 命名；`token_mask` 仅用于 attention。

### Test Plan
- 结构与语义测试：
  - `tokens.shape==(12,9)`、`scalar_context.shape==(7,)`、无 NaN/Inf。
  - 边界车道和无邻车场景下 mask/available 规则正确。
- 开关测试：
  - `--disable-scenario-frame` 时，不创建 ScenarioFrame、不打印 `[SCENARIO-FRAME]`。
  - `--enable-scenario-frame` 时，首步打印并通过 schema assert。
- 行为不漂移测试（关键）：
  - 固定 seed、episodes、start_steps，分别运行：
    1. `--disable-scenario-frame`
    2. `--enable-scenario-frame`（建议关闭额外逐步打印）
  - 对比前若干 episode 的 `action/reward/cost/done/replay_size`，应一致（浮点容差内）。

### Assumptions
- Step 1 不把 ScenarioFrame 写入 replay，不参与网络输入，不改变训练更新逻辑。
- 默认正式实验保持 `enable_scenario_frame=True`，如需做漂移对照可切到 `False`。
