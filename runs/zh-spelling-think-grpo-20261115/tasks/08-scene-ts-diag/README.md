# 08 鼎桥场景 TS 退步定位

- **起因**：05 里 avg3 在 `tdtech/scene-ts@v1-20261028` 上 CER 3.35 → 9.24，目标不在场时误出字 16 → 48 条（共 70 条）。
  - 退步只在“分开讲话”的切片上（例如 owner-light-sep-car 0.07 → 34.86），重叠切片持平或略好。
  - 看输出：884 条目标在场的“分开讲话”样本里，83 条把另一个人的整句也转写进来；目标不在场时直接转写另一个人。
  - 交付版那次 run 用旧 TS 提示（“The clip starts with a 3-second enrollment …”），avg3 用新提示（“Transcribe only the enrolled speaker …”），05 用的服务还开着 MTP。
- **要回答**：退步来自 MTP、提示变化，还是模型本身；如果是模型，出在 GRPO 还是之前的说字 SFT。
- **配置**：`configs/eval-diag-scene-ts-<名>/`，由 `scripts/make_scene_ts_diag_configs.py` 生成。都不开 MTP，每个模型一个 TS 服务（节点 GPU 4–7）。

| 名 | 模型 | 提示（ae） | 回答 |
|---|---|---|---|
| avg3-new | avg3 | 新（0.14.0） | 去掉 MTP 后是否还在 |
| avg3-old | avg3 | 旧（0.13.3-oldprompt） | 换回旧提示能否消除 |
| release-new | 11.06 交付版 | 新（0.14.0） | 新提示本身是否让交付版也出现 |
| r2-new | 第二轮 think checkpoint-3000（GRPO 起点） | 新（0.14.0） | GRPO 之前是否已经有 |

- **入口**：inbox 脚本 `61-scene-ts-diag.sh`，06 训练结束后与 07 同时跑，`scripts/run_eval.sh <配置> <W&B 代理> <ae> gate-scene-ts`。
- **完成条件**：四项 run 完整；按切片与交付版 run `run-20261106T021900Z-53c44b85` 配对比较。
- **进展**：
  - 2026-11-16 原节点被暂停前没开始；在新节点 `pt-1blli2zm` 上 07:44Z 跑完，四项完整。
  - 追加第一轮 think 组 / 对照组两项（`r1-think-new`、`r1-control-new`），排在 Emilia 扫描之后。
- **结果**：退步在这轮 GRPO 之前就有，出在两轮说字 SFT；MTP 不是原因，提示变化只是次要因素。数字见实验 README 的“08-scene-ts-diag”。
