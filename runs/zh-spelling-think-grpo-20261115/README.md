# 说字 think 的 GRPO（2026-11-15）

**状态**：2026-11-15 挖掘中，ACP 任务 `pt-uf80hisv`。

**要回答的问题**：说字 think 第二轮的 checkpoint-3000，think 开已经可用，说字也明显变强；但续训 SFT 把交付版 GRPO 压下来的误触发又冲了回去，和交付版做插值后 think 开又坏掉。照交付版的做法在它上面做一轮 GRPO，能不能同时做到下面两点？
- 守门回到交付版水平；
- think 开仍然可用，说字能力保住。

第二轮的结果见 `../zh-spelling-think-r2-20261115/README.md`。

## 设计

| | 交付版那一轮（oal-grpo 第十轮） | 本实验 |
|---|---|---|
| 起点 | v9 插值 50% | 第二轮 think checkpoint-3000 |
| 挖掘 | 在起点上重挖五份计划：热词、无语音、漏识、TS 缺席、TS 在场；7 路普通识别、7 路 TS；每条 16 次采样 | 同左，在本实验的起点上重挖，think 关 |
| 训练 | `train-grpo-v10.yaml`：8 卡全参 400 步，学习率 1e-6，KL 0.04（参考模型 = 起点），每步 32 条音频 × 8 次采样，奖励为准确率、出字判断、热词、格式 | 同左，只换起点和代码 |
| 候选 | 第 200/300/400 步的平均（avg3），以及它与起点等权的插值（插值 50%） | 同左 |
| 代码 | code-v12 | `code-v2`（main `f68e180`，与第二轮相同，已含 GRPO 代码） |

- **配置**：都由 `scripts/make_configs.py` 从 oal-grpo `runs/amphionasr2-grpo-20261101` 的第十轮配置复制，只改起点、代码和本实验的路径。
  - `configs/train-data-grpo.yaml`：回放与渲染同第十轮，五份计划读 `tasks/01-mining/`。
  - `configs/train-grpo.yaml`：训练配置。
  - `configs/mining/serve-*.yaml`：挖掘服务。
- **GRPO 只用 think 关**：
  - GRPO 的提示不经过 think 配置，交付版那一轮也是 think 关。
  - think 开靠两点保住：KL 的参考模型就是起点；插值的另一端也是起点，两端都会 think。
  - 第二轮的插值版 think 开坏掉，是因为另一端是没见过 think 的交付版。
- **脚本**：
  - `scripts/mining.sh`：同交付版那一轮的挖掘线、来源、候选数和 epoch。
  - `scripts/train.sh`：训练。
  - `scripts/eval_after_training.sh`：训练后做 avg3 和插值，放进 ae 模型缓存，再评测。
  - 评测配置的生成脚本从第二轮复制，只改了输出目录。

## 评测（训练后）

avg3 和插值 50% 同时测，项目和口径都同第二轮：
- 上下文纠偏：think 关、think 开各一遍；
- 守门 8 项：think 关；
- 中文 TS-ASR 测试集、鼎桥机主识别、asrreg：再用 think 开各跑一遍。

## 判定

- **守门回到交付版水平**：think 关，每项都不比交付版差。差异用配对 bootstrap（`ae open-audio-llm compare`）判断，看 95% CI。
  - 重点看 GigaSpeech 无语音误触发：交付版 27.82，第二轮 34.87。
  - 交付版的中文 TS-ASR 测试集和鼎桥机主识别用的是旧 TS 提示，这两项的差里含提示变化，与第二轮同样处理。
- **think 开可用**：同一模型 think 开减 think 关，第二轮那六项的差都不超过 0.5 个点。
- **说字保住**：think 开纠正成功率、真实挖掘准确率不明显低于第二轮 think 开（78.51%、109/127）。纠正成功率要打折看，原因见第二轮 README。
- **交付**：三项都满足的候选进入交付评估，avg3 和插值都满足时选守门更好的；都不满足就记录结果，交付版不变。

## 启动

```bash
python scripts/make_configs.py
open-audio-llm experiment run --config runs/zh-spelling-think-grpo-20261115/experiment.yaml --task grpo --dry-run
```

在 ACP 上用 inbox 执行器运行，执行器目录为 `/workspace/workspace/mingdong/scratch/acp-think-grpo`。投递了三个脚本：
- `10-mining.sh`：执行 `scripts/mining.sh`；
- `11-train.sh`：执行 `scripts/train.sh`，等挖掘退出码为 0 且五份计划都不为空才开训；
- `12-eval-after-training.sh`：执行 `scripts/eval_after_training.sh`，等训练完成。

## 执行记录（2026-11-15）

- **节点**：用户允许最多占一个 ACP 节点。第二轮的 `pt-4hzqvhue` 停止后，提交 1 节点任务 `pt-uf80hisv`（`AmphionASR-spelling-think-grpo-v10recipe-from-r2think3000-1node`），10:42Z 就绪，节点 IP 10.119.18.2，8 张 A800 空闲。
- **W&B 转发**：放行名单加入 10.119.18.2，原有 IP 不变；从节点经转发访问 W&B 返回 200。
- **预检**：`experiment run --dry-run` 通过；`grpo_mining.py` 与 oal-grpo code-v14（第十四轮所用）逐字相同。
- **挖掘**：10:44Z 投递。
