# TS-ASR GRPO

在已交付的 AmphionASR 包上做 on-policy GRPO，目标是目标说话人 ASR 的难例：相近声音在场时的拒识、
相近声音同时说话时只转写注册人。代码在 `src/open_audio_llm/rl/tsasr_grpo/`。

## 为什么不用 ms-swift GRPO

`examples/configs/train/grpo.yaml` 是另一个模型的单步冒烟配置。TS-ASR 的输入要把注册音和混合音分别提 Mel、
分别卷积，再插入 SEP；ms-swift 的 vLLM rollout 要支持这一点，需要改 rollout 服务和权重同步。
这里改为用正在训练的同一个 HF 模型生成 rollout（bf16 副本，每步同步），输入构造与 SFT 模板一致，
生成和求梯度走同一份模型代码。核对过：48 条样本中 47 条与 vLLM greedy 逐字一致，差的一条是 BF16 下
摇摆的出字判断。

## 算法

- 每步每卡取 `--prompts` 条样本，每条生成 `--num_generations` 条（temperature 1，不截断词表）。
- 组内优势 `(r - mean) / max(std, std_floor)`；奖励全组相同的样本不产生梯度。
- 每批 rollout 只更新一次，PPO 比值恒为 1，损失即组基线 REINFORCE：
  `-sum_i A_i * log pi(y_i) / 32`，对全部卡的全部 completion 取平均。不设参考模型和 KL，
  靠学习率和步数控制偏移。
- fp32 主权重 + bf16 autocast 更新；AdamW 在 1e-6 量级的更新低于 bf16 精度，所以不能用纯 bf16。
- 冻结参数与 SFT 相同：`embed_tokens` 和音频卷积。
- 共享 GPU 上任一卡 OOM 时，所有卡一起跳过该步（`grpo.jsonl` 记 `skipped_oom`），训练不中断。

## 奖励（`reward.py`）

文本先按 AmphionEval TS-ASR 口径归一化（NFKC、小写、去标点符号；中文按字、英文按词）。

| 情形 | 奖励 |
|---|---|
| 目标缺席 | 输出为空 1，否则 0 |
| 目标在场 | `1 - min(错误率, 2)`，范围 1 到 -1 |
| 目标在场但输出为空 | `--miss_reward`（默认 -0.5） |

`max(0, 1 - CER)` 会把"转成别人的话"和"漏识"都记 0；这里错误率可以超过 1，转错人的得分低于漏识。
漏识单独计分是第一轮实验的结论：记 0 时模型学会"拿不准就不出字"（见下文）。

## 数据

`examples/configs/data/tsasr_grpo_samples.yaml` 从 v7set4c on-the-fly 配方
（`runs/train-speed-20261019/configs/train-data-v7set4c-b24.yaml`）里选出：

- `ts_scene_pool@hard-20261016`：相近声音（ERes2NetV2 最近邻前 10 名）缺席/同说场景；
- `ts_scene_pool@multi-20261017`：多句场景，约 26% 缺席孪生，其中一半注册人是相近声音；
- aishellmix / librimix `train_1spk` 的在线混合：目标缺席、无人声窗口（`nonspeech_prob` 改为 1）、目标在场对照。

每条样本由 `CatalogSwiftDataset` 训练模式渲染，与 SFT 看到的一致，写成注册/混合两段 WAV。
留出集与训练池按记录 ID 互斥；训练池用 `--epoch 1` 渲染，干扰人、电平、房间响应与留出集不同。

```bash
python -m open_audio_llm.rl.tsasr_grpo.prepare --config examples/configs/data/tsasr_grpo_samples.yaml \
    --split heldout --out runs/tsasr-grpo/shared/samples/heldout
python -m open_audio_llm.rl.tsasr_grpo.prepare --config examples/configs/data/tsasr_grpo_samples.yaml \
    --split train --epoch 1 --exclude runs/tsasr-grpo/shared/samples/heldout/samples.jsonl \
    --out runs/tsasr-grpo/shared/samples/train
```

需要 Catalog 登记（`/workspace/data/registry`）和对应音频；路径不同时在样本配置里写 `catalog`、`roots`。
输出目录里的 `samples.jsonl` 使用相对路径，整个目录可以直接拷到别的机器。规模：留出 3000 条约 3.6 GB，
训练 6550 条约 7.9 GB（float32 WAV）。

## 在另一台机器上开始

1. 环境：Python 3.11，`torch==2.10.0`、`transformers==4.57.6`、`vllm==0.18.0`、`qwen-asr==0.0.6`、
   `rapidfuzz`、`soundfile`、`wandb`；生成样本还需要 `audio-data-contract` 和 `lhotse`。
2. 代码：`git checkout feat/tsasr-grpo`。
3. 权重：

   ```bash
   ab pull 'amphion-delivery-1373259071:AmphionASR-1.7B/releases/20261021/amphionasr-1.7b-v3-tsrobust-033avg3/' \
       models/amphionasr-1.7b-v3-tsrobust-033avg3/
   ```

4. 样本：按上一节生成，或把已生成的 `runs/tsasr-grpo/shared/samples/` 整个目录拷过来。
5. 训练（W&B 凭据从 `~/.bashrc` 读取）：

   ```bash
   open-audio-llm train --config examples/configs/train/tsasr-grpo.yaml --dry-run
   open-audio-llm train --config examples/configs/train/tsasr-grpo.yaml
   ```

   checkpoint 写在 `runs/tsasr-grpo/tasks/run/attempts/<编号>/artifacts/training/checkpoint-<步>/`，
   是包内 `model/` 目录的完整替换（权重 + 配置 + tokenizer）。

6. 留出集评测：用 TS-ASR 服务（`examples/configs/serve/tsasr.yaml`，`model` 改为 checkpoint 目录；
   它开启 SEP、全局注意力、关闭 prefix cache，并接受每条请求的 `enroll_seconds`），然后

   ```bash
   python -m open_audio_llm.rl.tsasr_grpo.evaluate run --samples runs/tsasr-grpo/shared/samples/heldout/samples.jsonl \
       --url http://127.0.0.1:8010 --model tsasr --output runs/tsasr-grpo/evals/<名字>.jsonl
   python -m open_audio_llm.rl.tsasr_grpo.evaluate compare base base.jsonl grpo ckpt240.jsonl
   ```

   基线（原始权重）先用同一服务跑一遍。v9 与普通 ASR 回归用 AmphionEval，按交付口径（交付包
   `serve.sh tsasr` / `serve.sh asr`，`MODEL` 指向 checkpoint 目录）。

## 已有结果（开发机，`amphionasr-1.7b-v4-enroll10-ckpt48000`）

第一轮：`--miss_reward 0`，学习率 2e-6，240 步，4 卡 × 4 条 × 8 个 rollout，system 为该包训练用的
"1-to-10-second" 句。

| 评测 | 指标 | 基线 | 240 步 |
|---|---|---:|---:|
| 留出集（3000 条，greedy） | 负样本误报 | 18.0% | 10.6% |
| | 相近声音在场、目标缺席误报（中文） | 47.6% | 30.5% |
| | 相近声音同说 CER（中文） | 120.7 | 92.3 |
| | 正样本漏识 | 8.9% | 14.8% |
| ts-asr-benchmark v9（22239 条，交付口径） | 误报 | 9.44% | 3.93% |
| | 漏识 | 8.97% | 14.77% |
| | CER | 24.26 | 25.54 |
| 普通 ASR（7 个测试集各 1500 条） | AISHELL-1 / LibriSpeech clean / CV-en | 0.72 / 2.06 / 10.14 | 0.73 / 2.07 / 10.12 |

结论：GRPO 能明显推动 TS 的出字判断，且不伤普通识别；但漏识记 0 分时代价是漏识上升。
第二轮把漏识改为 -0.5（本配置默认值），结果见实验记录。
