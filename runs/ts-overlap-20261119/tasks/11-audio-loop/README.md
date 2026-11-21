# 11 循环编码器：第 17–23 层再走一遍

## 要回答的问题

用户 2026-11-21 的想法：重叠表现差是编码器不够强，借鉴循环 Transformer，让隐状态再走几遍得到更强的表示。

- 支持的证据：任务 05，冻结的 LLM 拿到"注册 + 干净原句"的编码器表示，训练分布上文字 −log p 从 0.193 降到 0.065，瓶颈在编码器输出；复盘白盒，重叠 ≥ 0.9 的错字里 29% 的正确字排在 20 名以后。
- 任务 09：音频塔对目标的选择性从第 17 层到第 23 层一路上涨、没有到顶，最后一层往回降。
- 要回答：给编码器在这一段多一遍计算，重叠下的转写能不能变好，同时守门不退。

## 设计

- **结构**（[tsasr/audio_loop.py](../../shared/code-v3/src/open_audio_llm/tsasr/audio_loop.py)，PR 分支 `feat/ts-audio-loop`）：
  - 音频配置里的 `amphion_audio_loop = {start: 16, end: 23}` 让音频塔按 `layers[:23] + loop_layers + layers[23:]` 运行。也就是第 17–23 层算完之后，用这 7 层的一份副本再走一遍，然后进第 24 层。
  - 副本存在 `audio_tower.loop_layers.{i}`。训练（逐条与批处理两条前向）、HF 打分和 vLLM 服务读同一个配置，按同样的顺序搭层。
- **起点与 11.17 逐位相同**：副本由第 17–23 层复制，注意力输出投影和 fc2 的权重、偏置置零（LLaMA-Pro 扩层的做法，仓库里 joint MTP 的初始化也是这样）。每个副本一开始在残差流上是恒等映射。
  - 起点模型：`/workspace/data/ts-overlap-20261119/models/1117-loop16-23-init`，由 [make_loop_init.py](../../scripts/make_loop_init.py) 从 11.17（`model-afea989da22b77adc0ce9365`）生成。原权重用硬链接，新增 `model-loop.safetensors`（112 个张量，8,817 万参数）。
- **为什么第二遍用可训练副本，而不是共享权重**：共享权重就得训练原来的层，原来那一遍也跟着变。任务 04/08 已经说明，从 11.17 接着训原有参数一定退化。用副本就能冻结原有音频塔和 LLM，只训副本，变化只能来自多出来的这一遍。
- **训练**（[train-loop.yaml](../../configs/train-loop.yaml)）：
  - 数据配置与 08 相同（v96 配方，训练文本终版 final-v1）。
  - 冻结 LLM 和原有音频塔，只放开 `thinker.audio_tower.loop_layers`；副本学习率 2e-5。
  - 6000 步，8 卡、梯度累积 3（全局批量 192，与 08 相同）；代码 `shared/code-v3` = code-v2 加循环补丁。
- **代价**：编码器的 Transformer 计算多 7/24（约 29%）；参数多 8,817 万（编码器原有约 3 亿）。
- **评测服务**：`shared/pack-loop` = 11.17 守门所用的 pack-mtp 运行时，加同样的循环补丁（`vendor/qwen3_asr_vllm.py` 两处）。旧运行时会悄悄跳过循环权重，所以循环模型必须用 pack-loop。

## 完成条件（2026-11-21 训练前登记）

- **上线前核对**：
  - HF：循环起点与 11.17 的音频塔输出逐位相同；真循环调试模型（副本不置零）与等价的 31 层普通模型逐位相同。
  - vLLM：pack-loop 上的循环起点与 pack-mtp 上的 11.17 输出逐条相同；pack-loop 上的真循环与 pack-mtp 上的 31 层模型输出逐条相同。
  - 冒烟 20 步：可训练参数只有循环副本，副本确实在更新，checkpoint 带循环权重和配置，平均后的模型能被 HF 和 pack-loop 加载。
- **达标**（与 11.17 比）：
  - 中文 TS v12 守门 12.88 / 371 / 369；
  - 英文 TS 误报不高于 11.17（655）；中文普通识别不差于 11.17（4.39）；
  - 主指标：v12 重叠 ≥ 0.9 的转写错误（逐条配对）低于 11.17。
- **机制**：用 `ref_score.py` 量循环后第 23 层和最后一层的选择性，看它是否比 11.17 高。

## 进展

- 2026-11-21 17:00 UTC：HF 核对通过（`attempts/001/artifacts/hf-loop-check.json`）：
  - 循环起点与 11.17 逐位相同（最大差 0）；
  - 真循环与 31 层模型逐位相同；
  - 真循环与 11.17 不同（平均差 0.005），说明循环确实在算。
- 2026-11-21 17:07 UTC：vLLM 核对 attempt 001 未完成（exit 1）。
  - 脚本误用了 `prepare_audio` 默认的 3 秒注册，评测用的是 5 秒。第 63 条（`abs-short:0080`，总长 3.675 秒，短于 5 秒切分点）让原版 pack-mtp 上的 11.17 服务崩溃。这是打包服务对短音频的原有问题，与循环无关。
  - 前 62 条四个服务收到的输入相同：循环起点 = 11.17 62/62，真循环 = 31 层模型 62/62，真循环 ≠ 11.17 20/62。
  - 已改为读评测设置里的 `enroll_seconds`。attempt 002 读错了键（`settings.json` 里是 `effective.enroll_seconds`），一条请求都没发出就退出（exit 1）。
- 2026-11-21 17:20 UTC：vLLM 核对 attempt 004 通过（exit 0）。v12 每 100 条取 1 条，共 268 条 TS 请求，5 秒注册，逐条串行、贪心解码（`attempts/004/artifacts/serve-equivalence.jsonl`）：
  - pack-loop 上的循环起点 = pack-mtp 上的 11.17：268/268；
  - pack-loop 上的真循环 = pack-mtp 上的 31 层模型：268/268；
  - 真循环 ≠ 11.17：45/268。
- 2026-11-21 17:10 UTC：本地冒烟 attempt 003 启动（开发机 4–7 号卡，`configs/smoke-train-loop.yaml`）。数据预检在开发机上很慢（ACP 上约 5 分钟）。
- 2026-11-21 17:16 UTC：ACP `pt-0vbbz0mo` 起来（节点 IP 10.119.17.219，runner `/workspace/workspace/mingdong/scratch/acp-ts11`，W&B 经开发机 17922 端口转发）。
  - 17:22 UTC 放入正式训练（[run_loop_train_eval.sh](../../scripts/run_loop_train_eval.sh)：训练 → avg3 → 四套评测，评测服务用 pack-loop）。没有等冒烟：冒烟要核对的可训练参数在启动时就能看到，第一个 checkpoint 在第 500 步，比冒烟第 20 步的 checkpoint 晚。
- 2026-11-21 17:33 UTC：正式训练 attempt 005 启动后失败（exit 1）：`batching.accumulation must equal gradient_accumulation_steps`。08 的数据配置写的是 4 卡 × 累积 6，本任务改成 8 卡 × 累积 3，数据配置没跟着改。
  - 本地冒烟 attempt 003 同因失败。
  - 修复：新增 [train-data-loop.yaml](../../configs/train-data-loop.yaml)，与 08 的数据配置只差 `batching.accumulation: 3`，训练和冒烟配置都改用它。
  - 冒烟不再重跑：它要核对的内容，下面这次正式训练都核对到了。
- 2026-11-21 17:35 UTC：正式训练 attempt 006 启动（`pt-0vbbz0mo`）。
  - 可训练参数 88.1736M，恰好是循环副本的参数量（88,173,568）；`vit_lr` 2e-5 生效。
  - 第 500 步的 checkpoint 带全部 112 个循环张量，`config.json` 带 `amphion_audio_loop {start 16, end 23}`。
  - 副本的输出投影从 0 长到平均绝对值约 3e-4；原有层的 fc1 约 1.4e-2。
  - 训练分布上的交叉熵，同样步数下都略低于 08（第 500–990 步，TS 有目标 0.1854 / 0.1893，TS 无目标 0.0541 / 0.0552，普通识别 0.1034 / 0.1052）。两组数据顺序不同（8 卡 / 4 卡）；任务 04 也说明训练分布拟合得好不代表 v12 好，所以只作参考。
