# 09 上下文纠偏评测集 v2：交付版与 avg3

- **要回答的问题**：avg3 在上下文纠偏上的进步，能不能推广到训练没见过的说法、题型、声音和真实录音。
  v1 的合成题与训练数据同源，真实题只有 127 条，回答不了这个问题。
- **输入**：
  - 测试集：上下文纠偏评测集 v2 的定义（`configs/ctx-v2-data.yaml`），包含 v1 原有的三个切片，以及 v2 新增的独立合成题、独立合成 TS、Emilia 真实录音。
    - 数据 `context_correction_zh@v2-20261116` 登记在 audio-data-contract #251，catalog 取登记提交 9f8af4c 的快照（`/workspace/workspace/mingdong/repo/adc-ctx-v2-9f8af4c/catalog`）。
    - #251 合入、评测集发布为 `asr/context-correction-zh@v2-<日期>` 后，改按名字引用。
    - 设计和制作过程见 `/workspace/workspace/mingdong/data/amphiondata/context-correction-zh-v2-20261116/README.md`。
  - 模型：
    - 交付版 `amphionasr2-grpo-v10interp050`，think 关。它没有自己的草稿头，按交付验证时的方式不开 MTP。
    - avg3 挂 06 的 EAGLE-3 草稿头，k=3，think 关和开各一遍（07 已核对输出与不开 MTP 一致）。
  - 三组都用 ae 0.14.0-eedcf15 和新 TS 提示，与实验 README 里 v1 的表同口径。交付版训练时用旧 TS 提示，它的 TS 切片会因此吃亏，只看普通识别切片做比较。
- **配置**：`configs/eval-ctxv2-{release,avg3-eagle3-k3}/`，由 `scripts/make_ctx_v2_configs.py` 生成，只换模型、数据、GPU、端口、服务名和记录目录。
- **入口**：节点 `pt-1blli2zm` 的 inbox 作业 `65-ctx-v2-eval.sh`，两组服务同时跑：交付版用 GPU 0/1，avg3 用 GPU 2/3。每组调用 `scripts/run_eval.sh`。
- **完成条件**：三项评测都完整、没有失败样本，W&B 远端有指标。按组和 `train_overlap` 给出纠正成功率、保持率、成对正确率、真实录音准确率，并用配对 bootstrap 给出置信区间（AmphionEval !158 的 `ae open-audio-llm compare`）。
- **进展**：
  - 2026-11-16 22:25Z 四个服务就绪，22:34Z 三项都完整结束：各 9,892 条，0 条失败。
  - W&B 远端三个 run 都已 finished，各有约 690 项指标。
  - 节点 `pt-1blli2zm` 随后停掉。
- **结果**：见实验 README 的“09 结果”。
  - 配对 bootstrap 输出在 `compare/cmp-{release-avg3off,avg3off-avg3on,release-avg3on}.json`，用 AmphionEval !158 的 `ae open-audio-llm compare` 计算。
  - 真实录音按“原转写是否需要纠正”拆开的表由 `scripts/ctx_v2_mined_groups.py` 生成。
  - run：交付版 `run-20261116T222524Z-8996a9f7`，avg3 think 关 `run-20261116T222524Z-d5ab0981`，think 开 `run-20261116T222955Z-23902588`。目录 `/workspace/data/ae-runs/spelling-think-grpo-20261115-ctxv2-{release,avg3-eagle3-k3,avg3-eagle3-k3-think}/runs/`。
