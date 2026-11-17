# 02-eval

- **输入**：两组的 checkpoint-3000（经 ae 模型缓存取回），配置由 `scripts/make_eval_configs.py` 生成在 `configs/eval-{template,clue-target}/`。
  - 测试集 `asr/context-correction-zh@v2-20261117`，think 关、开各一遍；
  - ae 0.14.0-eedcf15，vLLM 0.32，自带 `mtp/` 做投机解码（EAGLE k=2）。
- **完成条件**：四个 run 都完整，没有失败样本，W&B 远端有指标。按实验 README 的“判定”给出配对 bootstrap 和 think 诊断。
- **进展**：等训练完成。
- **结果和证据**：服务记录在 `serve-<组>-<asr|tsasr>/attempts/`，评测 run 在 `/workspace/data/ae-runs/ctx-clue-target-20261117-<组>[-think]/runs/`。
