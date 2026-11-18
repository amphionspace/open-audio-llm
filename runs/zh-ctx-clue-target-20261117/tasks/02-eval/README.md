# 02-eval

- **输入**：各组的 checkpoint-3000（经 ae 模型缓存取回），配置由 `scripts/make_eval_configs.py` 生成在 `configs/eval-<组>/`。先测 template、clue-target，隐式题两组训练完再测。
  - 测试集 `asr/context-correction-zh@v2-20261117`，think 关、开各一遍；
  - ae 0.14.0-eedcf15，vLLM 0.32，自带 `mtp/` 做投机解码（EAGLE k=2）。
- **完成条件**：每组两个 run（think 关、开）都完整，没有失败样本，W&B 远端有指标。按实验 README 的“判定”给出配对 bootstrap 和 think 诊断。
- **进展**：
  - 2026-11-18 template、clue-target 的四个 run 完整（9,892/9,892，无失败），结论见实验 README“结果：第 1 问”。
  - 隐式题两组等数据正式登记后再测。
- **结果和证据**：
  - 服务记录在 `serve-<组>-<asr|tsasr>/attempts/`。
  - 评测 run 上传在 `whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/ae-runs/<组>[-think]/runs/`，每个 run 带 W&B 链接（`run.json` 的 `wandb_url`）。
  - `compare/` 里是这些文件：
    - 配对 bootstrap：`cmp-template-cluetarget-{off,on}.json`；
    - think 诊断：`think-clue-target-{template,clue-target}.md`；
    - Emilia 按是否需纠正拆开：`mined-groups-ab.md`。
  - `compare` 用 AmphionEval !158 的代码（`amphion-eval-ctx-compare` beb1f99，已合入 main）。已装的 ae 0.14.0、0.15.0 都还没有上下文纠偏指标。
