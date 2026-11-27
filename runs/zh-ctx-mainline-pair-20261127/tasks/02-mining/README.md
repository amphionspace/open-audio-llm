# 02 挖题（K0 起点，第十轮的五份计划）

- **输入**：K0 checkpoint-3000，`configs/train-data-grpo.yaml`、`configs/mining/`（`scripts/make_configs.py grpo START`，从 kgrpo 的配置改）。
- **做法**：`scripts/mining.sh`，同 kgrpo，去掉两路 ctx 挖题，切计划时不给 context targets（不产生 ctx 计划）。
- **完成条件**：hotword、nonspeech、miss、ts-absent、ts-present 五份计划都非空，`summary.json` 写出。
- **进展**：
  - 配置（16:31Z）：`make_configs.py grpo` 输出 133 个来源、窗口 5000，挖题来源 plain 5、miss 12、ts 42，计划只有五份（没有 plan-ctx）。
  - 第一次（节点脚本 20-k0-retry，16:31Z）：`mining.sh` 开头的检查报 “data config still has the ctx plan” 并退出 5，没有开始挖题。
    - 原因：检查用 grep 找 `context_targets`，匹配到的是 make_configs.py 写的文件头注释（“去掉 ctx 计划和 context_targets”）。解析后的配置里没有 context_targets，计划也只有五份。
    - 修正：检查跳过注释行。第一次修改把注释写在了 `{ ... }` 里，造成语法错误（30-k0-mining，立即退出 2，没有运行任何东西），再改正后通过 `bash -n`。
  - 第二次（31-k0-mining，16:35Z 起）：挖题 → GRPO → 评测。
  - 记录位置：挖题服务的执行记录写在 `tasks/01-mining/serve-*/attempts`。这个路径是从 kgrpo 的服务配置照搬的（kgrpo 的挖题是任务 01），相对于本实验，不碰 kgrpo 的记录。服务配置文件头里的模型名也还写着 K 的臂名，实际加载的路径是 K0 的 checkpoint-3000（`checkpoint-3000-29899967e537`）。挖题产物（计划、summary）在 `tasks/02-mining/`。
