# 08-compare-consistency：新旧链路逐条比较

- 输入：04、05 的新链路输出与旧实验脚本的原输出、原分数（evaluate-select 执行 002、score-meeting-benchmark 执行 001、evaluate-moss-meeting-benchmark 执行 001）。
- 方法：同一批旧输出用 AmphionEval 打分器重算，把总差拆成打分差与推理差；统计逐条字节一致、去时间戳一致、复读翻转。
- 结果：执行 001 因嵌套路径未解析失败（保留）；执行 002 完成，结论见实验 README。证据 `attempts/002/artifacts/consistency.json` 与 `*-samples.jsonl`。
