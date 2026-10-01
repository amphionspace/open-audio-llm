# 实验与分析记录：diversity-supplement-targeted-20260928

<!-- experiment-navigation -->

这是历史实验，产物保留原位。输入、完成条件、进展和结果见[任务导航](tasks/README.md)及原始方案；本次仅补齐导航，未重跑或重判质量。

执行状态：待核实；质量状态：沿用已有证据，缺少记录时保持未核实。

## 原始概览

# 多样性补充批次

合并源池：11,486 条、655 个身份、14.54 小时。
已生成并通过自动审计 66 条、1.38 小时，使用 134 个源说话人。

覆盖中文/英文、30/120 秒和 7 类场景。300/600 秒暂未发布，因为当前每个源说话人最多只有约 206 秒可用语音。

本批将英文短回应下限设为 0：英文源池没有精确匹配 whitelist 的短回应，继续强制会把完整句子误判为脏数据。中文短回应仍需单独扩充。

这是训练前的多样性验证批次，保留在这里供后续生成，不直接作为最终训练集。

| 音频 | 语言 | 时长 | 场景 | 说话人 |
|---|---|---:|---|---:|
| [generated/audio/0000/events-train-0000001.flac](generated/audio/0000/events-train-0000001.flac) | zh | 30s | gradual_entry | 2 |
| [generated/audio/0000/events-train-0000002.flac](generated/audio/0000/events-train-0000002.flac) | zh | 120s | gradual_entry | 3 |
| [generated/audio/0000/events-train-0000003.flac](generated/audio/0000/events-train-0000003.flac) | en | 30s | gradual_entry | 2 |
| [generated/audio/0000/events-train-0000004.flac](generated/audio/0000/events-train-0000004.flac) | en | 120s | gradual_entry | 3 |
| [generated/audio/0000/events-train-0000005.flac](generated/audio/0000/events-train-0000005.flac) | zh | 30s | turnover | 3 |
| [generated/audio/0000/events-train-0000006.flac](generated/audio/0000/events-train-0000006.flac) | zh | 120s | turnover | 3 |
| [generated/audio/0000/events-train-0000008.flac](generated/audio/0000/events-train-0000008.flac) | en | 120s | turnover | 3 |
| [generated/audio/0000/events-train-0000009.flac](generated/audio/0000/events-train-0000009.flac) | zh | 30s | long_return | 2 |
| [generated/audio/0000/events-train-0000010.flac](generated/audio/0000/events-train-0000010.flac) | zh | 120s | long_return | 3 |
| [generated/audio/0000/events-train-0000011.flac](generated/audio/0000/events-train-0000011.flac) | en | 30s | long_return | 2 |
| [generated/audio/0000/events-train-0000013.flac](generated/audio/0000/events-train-0000013.flac) | zh | 30s | repeated_return | 2 |
| [generated/audio/0000/events-train-0000014.flac](generated/audio/0000/events-train-0000014.flac) | zh | 120s | repeated_return | 3 |
| [generated/audio/0000/events-train-0000015.flac](generated/audio/0000/events-train-0000015.flac) | en | 30s | repeated_return | 2 |
| [generated/audio/0000/events-train-0000016.flac](generated/audio/0000/events-train-0000016.flac) | en | 120s | repeated_return | 3 |
| [generated/audio/0000/events-train-0000017.flac](generated/audio/0000/events-train-0000017.flac) | zh | 30s | overlap_entry | 2 |
| [generated/audio/0000/events-train-0000018.flac](generated/audio/0000/events-train-0000018.flac) | zh | 120s | overlap_entry | 3 |
| [generated/audio/0000/events-train-0000019.flac](generated/audio/0000/events-train-0000019.flac) | en | 30s | overlap_entry | 2 |
| [generated/audio/0000/events-train-0000022.flac](generated/audio/0000/events-train-0000022.flac) | zh | 120s | brief_visitor | 3 |
| [generated/audio/0000/events-train-0000024.flac](generated/audio/0000/events-train-0000024.flac) | en | 120s | brief_visitor | 3 |
| [generated/audio/0000/events-train-0000025.flac](generated/audio/0000/events-train-0000025.flac) | zh | 30s | acoustic_return | 2 |
| [generated/audio/0000/events-train-0000026.flac](generated/audio/0000/events-train-0000026.flac) | zh | 120s | acoustic_return | 3 |
| [generated/audio/0000/events-train-0000027.flac](generated/audio/0000/events-train-0000027.flac) | en | 30s | acoustic_return | 2 |
| [generated/audio/0000/events-train-0000029.flac](generated/audio/0000/events-train-0000029.flac) | zh | 30s | gradual_entry | 2 |
| [generated/audio/0000/events-train-0000030.flac](generated/audio/0000/events-train-0000030.flac) | zh | 120s | gradual_entry | 3 |
| [generated/audio/0000/events-train-0000031.flac](generated/audio/0000/events-train-0000031.flac) | en | 30s | gradual_entry | 2 |
| [generated/audio/0000/events-train-0000032.flac](generated/audio/0000/events-train-0000032.flac) | en | 120s | gradual_entry | 3 |
| [generated/audio/0000/events-train-0000033.flac](generated/audio/0000/events-train-0000033.flac) | zh | 30s | turnover | 3 |
| [generated/audio/0000/events-train-0000034.flac](generated/audio/0000/events-train-0000034.flac) | zh | 120s | turnover | 3 |
| [generated/audio/0000/events-train-0000036.flac](generated/audio/0000/events-train-0000036.flac) | en | 120s | turnover | 3 |
| [generated/audio/0000/events-train-0000037.flac](generated/audio/0000/events-train-0000037.flac) | zh | 30s | long_return | 2 |
| [generated/audio/0000/events-train-0000038.flac](generated/audio/0000/events-train-0000038.flac) | zh | 120s | long_return | 3 |
| [generated/audio/0000/events-train-0000039.flac](generated/audio/0000/events-train-0000039.flac) | en | 30s | long_return | 2 |
| [generated/audio/0000/events-train-0000041.flac](generated/audio/0000/events-train-0000041.flac) | zh | 30s | repeated_return | 2 |
| [generated/audio/0000/events-train-0000042.flac](generated/audio/0000/events-train-0000042.flac) | zh | 120s | repeated_return | 3 |
| [generated/audio/0000/events-train-0000043.flac](generated/audio/0000/events-train-0000043.flac) | en | 30s | repeated_return | 2 |
| [generated/audio/0000/events-train-0000044.flac](generated/audio/0000/events-train-0000044.flac) | en | 120s | repeated_return | 3 |
| [generated/audio/0000/events-train-0000045.flac](generated/audio/0000/events-train-0000045.flac) | zh | 30s | overlap_entry | 2 |
| [generated/audio/0000/events-train-0000046.flac](generated/audio/0000/events-train-0000046.flac) | zh | 120s | overlap_entry | 3 |
| [generated/audio/0000/events-train-0000047.flac](generated/audio/0000/events-train-0000047.flac) | en | 30s | overlap_entry | 2 |
| [generated/audio/0000/events-train-0000050.flac](generated/audio/0000/events-train-0000050.flac) | zh | 120s | brief_visitor | 3 |
| [generated/audio/0000/events-train-0000052.flac](generated/audio/0000/events-train-0000052.flac) | en | 120s | brief_visitor | 3 |
| [generated/audio/0000/events-train-0000053.flac](generated/audio/0000/events-train-0000053.flac) | zh | 30s | acoustic_return | 2 |
| [generated/audio/0000/events-train-0000054.flac](generated/audio/0000/events-train-0000054.flac) | zh | 120s | acoustic_return | 3 |
| [generated/audio/0000/events-train-0000055.flac](generated/audio/0000/events-train-0000055.flac) | en | 30s | acoustic_return | 2 |
| [generated/audio/0000/events-train-0000056.flac](generated/audio/0000/events-train-0000056.flac) | en | 120s | acoustic_return | 3 |
| [generated/audio/0000/events-train-0000057.flac](generated/audio/0000/events-train-0000057.flac) | zh | 30s | gradual_entry | 2 |
| [generated/audio/0000/events-train-0000058.flac](generated/audio/0000/events-train-0000058.flac) | zh | 120s | gradual_entry | 3 |
| [generated/audio/0000/events-train-0000059.flac](generated/audio/0000/events-train-0000059.flac) | en | 30s | gradual_entry | 2 |
| [generated/audio/0000/events-train-0000060.flac](generated/audio/0000/events-train-0000060.flac) | en | 120s | gradual_entry | 3 |
| [generated/audio/0000/events-train-0000061.flac](generated/audio/0000/events-train-0000061.flac) | zh | 30s | turnover | 3 |
| [generated/audio/0000/events-train-0000062.flac](generated/audio/0000/events-train-0000062.flac) | zh | 120s | turnover | 3 |
| [generated/audio/0000/events-train-0000065.flac](generated/audio/0000/events-train-0000065.flac) | zh | 30s | long_return | 2 |
| [generated/audio/0000/events-train-0000066.flac](generated/audio/0000/events-train-0000066.flac) | zh | 120s | long_return | 3 |
| [generated/audio/0000/events-train-0000067.flac](generated/audio/0000/events-train-0000067.flac) | en | 30s | long_return | 2 |
| [generated/audio/0000/events-train-0000068.flac](generated/audio/0000/events-train-0000068.flac) | en | 120s | long_return | 3 |
| [generated/audio/0000/events-train-0000069.flac](generated/audio/0000/events-train-0000069.flac) | zh | 30s | repeated_return | 2 |
| [generated/audio/0000/events-train-0000070.flac](generated/audio/0000/events-train-0000070.flac) | zh | 120s | repeated_return | 3 |
| [generated/audio/0000/events-train-0000071.flac](generated/audio/0000/events-train-0000071.flac) | en | 30s | repeated_return | 2 |
| [generated/audio/0000/events-train-0000073.flac](generated/audio/0000/events-train-0000073.flac) | zh | 30s | overlap_entry | 2 |
| [generated/audio/0000/events-train-0000074.flac](generated/audio/0000/events-train-0000074.flac) | zh | 120s | overlap_entry | 3 |
| [generated/audio/0000/events-train-0000075.flac](generated/audio/0000/events-train-0000075.flac) | en | 30s | overlap_entry | 2 |
| [generated/audio/0000/events-train-0000078.flac](generated/audio/0000/events-train-0000078.flac) | zh | 120s | brief_visitor | 3 |
| [generated/audio/0000/events-train-0000080.flac](generated/audio/0000/events-train-0000080.flac) | en | 120s | brief_visitor | 3 |
| [generated/audio/0000/events-train-0000081.flac](generated/audio/0000/events-train-0000081.flac) | zh | 30s | acoustic_return | 2 |
| [generated/audio/0000/events-train-0000082.flac](generated/audio/0000/events-train-0000082.flac) | zh | 120s | acoustic_return | 3 |
| [generated/audio/0000/events-train-0000083.flac](generated/audio/0000/events-train-0000083.flac) | en | 30s | acoustic_return | 2 |
