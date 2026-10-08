# 数据迁到本机（8×A800）

输入：LoRA 002 的生效数据配置、原机 catalog 与 2026-10-06 补传数据。原 catalog 字节不变，只改路径别名；原机目录被拆分的别名用逐文件硬链接镜像（audio-data-contract 拒绝解析到根目录外的符号链接）。

结果：执行 004 通过（433 个硬链接；有 SHA256 的 4,015 个文件一致，会议 398 个和 events 549 个音频逐条核对音频头）。001 漏了会议音频索引、002/003 用符号链接被拒绝，均保留作记录。

证据：`attempts/004/artifacts/verification.json`；配置 [localize-data.yaml](../../configs/localize-data.yaml)。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-10-06T14:34:07.706194+00:00；结束：2026-10-06T14:35:13.816664+00:00；退出码：0。

<!-- /execution-status -->
