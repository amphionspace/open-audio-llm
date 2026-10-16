# 03-final

输入：10% 插值模型 `tasks/01-merge/attempts/001/artifacts/base-10`；配置 `configs/eval-final.yaml`。

做法：部署推理栈（vLLM，GPU 3）跑结束面板 meeting-180s、CHiME-6、fixed338、AISHELL-5 dev、CS-Dialogue test，与 Emilia2 轮 6000 步在同模板同数据上的结束面板对照（不重测 6000 步）。

完成条件：五个面板都产出 summary.json，W&B 远端核验通过。

进展：001 失败——vLLM 启动做显存测算时，同卡上的声纹提取任务结束、释放显存，触发 vLLM 的显存一致性断言（与模型无关）；002 在同卡空闲后重跑，四个面板完成，CHiME-6 因 `configs/data/chime6-dev.yaml` 的 roots 相对路径（从 zh-overlap 复制时未改）找不到文件而失败；改为 zh-overlap 的 roots 绝对路径后，003 漏了复用设置、启动即停（退出码 143）；004 复用 002 的四个面板，只补跑 CHiME-6，完成。

## 结果（2026-10-15）

结论：不采用。只在选模面板（fixed338）上变好，最终面板多数变差，合成会议明显变差。

| 面板 | Emilia2 6000 步 | 10% 插值 |
| --- | --- | --- |
| fixed338 · AliMeeting | 20.81 | **20.00** |
| fixed338 · AISHELL-4 | 18.38 | **17.48** |
| AliMeeting test | **21.10** | 21.64 |
| AISHELL-4 test | 23.55 | **22.79** |
| AMI test | **33.22** | 35.14 |
| 合成中文 / 英文 / 中英 | **10.18 / 12.59 / 11.29** | 18.14 / 18.02 / 15.96 |
| AISHELL-5 dev | **27.79** | 30.36 |
| CS-Dialogue test | 7.25 | 7.16 |
| CHiME-6 dev（仅报告） | **58.48** | 68.80 |

均为不计复读 cpER（%）。复读也普遍增多（AISHELL-5 18 → 21，CHiME-6 4 → 15）。

证据：`attempts/004/status.json`（各面板 summary 来源）；W&B `weight-interp-20261014-final-base-10-{meeting180,fixed338,aishell5_dev,cs_dialogue_test}-001`（002 内）与 `...-chime6-001`（004 内），远端均已核验上传指标。
