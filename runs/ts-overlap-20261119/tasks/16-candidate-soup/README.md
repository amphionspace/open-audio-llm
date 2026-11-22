# 16 soup 作为交付候选：与 11.17 逐项对比

## 要回答的问题

[任务 15](../15-grpo-soup/README.md) 的 soup（13 avg3 与 14 avg3 的权重平均，带 MTP）过了中文 TS 守门和英文误报门槛，重叠转写显著好于 11.17。交付前还要在 11.17 发布时评过的全部套件上逐项对比，看有没有别处退步；同时测 MTP 接受长度（交付包必须带 MTP）。

## 输入

- 模型：ae 模型缓存 `ts-overlap-grpo-soup-t13t14-avg3`（`model-9d858e123483d3131107c136`）。
- 套件：11.17（`whitebox-merge-v18a-es116inc`）2026-11-17 发布评测的 27 个 run，配置取自各 run 自己的 `run.json`。
  - 生成脚本：[make_candidate_configs.py](../../scripts/make_candidate_configs.py)，只改模型名、输出目录和服务地址。
  - 其中 v12、英文 TS、普通识别、客户 TS 四个已在任务 15 评过；本任务评其余 23 个，覆盖西语识别与 TS、背景人声、热词（中 / 西 / TS / 客户）、警务、跨语言注册、鼎桥场景、上下文纠偏、CHiME-4 等。
- 服务：与 11.17 发布评测相同，pack-mtp 的 `serve.sh`，不开 MTP。节点脚本：[run_candidate_evals.sh](../../scripts/run_candidate_evals.sh)。

## 完成条件

- 23 个套件执行完成，W&B 远端核验。
- 逐项列出与 11.17 的差别，标出明显退步的项。
- MTP 接受长度（中英识别、中文 TS、西语识别）不明显低于 11.17 包（2.82 / 2.71 / 2.78）。
- 是否替换交付版本由用户决定。

## 进展

- 2026-11-22：配置生成（23 个套件），排进 ACP `pt-0vbbz0mo` 的 runner（`19-candidate-soup.sh`），在任务 15 之后跑。
- 2026-11-22：MTP 接受长度测量，用 joint-mtp-spec 的 `spec_check.py`，协议与 11.17 包相同：k=2 对照关闭 MTP。配置 [spec-check-soup-zhen-ts.yaml](../../configs/spec-check-soup-zhen-ts.yaml)、[spec-check-soup-esasr.yaml](../../configs/spec-check-soup-esasr.yaml)，脚本 [run_soup_spec.sh](../../scripts/run_soup_spec.sh)，排在 runner `20-soup-spec.sh`，结果写入 attempts 002 / 003。
