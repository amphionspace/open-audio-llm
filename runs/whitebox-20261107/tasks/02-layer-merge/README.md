# 02 免训练合并

- 输入：S = SFT 073 avg3；v15 avg3、v15 插值、v16 avg3（`scripts/layer_merge.py`）；v17 avg3、v18 avg3（`scripts/branch_merge.py`，两者都从 sum-full 起训）。合并都在 fp32 累加后转回 bf16，权重写到 `/workspace/data/ae-runs/whitebox-merge-<名称>/checkpoint`（未上传，可重建）。
- 评测：GRPO 仓库 `configs/eval-whitebox-<名称>-{core,asrfull,hotword,report}`，与 v16 插值的核心评测同一套集合。
- 配方（Δ15 = v15 avg3 − S，Δ16 = v16 avg3 − v15 插值）：audio-full / llm-full / mid-full（对应部分取 Δ16 全量，其余与插值相同）、sum-full（Δ15 + Δ16）、sum-1.25、sum-1.5、d15-0.75（0.75 Δ15 + Δ16）、branch-sum（sum-full + Δ17 + Δ18）、branch-avg（sum-full + (Δ17 + Δ18)/2）。
- 结果与结论见 `docs/experiments/2026-11-07-whitebox/README.md` 第二、五节：sum-full 最好；外推到 1.25 以上开始变差；分模块全量都落在插值与 avg3 之间；v17/v18 分支合并只在误报与漏识之间移动。
