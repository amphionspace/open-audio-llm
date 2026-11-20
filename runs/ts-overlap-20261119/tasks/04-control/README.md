# 04 只解冻音频塔（对照组）

## 作用

用来区分"解冻音频塔"本身和"干净原句 KL"各自的作用。

- 配置与 [03](../03-clean-kl/README.md) 完全相同，只是数据配置里没有 `clean_teacher_kl_weight`。
  - 配置：[../../configs/train-control.yaml](../../configs/train-control.yaml)，数据配置 [../../configs/train-data-control.yaml](../../configs/train-data-control.yaml)。
- 历史上 v95 之前音频塔一直随文本交叉熵更新，重叠转写没动。所以预期本组的重叠转写基本不变。
- 起点（11.17）和数据都与那些轮次不同，所以仍要跑这一组。

## 完成条件

同 03。评测与 03 配对比较。

## 进展

见 `attempts/`。
