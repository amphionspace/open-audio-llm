# dev 保真度变体

- 输入：`meeting_selection_panel@v1-20261008` 的 AliMeeting dev 138 窗、对应近讲录音、
  [configs/synth-v1-fidelity-alimeeting.yaml](../../configs/synth-v1-fidelity-alimeeting.yaml)、训练素材池。
- 做法：[scripts/build_dev_fidelity.py](../../scripts/build_dev_fidelity.py)，与训练同一份合成代码；生成 5 个变体
  （syn_acoustic、syn_fill_{near,far}_{alimeeting,aishell5}），音频写到
  `/workspace/data/datasets/meeting_synth_fidelity/versions/v1-20261031`，目录写到 `shared/catalog-fidelity`。
- 完成条件：5 个变体各 138 条，目录与 roots 写出。
- 进展：
  - [attempt 001](attempts/001/README.md) 作废：9/690 条整段全零（噪声库数字静音触发合成器分支缺陷，提交 0b3dfd7 修复，
    单测 `test_render_drops_digital_silence_from_the_noise_and_keeps_the_speech`）。
  - attempt 002 用修复后的代码重跑，命令同 001（[attempts/002/command.txt](attempts/002/command.txt)）；
    完成条件追加"无全零文件"。
