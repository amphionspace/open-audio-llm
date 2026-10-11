# 中文对话配比的数据配置

在上一轮（中文远场配比）冻结配置上：英文会议 6 → 3、旧版 1–5 人合成 10 → 4，新增 AISHELL-5 车载远场 5、CS-Dialogue 两人中英混说 4；其余来源和权重不变，种子 46。已注册数据从 `/workspace/data/registry` 的副本读取。

执行 001 失败：上一轮 roots 指向原位置，与新数据 roots（已指向注册副本）冲突；改为先替换注册副本再检查后，执行 002 通过。

证据：`attempts/<编号>/artifacts/train-data.yaml`、`report.json`。
