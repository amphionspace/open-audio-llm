# 合并 LoRA 002 / checkpoint-8000

用与训练内评测相同的 CPU 合并代码，把 8000 步 adapter 合并进统一格式 checkpoint-1000。产物用于 A800 同机评测、全参训练起点和 loss 打分。合并模型可由对象存储中的 adapter 与基座重新生成，未单独上传。
