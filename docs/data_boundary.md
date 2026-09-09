# 数据边界

数据集注册和准备归 `audio-data-contract` 管理，本项目负责训练时的数据选择、
混合、增强及输入渲染。

- `DatasetSpec` / Catalog：数据 ID、版本、split、清单 artifact 与根目录别名。
- `roots.json`：本机路径映射；不写入可移植的 AudioRecord。
- `AudioRecord`：任务、音频引用与槽位顺序、文本、语言、标签和真实热词。
- `data/records.py`：记录的运行时视图，以及 AudioExample 到 ms-swift 输入的转换。
- `data/online_dataset.py`：在线渲染 prompt 和答案、采样热词。
- `data/catalog_dataset.py`：从 Catalog 构造记录，按需加载、切段、重采样和增强。
- `data/catalog_sampler.py`：SFT 来源混合、逻辑分片轮换、时长/槽位组批和采样状态。
- `integrations/ms_swift/catalog_loader.py`：将 batch 计划和恢复游标接入训练循环。

Lhotse 负责音频清单、切段与解码；本项目在内存中做波形及特征增强。增强后的 WAV
bytes 直接传给模板，不写临时音频文件。验证来源显式配置，每次确定性渲染，关闭
训练增强；不再导出固定消息文件。

原 ShareGPT 生成器、SampleIndex v1 兼容层、离线索引构造、WAV 缓存落盘及独立
离线采样工具已移除。Catalog 中现有的 `audio-record/1.0` artifact 仍然可以消费，
但本项目不负责生成它。

SFT batch 按音频槽位数分组；GRPO 保留原生生成分组采样器。当前元数据驻留内存，
不支持物理 shard 按片懒加载。具体配置和限制见 [在线训练数据](online_training_data.md)。

vLLM 的命名评测注册表继续使用主分支的独立实现；历史模型的 prompt 和 checkpoint
推理适配也继续保留，它们不参与训练数据生成。
