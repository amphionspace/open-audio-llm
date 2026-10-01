# 实验状态与同步记录

迁移核验通过，全部旧文件按大小和 inode 对照，未复制权重。

[映射](migration-plan.json)、[迁移前清单](migration-before.json)、[核验](migration-verification.json)。

历史状态与 W&B 核验在 [history/](history/)；进程日志在 [logs/](logs/)。其中旧“running”及“training_started=false”记录已过期或有冲突，不能视为当前状态。新执行由统一入口记录，W&B 远端核验文件在对应 attempts 目录。
