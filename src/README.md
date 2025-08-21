# src/ 使用指南（Hydra + Lightning 快速上手）

本指南帮助你从零开始跑通训练与评估，并教你如何通过配置灵活修改数据、模型、训练器、回调与日志器。

## 0. TL;DR 快速开始

- 训练（CPU）
  ```bash
  python -m src.scripts.train
  ```
- 训练（GPU+TensorBoard）
  ```bash
  python -m src.scripts.train trainer=gpu logger=tensorboard
  ```
- 指定数据路径与批大小
  ```bash
  python -m src.scripts.train \
    data.batch_size=128 \
    data.train_data_path=/data/train \
    data.val_data_path=/data/val \
    data.test_data_path=/data/test
  ```
- 步进式训练常用参数
  ```bash
  python -m src.scripts.train \
    trainer.max_steps=10000 \
    trainer.val_check_interval=1000 \
    trainer.log_every_n_steps=100
  ```
- 断点续训/评估
  ```bash
  python -m src.scripts.train ckpt_path=/path/to/last.ckpt
  python -m src.scripts.eval ckpt_path=/path/to/best.ckpt trainer=gpu
  ```
- 开启/调整“每1000步推理一次”的回调
  ```bash
  python -m src.scripts.train callbacks.periodic_inference.interval_steps=500
  ```

## 1. 目录结构（src/）

- `scripts/` 入口脚本
  - `train.py` 训练入口：实例化 data/model/callbacks/loggers/trainer，执行 fit/test
  - `eval.py` 评估入口：加载 ckpt，对 test set 评估
- `data/` 数据模块
  - `datamodule.py` 通用 LightningDataModule 模板（不做自动切分），你需要在 `setup()` 中挂上 `self.data_train/val/test`
- `models/` 模型模块
  - `module.py` 通用 LightningModule：按 step 计数，记录 `train/loss`、`val/loss`、`test/loss` 与 `val/loss_best`
  - `components/simple_dense_net.py` 一个极简网络示例
- `callbacks/` 回调
  - `periodic_inference.py` 每 N 步轻量推理回调（默认已接入）
- `utils/` 实用函数
  - `instantiators.py` 通过 Hydra 按配置实例化回调与日志器
- `configs/` 配置目录（Hydra）
  - `train.yaml`、`eval.yaml` 为顶层入口配置
  - `data/`、`model/`、`trainer/`、`callbacks/`、`logger/` 等子配置可自由组合

## 2. 训练：如何启动与控制

入口：`src/scripts/train.py`，核心配置键：
- `data`：数据模块配置，见 `configs/data/datamodule.yaml`
- `model`：LightningModule 配置，见 `configs/model/module.yaml`
- `trainer`：Lightning Trainer 配置，见 `configs/trainer/*.yaml`
- `callbacks`：回调集合，见 `configs/callbacks/*.yaml`
- `logger`：日志器集合，见 `configs/logger/*.yaml`
- `train/test`：是否执行训练/测试（bool）
- `ckpt_path`：可选，续训或评估的 checkpoint 路径
- `optimized_metric`：用于 HPO 的指标名（一般为 `val/loss_best`）

常见覆盖（override）示例：
- 指定 GPU 训练与日志器：`trainer=gpu logger=tensorboard`
- 调整步进计划：`trainer.max_steps=20000 trainer.val_check_interval=2000`
- 调整日志频率：`trainer.log_every_n_steps=50`
- 关闭某个回调：选择不含该回调的预设，或命令行 `callbacks=none`

## 3. 配置数据：DataModule

文件：`src/data/datamodule.py`。这是一个模板，不做数据下载/切分。你需要在 `setup(stage)` 中：
- 根据 `train/val/test` 阶段，给 `self.data_train`、`self.data_val`、`self.data_test` 赋值（任意 PyTorch Dataset）
- DataLoader 已经就绪，直接返回 Dataloader

配置示例（`configs/data/datamodule.yaml`）：
```yaml
_target_: src.data.datamodule.DataModule
batch_size: 64
num_workers: 4
pin_memory: true
train_data_path: /data/train
val_data_path: /data/val
test_data_path: /data/test
```
命令行覆盖：
```bash
python -m src.scripts.train data.batch_size=128 data.train_data_path=/data/train
```

## 4. 配置模型：LitModule + Net/Opt/Sched

文件：`src/models/module.py`。通用做法：
- 通过 Hydra 注入 `net`（任意 `torch.nn.Module`）
- 在 `model_step(batch)` 中实现你的损失计算，并返回 `(loss, outputs)`
- 可选实现 `sample()` 以支持生成回调

最小 YAML（`configs/model/module.yaml`）示例（搭配 `SimpleDenseNet` 的正确参数名）：
```yaml
_target_: src.models.module.LitModule
net:
  _target_: src.models.components.simple_dense_net.SimpleDenseNet
  input_size: 784
  lin1_size: 256
  lin2_size: 256
  lin3_size: 256
  output_size: 784
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-3
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 1000
torch_compile: false
max_steps: 10000
log_every_n_steps: 100
val_check_interval: 1000
```
覆盖示例：
```bash
python -m src.scripts.train model.optimizer.lr=3e-4 model.scheduler=null
```

## 5. 配置训练器：步进式训练为主

参考 `configs/trainer/default.yaml` 或 `gpu.yaml`：
- `max_steps`：训练总步数上限
- `val_check_interval`：多少步验证一次
- `log_every_n_steps`：多少步记录一次日志
- `accelerator`/`devices`：设备类型与数量

示例：
```bash
python -m src.scripts.train trainer=gpu trainer.max_steps=50000 trainer.val_check_interval=2000
```

## 6. 回调与日志：监控与可视化

- Checkpoint：`ModelCheckpoint` 监控 `val/loss_best`，文件名 `epoch_{epoch:03d}`
- 早停：`EarlyStopping`（如需）
- 周期推理：`PeriodicInferenceCallback`（默认开启，每 1000 步）
- 日志器：TensorBoard/W&B/CSV 等任选其一或多选

按需启停/调参：
```bash
python -m src.scripts.train callbacks.periodic_inference.interval_steps=500
python -m src.scripts.train callbacks=none
python -m src.scripts.train logger=wandb
```

## 7. 评估：加载 ckpt 在 test set 上评测

入口：`src/scripts/eval.py`，必填 `ckpt_path`：
```bash
python -m src.scripts.eval ckpt_path=/path/to/best.ckpt trainer=gpu
```

## 8. 常见配方（Recipes）

- 小批量快速联调：`trainer.max_steps=200 trainer.val_check_interval=100 trainer.log_every_n_steps=10`
- 测试回调触发：`callbacks.periodic_inference.interval_steps=50`
- 仅验证/仅测试：`train=false test=true`（或使用 `trainer.validate/test` 相关入口）

## 9. 故障排查（Troubleshooting）

- `Batch size ... not divisible by world_size`：调整 `data.batch_size` 使其被设备数整除
- 没有任何推理日志：确认 datamodule 提供 `val_dataloader()` 或在模型实现 `sample()`
- 评估时报错：确保传入 `ckpt_path`，且模型/数据配置与训练时兼容

## 10. 模板质量评估（是否是“好的训练模板”）

结论：就通用性、可组合性与“按 step 的生成式训练”范式而言，本模板是一个成熟、可扩展的起点，具备生产所需的关键部件（Hydra 组合、Lightning 模块化、回调/日志、多日志器、可选编译、周期性推理）。需要你补齐具体任务相关的 Dataset 与 `model_step()` 实现即可。

亮点：
- 结构清晰：脚本（train/eval）、数据（DataModule）、模型（LitModule）、配置（Hydra）、回调与日志器分层明确。
- 组合灵活：所有对象经由 `_target_` 构造，支持命令行覆盖与实验配方复用。
- 步进式训练：以 `val_check_interval`、`log_every_n_steps` 控制频率，`val/loss_best` 统一作为监控指标。
- 可观测性：默认集成 LR 监控、模型摘要、富进度条，且支持多种日志器预设。
- 在线推理：`PeriodicInferenceCallback` 在训练中周期性做轻量推理，便于快速观察模型行为。

需要你介入/留白点：
- `src/data/datamodule.py` 不包含具体数据装载逻辑，需在 `setup()` 中赋值 `self.data_train/val/test`。
- `src/models/module.py` 的 `model_step()` 和实际损失函数是占位，需要你按任务实现；如需可视化样本，补充 `sample()`。
- `configs/model/module.yaml` 和 `configs/trainer/default.yaml` 中都存在 `max_steps/val_check_interval/log_every_n_steps` 字段：
  - 实际“训练调度频率”由 `trainer.*` 生效；
  - `model.log_every_n_steps` 用于训练步内日志频率；
  - `model.max_steps/val_check_interval` 目前仅存入 hparams 未直接驱动训练流程（信息性字段）。建议以 `trainer.*` 为准。

## 11. 配置 — 代码 一一对应关系（最常用条目）

全局入口
- `configs/train.yaml` → `src/scripts/train.py`：按 `defaults` 组合 `data/model/callbacks/logger/trainer/paths/extras/hydra`，并驱动 `fit/test`。
- `configs/eval.yaml` → `src/scripts/eval.py`：同上，但要求 `ckpt_path` 才能执行 `trainer.test(...)`。

数据（cfg.data → DataModule）
- `_target_: src.data.datamodule.DataModule` → 类定义与构造：`data_dir/batch_size/num_workers/pin_memory/train_data_path/val_data_path/test_data_path` 一一映射到构造参数与 `self.hparams`。
- `batch_size` → 在 `setup()` 中按 `trainer.world_size` 自动整除得到 `batch_size_per_device`，并在各 `*_dataloader()` 中使用。
- `num_workers/pin_memory` → 传入 `DataLoader` 构造。
- `train/val/test_data_path` → 供你在 `setup(stage)` 中装载具体 Dataset 使用。

模型（cfg.model → LitModule）
- `net._target_` → 作为 `self.net` 前向网络；例如 `SimpleDenseNet` 的参数名为 `input_size/lin1_size/lin2_size/lin3_size/output_size`。
- `optimizer._target_ + 超参` → 在 `configure_optimizers()` 中以 `self.trainer.model.parameters()` 初始化。
- `scheduler._target_ + 超参` → 在 `configure_optimizers()` 中与优化器一同返回，`monitor="val/loss_best"`、`interval="step"`。
- `torch_compile` → 在 `setup('fit')` 时调用 `torch.compile(self.net)`。
- 日志与指标：
  - `training_step()` 每 `model.log_every_n_steps` 步记录 `train/loss`；
  - `validation_step()`/`on_validation_epoch_end()` 记录 `val/loss` 与 `val/loss_best`；
  - `test_step()` 记录 `test/loss`；
  - `val/loss_best` 被 `ModelCheckpoint` 与 LR `scheduler` 共同监控。

训练器（cfg.trainer → lightning.Trainer）
- `max_steps/val_check_interval/log_every_n_steps/accelerator/devices` → 由 `train.py` 直接实例化 `Trainer(**cfg.trainer)` 生效。
- `trainer=gpu` 预设（`configs/trainer/gpu.yaml`）会覆盖 `accelerator=gpu`。

回调（cfg.callbacks → instantiate_callbacks）
- 组合方式：`configs/callbacks/default.yaml` 的 `defaults:` 列表引入通用回调预设；同文件中的键（如 `model_checkpoint/early_stopping/periodic_inference`）按 `_target_` 被实例化。
- `model_checkpoint.monitor: val/loss_best` → 对应模型里 `on_validation_epoch_end()` 的记录；`dirpath: ${paths.output_dir}/checkpoints` 依赖 `paths` 与 `hydra`。
- `periodic_inference.*` → 对应 `src/callbacks/periodic_inference.py` 的回调参数：`interval_steps/max_batches/log_outputs`。

日志器（cfg.logger → instantiate_loggers）
- 任选其一或多选：如 `logger=tensorboard`（`configs/logger/tensorboard.yaml`）、`logger=wandb`（`configs/logger/wandb.yaml`）。
- `save_dir/name/project` 等字段按各日志器类定义映射，实例化发生在 `utils/instantiators.py`。

路径与运行目录（cfg.paths / cfg.hydra / cfg.extras）
- `paths.default.yaml`：`data_dir/log_dir/output_dir` 等；`output_dir=${hydra:runtime.output_dir}`。
- `hydra.default.yaml`：`run.dir/sweep.dir` 指定输出路径模式，文件日志位置 `.../${task_name}.log`。
- `extras.default.yaml`：
  - `ignore_warnings` → 在 `utils.extras()` 中屏蔽 Python 警告；
  - `enforce_tags` → 运行前要求你填写标签；
  - `print_config` → 运行前用 Rich 打印配置树（也会保存成文件）。

指标与测试的对应
- 训练/验证/测试期间记录：`train/loss`、`val/loss`、`val/loss_best`、`test/loss`。
- `tests/test_eval.py` 断言 `test/loss` 存在且为有限值，与你的 `LitModule` 指标记录保持一致。

改进建议（可选）
- 去重 step 配置：长期建议仅保留 `trainer.*` 为训练调度“单一事实来源”，`model.*` 保留 `log_every_n_steps` 即可。
- 提供一个最小可运行的示例 DataModule（随机数据集）与 `LitModule.model_step()` 样例，便于一键验通。
- 增加多卡/混合精度范例配置与常见陷阱说明（如梯度累积、ZeRO）。
- 为 `PeriodicInferenceCallback` 增加示例 `sample()` 的占位实现与可视化片段（文本/图像）。

—— 完 ——