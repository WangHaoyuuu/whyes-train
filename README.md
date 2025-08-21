## Whyes-Train 使用指南（简明版）

本框架基于 PyTorch Lightning + Hydra，提供开箱即用的训练、评估与配置管理能力。适合新手快速上手，也支持进阶自定义。

### 1) 框架简介与主要功能
- **模块化**: `LightningModule` 与 `LightningDataModule` 解耦模型与数据。
- **配置化**: 通过 Hydra 组合/覆盖配置，复现实验更简单。
- **回调/日志**: 早停、模型检查点、学习率监控；可选 TensorBoard/W&B 等日志器。
- **多设备**: 支持 CPU、GPU、MPS(macOS) 与 DDP 分布式训练。

### 2) 环境配置要求
- Python 3.9+
- 依赖管理: uv（推荐）。如需 GPU，请按 PyTorch 官网安装对应 CUDA 版本。

安装 uv 与项目依赖：
```bash
pip install uv
git clone <your-repo-url>
cd whyes-train
make install
```

### 3) 快速入门步骤（3 分钟）
1. 默认配置启动 MNIST 训练（CPU）
```bash
uv run src/scripts/train.py
```
2. 使用不同设备
```bash
# 单 GPU
uv run src/scripts/train.py trainer=gpu

# macOS MPS（Apple 芯片）
uv run src/scripts/train.py trainer=mps

# 多 GPU（DDP，需按机器实际修改 devices）
uv run src/scripts/train.py trainer=ddp trainer.devices=4
```
3. 常用参数临时覆盖（命令行直接改）
```bash
uv run src/scripts/train.py \
  model.optimizer.lr=0.01 \
  data.batch_size=256 \
  trainer.max_epochs=20 \
  logger=tensorboard
```
4. 输出与权重位置
- 运行日志与输出目录：`logs/<task_name>/runs/<时间戳>/`
- 检查点：`logs/<task_name>/runs/<时间戳>/checkpoints/`

5. 评估已有权重
```bash
uv run src/scripts/eval.py ckpt_path=/absolute/path/to/checkpoint.ckpt
```

### 4) 核心模块与配置说明（最常用）

- `train.yaml`（主控，位于 `src/configs/train.yaml`）
  - 通过 defaults 组合各配置组：`data`、`model`、`trainer`、`callbacks`、`logger`、`paths`、`extras`、`hydra` 等。
  - 开关项：`train`（是否训练）、`test`（训练后是否测试）、`ckpt_path`（断点续训路径）。

- 数据模块 `data/*`（例：`src/configs/data/mnist.yaml`）
  - 绑定 `src/data/mnist_datamodule.py`。
  - 关键项：`batch_size`、`train_val_test_split`、`num_workers`、`pin_memory`。
  - 注意：`batch_size` 需能被设备数整除（DDP 情况下会自动校验）。

- 模型模块 `model/*`（例：`src/configs/model/mnist.yaml`）
  - 绑定 `src/models/mnist_module.py` 与网络 `src/models/components/simple_dense_net.py`。
  - 优化器与调度器：`model.optimizer.*`、`model.scheduler.*`。
  - 监控指标：`configure_optimizers.monitor: "val/loss"`（回调中还会监控 `val/acc`）。
  - 可选编译：`torch_compile: false`（PyTorch 2.0+ 可设为 true）。

- 训练器 `trainer/*`
  - `trainer=default`（CPU, 10 epochs）
  - `trainer=gpu`（单 GPU）
  - `trainer=mps`（macOS MPS）
  - `trainer=ddp`（多 GPU 分布式，默认 devices=4，按需覆盖）

- 回调 `callbacks/default.yaml`
  - `ModelCheckpoint`: 目录 `${paths.output_dir}/checkpoints`，监控 `val/acc`，`save_last: true`。
  - `EarlyStopping`: 监控 `val/acc`，`mode: max`。
  - `LearningRateMonitor`、`ModelSummary`、`RichProgressBar` 已启用。

- 日志器 `logger/*`
  - `logger=tensorboard` 使用 `src/configs/logger/tensorboard.yaml`，保存到 `${paths.output_dir}/tensorboard/`。

- 路径与 Hydra
  - `src/configs/paths/default.yaml`: `output_dir=${hydra:runtime.output_dir}`。
  - `src/configs/hydra/default.yaml`: 运行目录为 `logs/<task_name>/runs/<时间戳>/`。

### 5) 常见问题（FAQ）
- 训练只用 CPU？
  - 显式指定：`trainer=gpu` 或 `trainer=mps`。检查 `torch.cuda.is_available()` 或 Apple MPS 环境。

- 显存不够（OOM）怎么办？
  - 降低 `data.batch_size`；启用混合精度：`trainer.precision=16`；
  - 使用梯度累积：`trainer.accumulate_grad_batches=4`；
  - 降低网络宽度（修改 `model.net.*`）。

- DDP 报错 batch size 不可整除？
  - `batch_size` 必须能被设备数整除（会自动检查）。将 `data.batch_size` 调整为 `devices` 的倍数。

- 模型/日志输出在哪里？
  - 查看 `logs/<task_name>/runs/<时间戳>/`，检查点在其 `checkpoints/` 子目录。

- 如何断点续训或评估？
  - 续训：`uv run src/scripts/train.py ckpt_path=/abs/path/to/ckpt.ckpt`
  - 评估：`uv run src/scripts/eval.py ckpt_path=/abs/path/to/ckpt.ckpt`

### 6) 进阶使用技巧
- 命令行覆盖（推荐调参方式）
```bash
uv run src/scripts/train.py \
  model.optimizer.lr=0.005 \
  data.batch_size=256 \
  trainer.max_epochs=30 \
  callbacks.early_stopping.patience=10
```

- 自定义实验配置
```bash
cp src/configs/experiment/example.yaml src/configs/experiment/my_exp.yaml
# 编辑 my_exp.yaml 后：
uv run src/scripts/train.py experiment=my_exp
```

- 超参数搜索（Hydra 多运行）
```bash
uv run src/scripts/train.py -m model.optimizer.lr=0.001,0.01 data.batch_size=64,128
```

- 更换/新增数据集与模型
  - 在 `src/data/` 与 `src/models/` 新增模块，创建对应 `src/configs/data/<name>.yaml` 与 `src/configs/model/<name>.yaml`，运行时通过 `data=<name> model=<name>` 指定。

如需更多细节，直接阅读：`src/scripts/train.py`、`src/scripts/eval.py`、`src/models/mnist_module.py`、`src/data/mnist_datamodule.py`。


