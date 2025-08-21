"""
Training entrypoint (Hydra + Lightning).

What this script does
- Instantiates: datamodule (cfg.data), model (cfg.model), callbacks (cfg.callbacks), loggers (cfg.logger), trainer (cfg.trainer)
- Trains if cfg.train is true; tests if cfg.test is true (uses best checkpoint if available)
- Logs hyperparameters and returns a dict of metrics for HPO (cfg.optimized_metric)

Key config sections (Hydra)
- cfg.data: LightningDataModule config (see src/data/datamodule.py)
- cfg.model: LightningModule config (see src/models/module.py)
- cfg.trainer: Lightning Trainer config (see src/configs/trainer/*.yaml)
- cfg.callbacks: Dict of callback configs (see src/configs/callbacks/*.yaml)
- cfg.logger: Dict of logger configs (see src/configs/logger/*.yaml)
- cfg.train / cfg.test: booleans controlling fit/test
- cfg.ckpt_path: optional checkpoint path for resuming training or testing
- cfg.optimized_metric: metric name to return for hyperparameter search

Quickstart commands
- Basic training (CPU):
  python -m src.scripts.train
- Use GPU trainer + TensorBoard logger:
  python -m src.scripts.train trainer=gpu logger=tensorboard
- Set data paths and batch size:
  python -m src.scripts.train data.batch_size=128 data.train_data_path=/path/train data.val_data_path=/path/val data.test_data_path=/path/test
- Step-based schedule tuning:
  python -m src.scripts.train trainer.max_steps=10000 trainer.val_check_interval=1000 trainer.log_every_n_steps=100
- Resume from checkpoint:
  python -m src.scripts.train ckpt_path=/path/to/checkpoints/last.ckpt
- Train then test with best ckpt:
  python -m src.scripts.train train=true test=true
- Enable/adjust periodic inference callback (runs lightweight inference during training):
  python -m src.scripts.train callbacks.periodic_inference.interval_steps=500

Notes
- All sections are Hydra-composable; you can pick alternative trainer/logger configs with `+` or direct overrides.
- The returned value is suitable for Hydra/Optuna sweeps via cfg.optimized_metric (e.g., "val/loss_best").
"""
from typing import Any, Optional

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
