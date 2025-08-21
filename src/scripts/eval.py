"""
Evaluation entrypoint (Hydra + Lightning).

What this script does
- Loads datamodule (cfg.data), model (cfg.model), loggers (cfg.logger), and trainer (cfg.trainer)
- Requires cfg.ckpt_path to evaluate a checkpoint on the test set
- Returns metrics via trainer.callback_metrics (e.g., test/loss)

Quickstart commands
- CPU evaluation:
  python -m src.scripts.eval ckpt_path=/path/to/model.ckpt
- GPU evaluation:
  python -m src.scripts.eval ckpt_path=/path/to/model.ckpt trainer=gpu
- Change data paths or batch size on the fly:
  python -m src.scripts.eval ckpt_path=/path/to/model.ckpt data.batch_size=128 data.test_data_path=/path/test

Tips
- You can select alternative logger/trainer presets via overrides (e.g., logger=wandb trainer=gpu).
- For predictions instead of testing, use `trainer.predict(...)`—see inside the file for a starting point.
"""
from typing import Any

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    if cfg.ckpt_path is None or cfg.ckpt_path == "":
        error_msg = "Please provide a checkpoint path for evaluation!"
        raise ValueError(error_msg)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
