"""
Instantiation utilities for callbacks and loggers (Hydra-based).

How it works
- Expects a DictConfig where each item is a Hydra-constructable object (has `_target_`).
- Iterates and instantiates each entry; non-Dict entries are skipped silently.

YAML examples
Callbacks (configs/callbacks/default.yaml)
```yaml
model_checkpoint:
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  monitor: "val/loss_best"
  mode: "min"
periodic_inference:
  _target_: src.callbacks.periodic_inference.PeriodicInferenceCallback
  interval_steps: 1000
```
Loggers (configs/logger/tensorboard.yaml)
```yaml
tensorboard:
  _target_: lightning.pytorch.loggers.TensorBoardLogger
  save_dir: ${paths.output_dir}/tb
  name: ${task_name}
```

CLI overrides
- Pick a logger preset:
  python -m src.scripts.train logger=tensorboard
- Add/override a callback param on the fly:
  python -m src.scripts.train callbacks.periodic_inference.interval_steps=500
"""
import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        error_msg = "Callbacks config must be a DictConfig!"
        raise TypeError(error_msg)

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        error_msg = "Logger config must be a DictConfig!"
        raise TypeError(error_msg)

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
