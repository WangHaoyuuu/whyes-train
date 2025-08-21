"""
PeriodicInferenceCallback: run lightweight inference every N steps during training.

Purpose
- Monitor basic generation/forward behavior without task-specific post-processing
- Safe in DDP (rank0-only), auto-switches eval()/train(), and uses no-grad

Parameters
- interval_steps (int): run every N global steps (default 1000)
- max_batches (int): number of mini-batches to sample from val dataloader when `sample()` is unavailable
- log_outputs (bool): optionally enable richer logging of outputs (extend as needed)

Enable via config (already wired in callbacks/default.yaml)
```yaml
periodic_inference:
  _target_: src.callbacks.periodic_inference.PeriodicInferenceCallback
  interval_steps: 1000
  max_batches: 1
```

CLI override examples
- Run more frequently:
  python -m src.scripts.train callbacks.periodic_inference.interval_steps=500
- Disable at runtime (choose a callbacks preset without it):
  python -m src.scripts.train callbacks=none

Tip
- Implement `def sample(self): ...` on your LightningModule to let this callback call your preferred sampling routine.
"""
from typing import Any

import torch
from lightning import Callback


class PeriodicInferenceCallback(Callback):
    """Run a lightweight inference routine every N training steps.

    This callback is designed to be generic and safe:
    - Runs only on the global zero process
    - Tries `pl_module.sample()` if available (unconditional generation)
    - Otherwise, tries a single forward pass on one validation batch if accessible
    - Logs simple scalar metrics so it does not depend on task-specific post-processing
    """

    def __init__(self, interval_steps: int = 1000, max_batches: int = 1, log_outputs: bool = False) -> None:
        super().__init__()
        self.interval_steps = int(interval_steps)
        self.max_batches = int(max_batches)
        self.log_outputs = bool(log_outputs)

    def on_train_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int) -> None:
        step = trainer.global_step
        if self.interval_steps <= 0:
            return
        # skip at step==0 and only run every `interval_steps`
        if step == 0 or (step % self.interval_steps) != 0:
            return
        # run only on rank 0 to avoid duplication in DDP
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return
        self._run_inference(trainer, pl_module)

    def _run_inference(self, trainer, pl_module) -> None:
        was_training = pl_module.training
        try:
            pl_module.eval()
            with torch.no_grad():
                # Always log the step as a heartbeat for the callback
                try:
                    pl_module.log("inference/step", float(trainer.global_step), on_step=True, on_epoch=False, prog_bar=False, logger=True)
                except Exception:
                    pass

                # Prefer calling `sample()` if the module implements it (common in generative models)
                if hasattr(pl_module, "sample") and callable(getattr(pl_module, "sample")):
                    try:
                        samples = pl_module.sample()  # user-defined signature
                        # Optionally log a simple scalar derived from samples, if tensor-like
                        try:
                            if torch.is_tensor(samples):
                                val = samples.float().abs().mean().item()
                                pl_module.log("inference/sample_abs_mean", val, on_step=True, on_epoch=False, prog_bar=False, logger=True)
                        except Exception:
                            pass
                        return
                    except Exception:
                        # fallback to val batch if sampling is not implemented or failed
                        pass

                # Fallback path: try to fetch a small batch from val_dataloader and run a forward pass
                dl = None
                try:
                    if trainer.datamodule is not None and hasattr(trainer.datamodule, "val_dataloader"):
                        dl = trainer.datamodule.val_dataloader()
                except Exception:
                    dl = None

                if dl is None:
                    return

                it = iter(dl)
                batches = 0
                while batches < max(1, self.max_batches):
                    try:
                        b = next(it)
                    except StopIteration:
                        break
                    batches += 1

                    # Try to extract inputs conservatively: assume (x, y) or x
                    x = b[0] if isinstance(b, (list, tuple)) else b
                    try:
                        out = pl_module(x)
                    except Exception:
                        break

                    # Log a very cheap summary statistic if tensor-like
                    try:
                        if torch.is_tensor(out):
                            val = out.float().abs().mean().item()
                            pl_module.log("inference/output_abs_mean", val, on_step=True, on_epoch=False, prog_bar=False, logger=True)
                    except Exception:
                        pass
        finally:
            if was_training:
                pl_module.train()