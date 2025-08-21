"""
Generic LightningModule for step-based generative training.

How to use
- Provide a backbone `net` (any torch.nn.Module) and define your loss in `model_step(batch)` by returning (loss, outputs).
- Optionally implement `sample(self)` for unconditional sampling; callbacks may call it (e.g., PeriodicInferenceCallback).
- The default logging scheme tracks train/val/test loss and `val/loss_best` for checkpointing & schedulers.

Constructor parameters
- net (torch.nn.Module): your network to be called in `forward`.
- optimizer (callable): e.g., torch.optim.AdamW configured via Hydra, called with `params=self.trainer.model.parameters()`.
- scheduler (callable or None): LR scheduler factory; monitored metric is `val/loss_best`, interval="step".
- torch_compile (bool): whether to compile the model in `setup('fit')`.
- max_steps (int): expected max training steps (used for reference/tuning).
- log_every_n_steps (int): frequency of on_step logging during training.
- val_check_interval (int): frequency of validation in steps.

Minimal Hydra example (configs/model/module.yaml)
```yaml
_target_: src.models.module.LitModule
net:
  _target_: src.models.components.simple_dense_net.SimpleDenseNet
  in_dim: 32
  hidden_dim: 64
  out_dim: 32
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

CLI overrides example
- Change learning rate:
  trainer=default model.optimizer.lr=3e-4
- Disable scheduler:
  model.scheduler=null
- Use a different net:
  model.net._target_=your.module.YourNet
"""
from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric


class LitModule(LightningModule):
    """Generic `LightningModule` template for generative models trained with step-based scheduling.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        torch_compile: bool,
        max_steps: int = 100000,
        log_every_n_steps: int = 100,
        val_check_interval: int = 1000,
    ) -> None:
        """Initialize a `LitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param torch_compile: Whether to compile the model with torch.compile.
        :param max_steps: Maximum number of training steps.
        :param log_every_n_steps: Log metrics every n steps.
        :param val_check_interval: Run validation every n steps.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function - to be implemented in subclasses
        self.criterion = None  # define your generative loss (e.g., reconstruction + KL) in subclasses

        # metric objects for calculating and averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        # step counters
        self.training_step_count = 0
        self.validation_step_count = 0
        self.test_step_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of input data.
        :return: A tensor of model outputs.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.training_step_count = 0

    def model_step(self, batch: Any) -> tuple[torch.Tensor, Any]:
        # Implement model forward pass and loss calculation here for generative training
        # Should return (loss, outputs). Override this in your concrete subclass.
        pass

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, outputs = self.model_step(batch)
        
        # update step counter
        self.training_step_count += 1

        # update and log metrics
        self.train_loss(loss)
        
        # log every n steps instead of every epoch
        if self.training_step_count % self.hparams.log_every_n_steps == 0:
            self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log("train/step", float(self.training_step_count), on_step=True, on_epoch=False)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data.
        :param batch_idx: The index of the current batch.
        """
        loss, outputs = self.model_step(batch)
        
        # update step counter
        self.validation_step_count += 1

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/step", float(self.validation_step_count), on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data.
        :param batch_idx: The index of the current batch.
        """
        loss, outputs = self.model_step(batch)
        
        # update step counter
        self.test_step_count += 1

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/step", float(self.test_step_count), on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.torch_compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_best",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Lightning hook that is called when saving a checkpoint."""
        # Save step counters
        checkpoint["training_step_count"] = self.training_step_count
        checkpoint["validation_step_count"] = self.validation_step_count
        checkpoint["test_step_count"] = self.test_step_count

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Lightning hook that is called when loading a checkpoint."""
        # Restore step counters
        self.training_step_count = checkpoint.get("training_step_count", 0)
        self.validation_step_count = checkpoint.get("validation_step_count", 0)
        self.test_step_count = checkpoint.get("test_step_count", 0)


if __name__ == "__main__":
    _ = LitModule(None, None, None, None)