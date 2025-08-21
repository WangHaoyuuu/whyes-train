"""
Generic LightningDataModule template (no auto-splitting).

What it provides
- A structured place to load/prepare train/val/test datasets in `setup(stage)`
- Ready-made DataLoaders using `batch_size`, `num_workers`, `pin_memory`
- World-size aware `batch_size_per_device` for DDP

Constructor parameters
- data_dir (str): base directory for your data (optional helper).
- batch_size (int): global batch size (will be divided by world_size when trainer is attached).
- num_workers (int): dataloader workers per process.
- pin_memory (bool): enable pinned memory for faster host-to-device transfer.
- train_data_path / val_data_path / test_data_path (str|None): optional dataset location hints.

How to plug in your datasets
- Implement loading in `setup(stage)`, e.g.:
  ```python
  if stage in ("fit", None):
      if self.data_train is None:
          self.data_train = YourTrainDataset(self.hparams.train_data_path)
      if self.data_val is None:
          self.data_val = YourValDataset(self.hparams.val_data_path)
  if stage in ("test", None):
      if self.data_test is None:
          self.data_test = YourTestDataset(self.hparams.test_data_path)
  ```

Hydra example (configs/data/datamodule.yaml)
```yaml
_target_: src.data.datamodule.DataModule
batch_size: 64
num_workers: 4
pin_memory: true
train_data_path: "/data/train"
val_data_path: "/data/val"
test_data_path: "/data/test"
```

CLI overrides
- Change batch size and paths:
  python -m src.scripts.train data.batch_size=128 data.train_data_path=/path/train data.val_data_path=/path/val
- Dataloader threads tweak:
  python -m src.scripts.train data.num_workers=8 data.pin_memory=true
"""
from typing import Any, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DataModule(LightningDataModule):
    """`LightningDataModule` for generative models.

    A generic data module that supports separate train, validation, and test datasets
    without automatic splitting. Designed for generative models that may work with
    various data types and formats.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_data_path: Optional[str] = None,
        val_data_path: Optional[str] = None,
        test_data_path: Optional[str] = None,
    ) -> None:
        """Initialize a `DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param train_data_path: Path to training dataset. Defaults to `None`.
        :param val_data_path: Path to validation dataset. Defaults to `None`.
        :param test_data_path: Path to test dataset. Defaults to `None`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # expose commonly accessed attributes directly
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # Implement data downloading/preparation logic here
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                error_msg = f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                raise RuntimeError(error_msg)
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load datasets based on stage
        if stage == "fit" or stage is None:
            if self.data_train is None:
                # Implement training dataset loading logic here
                pass
            if self.data_val is None:
                # Implement validation dataset loading logic here
                pass

        if stage == "test" or stage is None:
            if self.data_test is None:
                # Implement test dataset loading logic here
                pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.data_train is None:
            raise RuntimeError("Training dataset not loaded. Call setup() first.")
        
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.data_val is None:
            raise RuntimeError("Validation dataset not loaded. Call setup() first.")
        
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if self.data_test is None:
            raise RuntimeError("Test dataset not loaded. Call setup() first.")
        
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DataModule()