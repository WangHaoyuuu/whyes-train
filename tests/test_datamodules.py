from pathlib import Path

import pytest
import torch

from src.data.datamodule import DataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_datamodule(batch_size: int) -> None:
    """Tests `DataModule` to verify that it can be instantiated correctly
    and that the necessary attributes and methods exist.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        train_data_path=None,
        val_data_path=None,
        test_data_path=None,
        num_workers=0,
        pin_memory=False
    )
    
    # Test that the datamodule can be instantiated
    assert dm is not None
    assert dm.batch_size == batch_size
    assert dm.data_dir == data_dir
    
    # Test that required methods exist
    assert hasattr(dm, 'prepare_data')
    assert hasattr(dm, 'setup')
    assert hasattr(dm, 'train_dataloader')
    assert hasattr(dm, 'val_dataloader')
    assert hasattr(dm, 'test_dataloader')
    assert hasattr(dm, 'teardown')
    assert hasattr(dm, 'state_dict')
    assert hasattr(dm, 'load_state_dict')
    
    # Test that methods can be called without errors
    dm.prepare_data()
    dm.setup('fit')
    dm.teardown('fit')
    
    # Test state dict functionality
    state = dm.state_dict()
    assert isinstance(state, dict)
    dm.load_state_dict(state)
