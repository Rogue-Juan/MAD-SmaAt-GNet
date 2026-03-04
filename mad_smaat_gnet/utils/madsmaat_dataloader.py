"""Training, testing, and validation data loaders for the MAD-SmaAt data class"""

from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from mad_smaat_gnet.utils.madsmaat_data import madsmaat_h5

# Taken from: https://github.com/HansBambel/SmaAt-UNet/tree/snapshot-paper/utils
def get_train_valid_loader(
    data_fn: str,
    batch_size: int,
    random_seed: int,
    num_input_images: int,
    num_output_images: int,
    valid_size: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 1,
    pin_memory: bool = False,
    transform=None,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MAD-SmaAt-GNet dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Parameters:
    ------
    - data_fn: file name of the dataset.
    - batch_size: how many samples per batch to load.
    - transform: whether to apply a transformation on the data which
      is performed in the data class.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns:
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # load the datasets
    train_dataset = madsmaat_h5(
        in_file=data_fn,
        num_input_images=num_input_images,
        num_output_images=num_output_images,
        train=True,
        transform=transform,
    )

    valid_dataset = madsmaat_h5(
        in_file=data_fn,
        num_input_images=num_input_images,
        num_output_images=num_output_images,
        train=True,
        transform=transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idxs, valid_idxs = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idxs)
    valid_sampler = SubsetRandomSampler(valid_idxs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader


def get_test_loader(
    data_fn: str,
    batch_size: int,
    num_input_images: int,
    num_output_images: int,
    shuffle: bool = False,
    shuffle_input: bool = False,
    num_workers: int = 1,
    pin_memory: bool = False,
    transform=None,
):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MAD-SmaAt-GNet dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Parameters:
    ------
    - data_fn: file name of the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns:
    -------
    - data_loader: test set iterator.
    """
    dataset = madsmaat_h5(
        in_file=data_fn,
        num_input_images=num_input_images,
        num_output_images=num_output_images,
        train=False,
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader

# Sanity check
if __name__ == "__main__":
    # Change 'from mad_smaat_gnet.utils.madsmaat_data import madsmaat_h5' to 'from madsmaat_data import madsmaat_h5'
    # to run this script
    print("Opening files...")
    filename = "path/to/your/data.h5"
    train_dl, valid_dl = get_train_valid_loader(
        data_fn=filename,
        random_seed=42,
        batch_size=2,
        num_input_images=4,
        num_output_images=4,
        valid_size=0.1,
        shuffle=True,
        num_workers=1,  # Working with cuda
        pin_memory=True,  # Working with cuda
    )

    print(f"Length train_dl: {len(train_dl)}")
    print(f"Length valid_dl: {len(valid_dl)}")

    for batch, ((x_in, z), yb) in enumerate(train_dl):
        print(f"Batch {batch}:")
        print(f"    Shape x_in: {x_in.shape}")
        print(f"    Shape yb: {yb.shape}")
        print(f"    Shape z: {z.shape}")
        if batch >= 7:
            break
