from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


# Taken from: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
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


# Data class for MAD-SmaAt-GNet data
class madsmaat_h5(Dataset):
    def __init__(
        self,
        in_file: str,
        num_input_images: int,
        num_output_images: int,
        train: bool = True,
        transform=None,
    ):
        super().__init__()

        self.file_name = in_file
        self.n_samples = h5py.File(self.file_name, "r")["train" if train else "test"][
            "harmo_rain_imgs"
        ].shape[0]
        self.sequence_length = h5py.File(self.file_name, "r")[
            "train" if train else "test"
        ]["harmo_rain_imgs"].shape[1]

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.prediction_length = num_input_images + num_output_images

        self.train = train
        # Dataset is all the samples
        self.size_dataset = self.n_samples
        self.transform = transform
        self.dataset = None

    def __getitem__(self, sample_idx):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, "r", rdcc_nbytes=1024**3)[
                "train" if self.train else "test"
            ]
        rain_imgs = np.array(self.dataset["harmo_rain_imgs"][sample_idx])
        harmo_imgs = np.array(self.dataset["harmonie_images"][sample_idx])

        temp = harmo_imgs[0 : self.num_input]
        press = harmo_imgs[self.sequence_length : self.sequence_length + self.num_input]
        humid = harmo_imgs[
            self.sequence_length * 2 : self.sequence_length * 2 + self.num_input,
        ]
        Uwind = harmo_imgs[
            self.sequence_length * 3 : self.sequence_length * 3 + self.num_input,
        ]
        Vwind = harmo_imgs[
            self.sequence_length * 4 : self.sequence_length * 4 + self.num_input,
        ]
        harmo_in = np.concatenate((temp, press, humid, Uwind, Vwind), axis=0)

        # apply transformations
        if self.transform is not None:
            rain_imgs = self.transform(rain_imgs)
            harmo_in = self.transform(harmo_in)
        rain_in = rain_imgs[: self.num_input]  # first 4 images in the paper

        target_imgs = rain_imgs[
            self.num_input : self.prediction_length
        ]  # 4 images ahead in the paper

        return (rain_in, harmo_in), target_imgs

    def __len__(self):
        return self.size_dataset


if __name__ == "__main__":
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
