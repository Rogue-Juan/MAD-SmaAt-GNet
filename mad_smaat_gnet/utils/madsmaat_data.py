from torch.utils.data import Dataset
import h5py
import torch
import numpy as np

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
