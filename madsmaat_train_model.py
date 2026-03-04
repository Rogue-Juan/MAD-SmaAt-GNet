"""Code for training MAD-SmaAt-GNet and ablation study models"""

import argparse
import torchsummary
import os
import sys
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from tqdm import tqdm
from mad_smaat_gnet.utils import madsmaat_dataloader
from mad_smaat_gnet.utils.get_model_from_str import get_model_from_str
from mad_smaat_gnet.utils.train_model import fit



if __name__ == "__main__":
    print("Parser:")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.dev = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )  # When available, CUDA is faster and preferred
    args.learning_rate = 0.001
    args.n_channels = 4  # Number of input rain images
    args.n_classes = 4  # Number of target rain images
    args.lr_patience = 5  # Hyperparameter; number of epochs without change in validation score before lowering learning rate
    args.two_stream_channels = 20  # Number of input images of other variables
    args.img_height = 115  # Height of input images
    args.img_width = 115  # Width of input images
    args.base_c = 16  # Hyperparameter; base number of channels for the tensors
    args.rain_kernelsPL = (
        2  # Hyperparameter; # Number of kernels/filters per double convolutional layer
    )
    args.rain_reduc_ratio = 16  # Hyperparameter; Reduction factor in CBAM
    args.var_kernelsPL = (
        2  # Hyperparameter; # Number of kernels/filters per double convolutional layer
    )
    args.var_reduc_ratio = 16  # Hyperparameter; Reduction factor in CBAM
    args.dec_kernelsPL = (
        2  # Hyperparameter; # Number of kernels/filters per double convolutional layer
    )
    args.evo_bilinear = (
        True  # Hyperparameter; use bilinear interpolation in up-operation
    )
    args.smaat_bilinear = (
        True  # Hyperparameter; use bilinear interpolation in up-operation
    )
    args.dataset_folder = "path/to/your/data/"
    args.dataset_fn = "data.h5"  # HDF5 file of dataset
    args.earlystopping = 15  # Hyperparameter; number of epochs without change in validation score before stopping
    args.save_every = 10  # Save a checkpoint file of the model after this many epochs
    args.batch_size = 16  # Number of samples per training batch
    args.epochs = 100  # Maximum number of epochs before stopping
    args.valid_size = 0.1  # Percentage of training batches used for validation
    args.model_name = "evo-net model"
    args.evo_net = False  # Whether to use pre-trained parameters for the Evo-Net
    if args.model_name.lower() in [
        "mad_smaat_gnet",
        "full model",
        "mad-smaat-gnet" "evo-net model",
        "madsmaat_evo",
    ]:
        args.evo_net = True

    args.only_rain_data = False  # Use only rain input images
    if args.model_name.lower() in [
        "smaat_unet",
        "smaat u-net",
        "evo-net model",
        "madsmaat_evo",
        "evo-net",
        "evonet",
    ]:
        args.only_rain_data = True

    print("   Arguments parsed")

    print("Loading datasets:")
    dataset_file = args.dataset_folder + args.dataset_fn
    # Load your dataset into data loaders here
    train_dl, valid_dl = madsmaat_dataloader.get_train_valid_loader(
        data_fn=dataset_file,
        batch_size=args.batch_size,
        random_seed=42,
        num_input_images=args.n_channels,
        num_output_images=args.n_classes,
        valid_size=args.valid_size,
        shuffle=True,
        num_workers=1,  # Working with cuda
        pin_memory=True,  # Working with cuda
    )

    print("    Datasets loaded")

    # Load Model
    model = get_model_from_str(args)

    if args.evo_net:
        path_chkpt = f"path/to/your/checkpoints/best_MSELoss_EvoNet.pt"  #### CHANGED # Checkpoint file for model
        model_params = torch.load(path_chkpt, weights_only=False)  # weights_only=True
        state_dict = model_params["state_dict"]
        state_dict = {
            key[8:]: value for key, value in state_dict.items() if "evo_net" in key
        }

        # # Freeze evolution network's parameters (pre-trained)
        # model.evo_net.load_state_dict(state_dict)
        # for param in model.evo_net.parameters():
        #     param.requires_grad = False

    # Move model to device
    model.to(args.dev)

    summary_in = (
        [(args.n_channels, args.img_height, args.img_width)]
        if args.only_rain_data
        else [
            (args.n_channels, args.img_height, args.img_width),
            (args.two_stream_channels, args.img_height, args.img_width),
        ]
    )
    torchsummary.summary(
        model, summary_in, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"Channnels: {args.n_channels}; classes (number of output images): {args.n_classes}"
    )

    # Define Optimizer and loss
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_func = nn.MSELoss()  # reduction="sum"

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=args.lr_patience
    )  # LR scheduler will decrease learning rate if MSE loss does not decrease

    print(f"Start training: {model.__class__.__name__}")
    # Train network
    fit(
        epochs=args.epochs,
        model=model,
        loss_func=loss_func,
        opt=opt,
        train_dl=train_dl,
        valid_dl=valid_dl,
        dev=args.dev,
        save_every=args.save_every,
        tensorboard=True,
        earlystopping=args.earlystopping,
        lr_scheduler=lr_scheduler,
        only_rain_data=args.only_rain_data,
        input_len=args.n_channels,
        output_len=args.n_classes,
    )
    print("Process complete")
