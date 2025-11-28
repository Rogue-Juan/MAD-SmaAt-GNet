"""Code for training MAD-SmaAt-GNet and ablation study models"""

# Code was based on and adapted from the training script from: https://github.com/HansBambel/SmaAt-UNet
import argparse
from mad_smaat_gnet.models.MAD_SmaAt_GNet import MAD_SmaAt_GNet
from mad_smaat_gnet.models.madsmaat_2stream import madsmaat_2stream
from mad_smaat_gnet.models.madsmaat_evo import madsmaat_evo
from mad_smaat_gnet.models.madsmaat_components import EvoNet
from smaat.SmaAt_UNet import SmaAt_UNet
import torchsummary
import os
import sys

from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import time
from tqdm import tqdm
from mad_smaat_gnet.utils import madsmaat_data


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_model_from_str(args):
    if args.model_name.lower() in ["mad_smaat_gnet", "full model", "mad-smaat-gnet"]:
        return MAD_SmaAt_GNet(hparams=args)
    elif args.model_name.lower() in ["2-stream model", "madsmaat_2stream"]:
        return madsmaat_2stream(hparams=args)
    elif args.model_name.lower() in ["evo-net model", "madsmaat_evo"]:
        return madsmaat_evo(hparams=args)
    elif args.model_name.lower() in ["smaat_unet", "smaat u-net"]:
        return SmaAt_UNet(
            n_channels=args.n_channels,
            n_classes=args.n_classes,
            kernels_per_layer=args.rain_kernelsPL,
            bilinear=args.smaat_bilinear,
            reduction_ratio=args.rain_reduc_ratio,
            base_c=args.base_c,
        )
    elif args.model_name.lower() in ["evo-net", "evonet"]:
        return EvoNet(
            n_channels=args.n_channels,
            n_classes=args.n_classes,
            base_c=args.base_c,
            evo_bilinear=args.evo_bilinear,
            height_rain=args.img_height,
            width_rain=args.img_width,
        )
    else:
        assert True == False, f"Unknown model, got: {args.model_name}"


def fit(
    epochs: int,
    model,
    loss_func,
    opt,
    train_dl,
    valid_dl,
    dev=None,
    save_every: Optional[int] = None,
    tensorboard: bool = False,
    earlystopping=None,
    lr_scheduler=None,
    only_rain_data: bool = False,
    input_len: int = 3,
    output_len: int = 4,
):
    writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(comment=f"{model.__class__.__name__}")

    if dev is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = os.getcwd()
    default_save_path = root + "path/to/your/checkpoints"

    seq_len = input_len + output_len
    rain_norm = 91.55  # Normalise rain input to [0,1]; max value in training set
    temp_norm = 307.72  # Normalisation with max value in training set
    press_norm = 104281.06  # Idem
    Uwind_norm = 39.58
    Vwind_norm = 33.52

    start_time = time.time()
    best_score = sys.float_info.max
    earlystopping_counter = 0
    # print_shapes = True
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        model.train()
        train_loss = 0.0
        for _, (xtuple, yb) in enumerate(tqdm(train_dl, desc="Batches", leave=False)):
            if only_rain_data:
                xb, _ = xtuple
                xb = xb.float().to(dev) / rain_norm
                y_pred = model(xb)
            else:
                xb, zb = xtuple
                xb = xb.float().to(dev) / rain_norm
                zb = zb.float().to(dev)
                temp = zb[:, 0:input_len, :, :] / temp_norm
                press = zb[:, input_len : input_len + input_len, :, :] / press_norm
                humid = zb[:, input_len * 2 : input_len * 2 + input_len, :, :]
                Uwind = (
                    zb[:, input_len * 3 : input_len * 3 + input_len, :, :] / Uwind_norm
                )
                Vwind = (
                    zb[:, input_len * 4 : input_len * 4 + input_len, :, :] / Vwind_norm
                )
                zb = torch.cat((temp, press, humid, Uwind, Vwind), dim=1)
                y_pred = model(xb, zb)
            y_pred = y_pred * rain_norm  # Denormalise output to mm/h
            loss = loss_func(y_pred.float(), yb.float().to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Calc validation loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for xtuple, yb in tqdm(valid_dl, desc="Validation", leave=False):
                if only_rain_data:
                    xb, _ = xtuple
                    xb = xb.float().to(dev) / rain_norm
                    y_pred = model(xb)
                else:
                    xb, zb = xtuple
                    xb = xb.float().to(dev) / rain_norm
                    zb = zb.float().to(dev)
                    temp = zb[:, 0:input_len, :, :] / temp_norm
                    press = zb[:, input_len : input_len + input_len, :, :] / press_norm
                    humid = zb[:, input_len * 2 : input_len * 2 + input_len, :, :]
                    Uwind = (
                        zb[:, input_len * 3 : input_len * 3 + input_len, :, :]
                        / Uwind_norm
                    )
                    Vwind = (
                        zb[:, input_len * 4 : input_len * 4 + input_len, :, :]
                        / Vwind_norm
                    )
                    zb = torch.cat((temp, press, humid, Uwind, Vwind), dim=1)
                    y_pred = model(xb, zb)
                y_pred = y_pred * rain_norm  # Convert values back to mm/hour
                loss = loss_func(y_pred.float(), yb.float().to(dev))
                val_loss += loss.item()
            val_loss /= len(valid_dl)

        # Save the model with the best validation loss so far;
        # if it does not decrease, model will early stop after specified number of epochs
        if val_loss < best_score:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "model": model,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                },
                default_save_path + f"best_MSELoss_{model.__class__.__name__}.pt",
            )
            best_score = val_loss
            earlystopping_counter = 0

        else:
            earlystopping_counter += 1
            if earlystopping is not None and earlystopping_counter >= earlystopping:
                print(
                    f"\nStopping early --> mean validation loss has not decreased over {earlystopping} epochs"
                )
                break

        print(
            f"\nEpoch: {epoch:5d}, Time: {(time.time() - start_time) / 60:.3f} min,"
            f" Train_loss: {train_loss:.5f}, Val_loss: {val_loss:.5f},",
            f"lr: {get_lr(opt)},",
            (
                f"Early stopping counter: {earlystopping_counter}/{earlystopping}"
                if earlystopping is not None
                else ""
            ),
        )

        if writer:
            # add to tensorboard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Parameters/learning_rate", get_lr(opt), epoch)
        if save_every is not None and epoch % save_every == 0:
            # save model
            torch.save(
                {
                    "model": model,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                },
                default_save_path + f"{model.__class__.__name__}_epoch{epoch}.pt",
            )
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)


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
    args.dataset_fn = "path/to/your/data.h5"  # HDF5 file of dataset
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
    # Load your dataset into data loaders here
    train_dl, valid_dl = madsmaat_data.get_train_valid_loader(
        data_fn=args.dataset_fn,
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
