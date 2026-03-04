"""Code for training MAD-SmaAt-GNet and ablation study models"""

# Code was based on and adapted from the training script from: https://github.com/HansBambel/SmaAt-UNet
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
from typing import Optional
import time

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

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
    default_save_path = root + "/checkpoints/"

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
                press = zb[:, input_len : input_len * 2, :, :] / press_norm
                humid = zb[:, input_len * 2 : input_len * 3, :, :]
                Uwind = zb[:, input_len * 3 : input_len * 4, :, :] / Uwind_norm
                Vwind = zb[:, input_len * 4 : input_len * 5, :, :] / Vwind_norm
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
                    press = zb[:, input_len : input_len * 2, :, :] / press_norm
                    humid = zb[:, input_len * 2 : input_len * 3, :, :]
                    Uwind = zb[:, input_len * 3 : input_len * 4, :, :] / Uwind_norm
                    Vwind = zb[:, input_len * 4 : input_len * 5, :, :] / Vwind_norm
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
    print("Model is trained")
    return True