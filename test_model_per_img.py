"""Code for obtaining metrics of NowcastSmaAt and ablation study models per image"""

# Code was based on and adapted from the training script from: https://github.com/HansBambel/SmaAt-UNet
import argparse
from mad_smaat_gnet.models.MAD_SmaAt_GNet import MAD_SmaAt_GNet
from mad_smaat_gnet.models.madsmaat_2stream import madsmaat_2stream
from mad_smaat_gnet.models.madsmaat_evo import madsmaat_evo
from mad_smaat_gnet.models.madsmaat_components import EvoNet
from smaat.SmaAt_UNet import SmaAt_UNet
from mad_smaat_gnet.utils.binary_metrics import *
import os
import sys
import json
import matplotlib.pyplot as plt
import torchsummary

from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import time
from tqdm import tqdm
import argparse
from mad_smaat_gnet.utils import madsmaat_data


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


def plot_preds(example_ytrue, example_ypred, model_name: str, num_imgs: int = 4):
    layout_var = "none" if num_imgs == 1 else "constrained"
    fig, ax = plt.subplots(ncols=2, nrows=num_imgs, layout=layout_var)
    # plt.suptitle("Persistence model", fontsize=14)

    max_val = example_ytrue.max()
    min_val = example_ytrue.min()

    if num_imgs == 1:
        ax = [ax]

    row_num = 0
    col_num = 0
    for row in ax:
        for col in row:
            if col_num % 2 == 0:
                if row_num == 0:
                    col.set_title(model_name)
                im = col.imshow(
                    example_ypred[0, row_num],
                    cmap="viridis",
                    vmin=min_val,
                    vmax=max_val,
                )
                col.set_yticks([])
                col.set_xticks([])
                col.set_ylabel(f"t={row_num}")
            else:
                if row_num == 0:
                    col.set_title("Ground truth")
                im = col.imshow(
                    example_ytrue[0, row_num],
                    cmap="viridis",
                    vmin=min_val,
                    vmax=max_val,
                )
                col.set_yticks([])
                col.set_xticks([])
                col.set_ylabel(f"t={row_num}")
            col_num += 1
        row_num += 1

    # fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
    fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="[mm/h]")
    plt.show()
    return True


def test_model(
    model,
    dev,
    test_dl,
    only_rain_data: bool,
    save_fn: str = "",
    input_len: int = 4,
    output_len: int = 4,
):

    example_ytrue = None
    example_ypred = None

    norm_factor = 91.55  # Normalise input to [0,1]
    seq_len = input_len + output_len

    # Binary metrics
    firstPred = True
    thresholds = [0.5, 4, 8, 16]
    bin_mets_per_img = []

    # Calculate total loss
    total_samples = 0
    total_loss_per_img = []
    img_shape = None
    checks = True
    print_shapes = False
    model.eval()
    with torch.no_grad():
        count = -1
        for xtuple, yb in tqdm(test_dl, desc="Test set", leave=True):
            count += 1
            if only_rain_data:
                xb, _ = xtuple
                xb = xb.float().to(dev) / norm_factor
                y_pred = model(xb)
            else:
                xb, zb = xtuple
                xb = xb.float().to(dev) / norm_factor
                zb = zb.float().to(dev)
                temp = (
                    zb[:, 0:input_len, :, :] / 307.72
                )  # Normalisation with max value in training set
                press = zb[:, seq_len : seq_len + input_len, :, :] / 104281.06  # Idem
                humid = zb[:, seq_len * 2 : seq_len * 2 + input_len, :, :]
                Uwind = zb[:, seq_len * 3 : seq_len * 3 + input_len, :, :] / 39.58
                Vwind = zb[:, seq_len * 4 : seq_len * 4 + input_len, :, :] / 33.52
                zb = torch.cat((temp, press, humid, Uwind, Vwind), dim=1)
                y_pred = model(xb, zb)
            y_pred = y_pred * norm_factor  # Convert values back to mm/hour

            y_pred = y_pred.to(dev)
            yb = yb.float().to(dev)

            if count == 120:  # 22 # 40, 120 01mm
                example_ytrue = yb.cpu().numpy()
                example_ypred = y_pred.cpu().numpy()
                # break

            if checks:
                assert (
                    y_pred.shape == yb.shape
                ), f"Wrong shapes; y_pred: {y_pred.shape}, yb: {yb.shape}"
                img_shape = (y_pred.shape[2], y_pred.shape[3])
                batch_size = y_pred.shape[0]
                checks = False
            if print_shapes:
                print(f"\nShape xb: {xb.shape}")
                print(f"xb max: {xb.max()}")
                if not only_rain_data:
                    print(f"Shape zb: {zb.shape}")
                    print(f"Temp max: {zb[:, 0:input_len, :, :].max()}")
                    print(f"Press max: {zb[:, input_len:2*input_len, :, :].max()}")
                    print(f"Humid max: {zb[:, 2*input_len:3*input_len, :, :].max()}")
                    print(
                        f"Uwind max: {(torch.abs(zb[:, 3*input_len:4*input_len, :, :])).max()}"
                    )
                    print(
                        f"Vwind max: {(torch.abs(zb[:, 4*input_len:5*input_len, :, :])).max()}"
                    )
                print(f"Shape y_pred: {y_pred.shape}")
                print(f"Shape yb: {yb.shape}")
                print(f"Shape img: {img_shape}")
                print_shapes = False

            for i_img in range(output_len):
                if firstPred:
                    for _ in range(output_len):
                        total_loss_per_img.append(0.0)
                        bin_mets_per_img.append([0, 0, 0, 0])
                    firstPred = False

                y_pred_i = y_pred[:, i_img, :, :]
                yb_i = yb[:, i_img, :, :]

                loss = (
                    torch.nn.functional.mse_loss(
                        y_pred_i,
                        yb_i,
                        reduction="sum",
                    )
                ).to(dev)
                total_loss_per_img[i_img] += loss.item()

                for i_thresh in range(len(thresholds)):
                    true_pos, true_negs, false_pos, false_negs = calc_bin_classes(
                        y_pred=y_pred_i, yb=yb_i, thresh=thresholds[i_thresh]
                    )
                    bin_mets_per_img[i_img][0] = bin_mets_per_img[i_img][0] + true_pos
                    bin_mets_per_img[i_img][1] = bin_mets_per_img[i_img][1] + true_negs
                    bin_mets_per_img[i_img][2] = bin_mets_per_img[i_img][2] + false_pos
                    bin_mets_per_img[i_img][3] = bin_mets_per_img[i_img][3] + false_negs
            total_samples += 1

    num_pixels = img_shape[0] * img_shape[1]
    for k in range(output_len):
        total_loss_per_img[k] /= len(test_dl)
        total_loss_per_img[k] /= num_pixels

    acc_list, prec_list, rec_list, f1_list, csi_list, mcc_list = (
        calc_all_metrics_from_bins(binObins=bin_mets_per_img)
    )

    print(f"Prediction length: {output_len}")
    results = {
        "model": model.__class__.__name__,
        "num_samples": total_samples,
        "batch_size": batch_size,
        "img_shape": img_shape,
        "pred_len": output_len,
        "mse": total_loss_per_img,
        "thresholds": thresholds,
        "acc_per_img": acc_list,
        "prec_per_img": prec_list,
        "rec_per_img": rec_list,
        "f1_per_img": f1_list,
        "csi_per_img": csi_list,
        "mcc_per_img": mcc_list,
    }
    print(
        f"---------------------------------------------\n{model.__class__.__name__}\n---------------------------------------------"
    )
    print(f"MSE loss per img: {[float(f'{el:.4f}') for el in results['mse']]}")
    print(f"Mean accuracy per img: {[float(f'{el:.4f}') for el in acc_list]}")
    print(f"Mean precision per img: {[float(f'{el:.4f}') for el in prec_list]}")
    print(f"Mean recall per img: {[float(f'{el:.4f}') for el in rec_list]}")
    print(f"Mean F1-score per img: {[float(f'{el:.4f}') for el in f1_list]}")
    print(f"Mean CSI per img: {[float(f'{el:.4f}') for el in csi_list]}")
    print(f"Mean MCC per img: {[float(f'{el:.4f}') for el in mcc_list]}")
    print(f"Thresholds: {thresholds} mm/h")

    plot_preds(
        example_ytrue,
        example_ypred,
        model_name=model.__class__.__name__,
        num_imgs=output_len,
    )

    if save_fn != "":
        exists = os.path.exists(save_fn)
        if exists:
            print("File already exists")
            new_fn = (
                f"path/to/your/results/test_results_per_img_{model.__class__.__name__}_"
            )
            add2fn = input(
                "Please give an additional description to the file name or 'overwrite/replace': "
            )
            new_fn = (
                save_fn
                if add2fn in ["overwrite", "replace"]
                else new_fn + add2fn + ".json"
            )
            if add2fn not in ["overwrite", "replace"]:
                results["model"] = results["model"] + "_" + add2fn
            with open(new_fn, "w") as f:
                json.dump(results, f, indent=4)
            print("Results saved")
        else:
            print("Saving results to JSON...")
            with open(save_fn, "w") as f:
                json.dump(results, f, indent=4)
            print("Results saved")

    print("Testing complete")

    return True


def main():
    fn_test_set = "path/to/your/data.h5"  # HDF5 file with test set
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.n_channels = 4  # Number of input rain images
    args.n_classes = 4  # Number of target rain images
    args.harmonie_n_channels = 20  # Number of input images of other variables
    args.img_height = 115  # Height of input images
    args.img_width = 115  # Width of input images
    args.evo_bilinear = True  # Bilinear interpolation is used or not
    args.smaat_bilinear = True  # Bilinear interpolation is used or not
    args.base_c = 16  # Base number of channels (32)
    args.rain_kernelsPL = 2  # kernels per layer
    args.rain_reduc_ratio = 16  # Reduction factor in CBAM
    args.harmo_kernelsPL = 2  # Number of kernels/filters per double convolutional layer
    args.harmo_reduc_ratio = 16  # Reduction factor in CBAM
    args.dec_kernelsPL = 2  # Number of kernels/filters per double convolutional layer
    args.model_name = "mad-smaat-gnet"
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

    print("Loading model:")
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # load model weights and biases
    model = get_model_from_str(args)
    path_chkpt = f"path/to/your/checkpoints/best_MSELoss_{model.__class__.__name__}.pt"  # Checkpoint file for model
    model_params = torch.load(path_chkpt, weights_only=False)

    model.load_state_dict(model_params["state_dict"])
    model.to(dev)
    model.eval()

    save_fn = f"path/to/your/results/test_results_per_img_{model.__class__.__name__}.json"  # Path and file name to store the results in

    ######## Summary of parameters and model size
    # summary_in = (
    #     [(args.n_channels, args.heigth_rain, args.img_width)]
    #     if args.only_rain_data
    #     else [
    #         (args.n_channels, args.heigth_rain, args.img_width),
    #         (args.two_stream_channels, args.heigth_rain, args.img_width),
    #     ]
    # )
    # torchsummary.summary(
    #     model, summary_in, device="cuda" if torch.cuda.is_available() else "cpu"
    # )

    print(f"    Model loaded: {model.__class__.__name__}")
    print(f"    Epoch number: {model_params['epoch']}")

    print("Loading test set:")
    # load test set using nowcastsmaat utils
    test_dl = madsmaat_data.get_test_loader(
        data_fn=fn_test_set,
        batch_size=1,  # USE CONSISTENT BATCH SIZE WHEN COMPARING MODELS
        num_input_images=args.n_channels,
        num_output_images=args.n_classes,
        shuffle=False,
        num_workers=1,  # Working with cuda
        pin_memory=True,  # Working with cuda
    )

    print("    Test set loaded")

    print("Running tests per img...")

    test_model(
        model=model,
        dev=dev,
        test_dl=test_dl,
        only_rain_data=args.only_rain_data,
        save_fn=save_fn,
        input_len=args.n_channels,
        output_len=args.n_classes,
    )

    print("Process complete")

    return True


if __name__ == "__main__":
    main()
