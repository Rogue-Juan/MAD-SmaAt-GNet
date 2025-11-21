"""Code for obtaining persistence model metrics"""

# Code was based on and adapted from the training script from: https://github.com/HansBambel/SmaAt-UNet
import os
import sys
import json

import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from mad_smaat_gnet.utils import madsmaat_data
import numpy as np
from mad_smaat_gnet.utils.binary_metrics import *


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
                col.set_ylabel(f"t={row_num+1}")
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
                col.set_ylabel(f"t={row_num+1}")
            col_num += 1
        row_num += 1

    # fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes((0.85, 0.15, 0.05, 0.7))
    fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="[mm/h]")
    plt.show()
    return True


def calc_persistence_model(
    dev,
    test_dl,
    save_fn: str = "",
    input_len: int = 4,
    output_len: int = 4,
    norm_factor: float = 1,
):
    # Calculate total loss
    example_ytrue = None
    example_ypred = None
    total_samples = 0
    total_loss = 0.0
    firstBinary = True
    thresholds = [0.5, 4, 8, 16]
    binObins = []
    img_shape = None
    checks = True
    print_shapes = False
    count = -1
    for xtuple, yb in tqdm(test_dl, desc="Test set", leave=False):
        count += 1

        xb, _ = xtuple
        n_channels = xb.shape[1]
        n_classes = yb.shape[1]
        y_pred = xb[:, n_channels - 1 : n_channels, :, :].repeat((1, output_len, 1, 1))

        y_pred = y_pred.float().to(dev)
        yb = yb.float().to(dev)

        y_pred = y_pred / norm_factor
        yb = yb / norm_factor

        if count == 120:
            example_ytrue = yb.cpu().numpy()
            example_ypred = y_pred.cpu().numpy()
            break

        if checks:
            assert (
                y_pred.shape == yb.shape
            ), f"Wrong shapes; y_pred: {y_pred.shape}, yb: {yb.shape}"
            img_shape = (y_pred.shape[2], y_pred.shape[3])
            batch_size = y_pred.shape[0]
            checks = False
        if print_shapes:
            print(f"n_channels: {n_channels}")
            print(f"n_classes: {n_classes}")
            # print(xb[0, -1, :, :])
            # print(y_pred[0, 0, :, :])
            print(f"Shape xb: {xb.shape}")
            print(f"Shape y_pred: {y_pred.shape}")
            # print(y_pred)
            print(f"Shape yb: {yb.shape}")
            print(f"Shape img: {img_shape}")
            print_shapes = False

        loss = (torch.nn.functional.mse_loss(y_pred, yb, reduction="sum")).to(dev)
        total_loss += loss.item()
        total_samples += 1

        for i in range(len(thresholds)):
            true_pos, true_negs, false_pos, false_negs = calc_bin_classes(
                y_pred=y_pred, yb=yb, thresh=thresholds[i]
            )
            if firstBinary:
                if i == len(thresholds) - 1:
                    firstBinary = False
                binObins.append([true_pos, true_negs, false_pos, false_negs])
            else:
                binObins[i][0] = binObins[i][0] + true_pos
                binObins[i][1] = binObins[i][1] + true_negs
                binObins[i][2] = binObins[i][2] + false_pos
                binObins[i][3] = binObins[i][3] + false_negs

    num_pixels = img_shape[0] * img_shape[1]
    total_loss /= len(test_dl)
    total_loss /= num_pixels
    total_loss /= output_len

    acc_list, prec_list, rec_list, f1_list, csi_list, mcc_list = (
        calc_all_metrics_from_bins(binObins=binObins)
    )

    print(f"Prediction length: {output_len}")
    results = {
        "model": "Persistence model",
        "num_samples": total_samples,
        "batch_size": batch_size,
        "img_shape": img_shape,
        "pred_len": output_len,
        "mse": total_loss,
        "thresholds": thresholds,
        "acc": acc_list,
        "prec": prec_list,
        "rec": rec_list,
        "f1": f1_list,
        "csi": csi_list,
        "mcc": mcc_list,
    }
    print(
        f"---------------------------------------------\nPersistence model\n---------------------------------------------"
    )
    print(f"MSE loss: {results['mse']:.4f}")
    # print(f"MSE loss over total sequence: {results['mse']*seq_len:.4f}")
    print(f"Mean accuracy: {sum(acc_list)/len(acc_list):.4f}")
    print(f"Mean precision: {sum(prec_list)/len(prec_list):.4f}")
    print(f"Mean recall: {sum(rec_list)/len(rec_list):.4f}")
    print(f"Mean F1-score: {sum(f1_list)/len(f1_list):.4f}")
    print(f"Mean MCC: {sum(mcc_list)/len(mcc_list):.4f}")
    print(f"Thresholds: {thresholds} mm/h")
    print(f"Accuracy: {[float(f'{el:.4f}') for el in acc_list]}")
    print(f"Precision: {[float(f'{el:.4f}') for el in prec_list]}")
    print(f"Recall: {[float(f'{el:.4f}') for el in rec_list]}")
    print(f"F1-scores: {[float(f'{el:.4f}') for el in f1_list]}")
    print(f"CSI: {[float(f'{el:.4f}') for el in csi_list]}")
    print(f"MCC: {[float(f'{el:.4f}') for el in mcc_list]}")

    plot_preds(
        example_ytrue,
        example_ypred,
        model_name="persistence model",
        num_imgs=output_len,
    )

    if save_fn != "":
        exists = os.path.exists(save_fn)
        if exists:
            print("File already exists")
            new_fn = f"path/to/your/results/test_results_persistence_model_"
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
    print("Setting up...")
    fn_test_set = "path/to/your/data.h5"  # HDF5 dataset file with test set
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    save_fn = "path/to/your/results/test_results_persistence_model.json"  # Path and file name to store the results in
    n_channels = 4  # Number of input rain images
    n_classes = 4  # Number of target rain images

    print("Loading test set:")
    # load test set using nowcastsmaat utils
    test_dl = madsmaat_data.get_test_loader(
        data_fn=fn_test_set,
        batch_size=1,  # USE CONSISTENT BATCH SIZE WHEN COMPARING MODELS
        num_input_images=n_channels,
        num_output_images=n_classes,
        shuffle=False,
        num_workers=1,  # Working with cuda
        pin_memory=True,  # Working with cuda
    )

    print("   Test set loaded")

    print("Calculating metrics persistence model...")

    calc_persistence_model(
        dev=dev,
        test_dl=test_dl,
        save_fn=save_fn,
        input_len=n_channels,
        output_len=n_classes,
    )

    print("Process complete")

    return True


if __name__ == "__main__":
    main()
