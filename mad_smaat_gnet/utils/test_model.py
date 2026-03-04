"""Code for testing MAD-SmaAt-GNet and ablation study models"""

# Code was based on and adapted from the training script from: https://github.com/HansBambel/SmaAt-UNet
from mad_smaat_gnet.utils.binary_metrics import *
import os
import sys
import json
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import time
from tqdm import tqdm
from mad_smaat_gnet.utils import madsmaat_dataloader
from mad_smaat_gnet.utils.plot_preds import plot_preds

def test_model(
    model,
    dev,
    test_dl,
    only_rain_data: bool,
    save_folder: str = "",
    save_fn: str = "",
    input_len: int = 4,
    output_len: int = 4,
):

    example_ytrue = None
    example_ypred = None

    rain_norm = 91.55  # Normalise rain input to [0,1]; max value in training set
    temp_norm = 307.72  # Normalisation with max value in training set
    press_norm = 104281.06  # Idem
    Uwind_norm = 39.58
    Vwind_norm = 33.52
    seq_len = input_len + output_len

    # Binary metrics
    firstBinary = True
    thresholds = [0.5, 4, 8, 16]
    binObins = []

    # Calculate total loss
    total_samples = 0
    total_loss = 0.0
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
        "model": model.__class__.__name__,
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
        f"---------------------------------------------\n{model.__class__.__name__}\n---------------------------------------------"
    )
    print(f"MSE loss: {results['mse']:.4f}")
    print(f"Mean accuracy: {sum(acc_list)/len(acc_list):.4f}")
    print(f"Mean precision: {sum(prec_list)/len(prec_list):.4f}")
    print(f"Mean recall: {sum(rec_list)/len(rec_list):.4f}")
    print(f"Mean F1-score: {sum(f1_list)/len(f1_list):.4f}")
    print(f"Mean CSI: {sum(csi_list)/len(csi_list):.4f}")
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
        model_name=model.__class__.__name__,
        num_imgs=output_len,
    )

    if save_folder != "":
        file_name = save_folder + save_fn
        exists = os.path.exists(file_name)
        if exists:
            print("File already exists")
            new_fn = f"test_results_{model.__class__.__name__}_"
            add2fn = input(
                "Please give an additional description to the file name or 'overwrite/replace': "
            )
            new_fn = (
                file_name
                if add2fn in ["overwrite", "replace"]
                else save_folder + new_fn + add2fn + ".json"
            )
            if add2fn not in ["overwrite", "replace"]:
                results["model"] = results["model"] + "_" + add2fn
            with open(new_fn, "w") as f:
                json.dump(results, f, indent=4)
            print("Results saved")
        else:
            print("Saving results to JSON...")
            with open(file_name, "w") as f:
                json.dump(results, f, indent=4)
            print("Results saved")

    print("Testing complete")

    return True