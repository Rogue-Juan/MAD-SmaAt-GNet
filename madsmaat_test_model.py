"""Code for testing MAD-SmaAt-GNet and ablation study models"""

import torchsummary
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from tqdm import tqdm
import argparse
from mad_smaat_gnet.utils import madsmaat_data
from mad_smaat_gnet.utils.test_model import test_model
from mad_smaat_gnet.utils.get_model_from_str import get_model_from_str



def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.folder_test_set = "path/to/your/data/"
    args.fn_test_set = "data.h5"  # HDF5 file with test set
    args.n_channels = 4  # Number of input rain images
    args.n_classes = 4  # Number of target rain images
    args.two_stream_channels = 20  # Number of input images of other variables
    args.img_height = 115  # Height of input images
    args.img_width = 115  # Width of input images
    args.evo_bilinear = True  # Bilinear interpolation is used or not
    args.smaat_bilinear = True  # Bilinear interpolation is used or not
    args.base_c = 16  # Base number of channels (32)
    args.rain_kernelsPL = 2  # kernels per layer
    args.rain_reduc_ratio = 16  # Reduction factor in CBAM
    args.var_kernelsPL = 2  # Number of kernels/filters per double convolutional layer
    args.var_reduc_ratio = 16  # Reduction factor in CBAM
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

    args.save_folder = "path/to/your/results/"

    print("Loading model:")
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # load model weights and biases
    model = get_model_from_str(args)
    path_chkpt = f"./checkpoints/statedict_{model.__class__.__name__}.pt"  # Checkpoint file for model
    state_dict = torch.load(path_chkpt, weights_only=True)

    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()
    save_fn = f"test_results_{model.__class__.__name__}.json"  # Path and file name to store the results in

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

    print("Loading test set:")
    dataset_fn = args.folder_test_set + args.fn_test_set
    # load test set using mad-smaat-gnet utils
    test_dl = madsmaat_data.get_test_loader(
        data_fn=dataset_fn,
        batch_size=1,  # USE CONSISTENT BATCH SIZE WHEN COMPARING MODELS
        num_input_images=args.n_channels,
        num_output_images=args.n_classes,
        shuffle=False,
        num_workers=1,  # Working with cuda
        pin_memory=True,  # Working with cuda
    )

    print("    Test set loaded")

    print("Running tests...")

    test_model(
        model=model,
        dev=dev,
        test_dl=test_dl,
        only_rain_data=args.only_rain_data,
        save_folder=args.save_folder,
        save_fn=save_fn,
        input_len=args.n_channels,
        output_len=args.n_classes,
    )

    print("Process complete")

    return True


if __name__ == "__main__":
    main()
