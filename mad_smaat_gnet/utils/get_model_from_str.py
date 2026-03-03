import argparse
from mad_smaat_gnet.models.MAD_SmaAt_GNet import MAD_SmaAt_GNet
from mad_smaat_gnet.models.madsmaat_2stream import madsmaat_2stream
from mad_smaat_gnet.models.madsmaat_evo import madsmaat_evo
from mad_smaat_gnet.models.madsmaat_components import EvoNet
from smaat.SmaAt_UNet import SmaAt_UNet
import os
import sys

def get_model_from_str(args):
    if args.model_name.lower() in ["mad_smaat_gnet", "full model", "mad-smaat-gnet"]:
        return MAD_SmaAt_GNet(hparams=args)
    elif args.model_name.lower() in ["2-stream model", "madsmaat_2stream"]:
        return madsmaat_2stream(hparams=args)
    elif args.model_name.lower() in ["evo-net model", "madsmaat_evo"]:
        return madsmaat_evo(hparams=args)
    elif args.model_name.lower() in ["smaat_unet", "smaat u-net", "smaat-unet"]:
        return SmaAt_UNet(
            n_channels=args.n_channels,
            n_classes=args.n_classes,
            kernels_per_layer=args.rain_kernelsPL,
            bilinear=args.smaat_bilinear,
            reduction_ratio=args.rain_reduc_ratio,
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