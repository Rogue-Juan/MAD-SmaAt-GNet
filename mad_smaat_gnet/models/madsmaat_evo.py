"""SmaAt-UNet with Evo-Net: the SmaAt-UNet model extended with the evolution network from the paper Skilful nowcasting of extreme precipitation with NowcastNet"""

# Base model taken from: https://github.com/HansBambel/SmaAt-UNet
# Evo-Net taken from: https://doi.org/10.24433/CO.0832447.v1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mad_smaat_gnet.models.madsmaat_components import EvoNet
from mad_smaat_gnet.models.madsmaat_components import UpDSwithSPADE, SPADE, UnetEncoder
from smaat.unet_parts import OutConv
from smaat.unet_parts_depthwise_separable import UpDS


class EvoDecoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        dec_kernelsPL: int,
        base_c: int = 32,
        smaat_bilinear: bool = True,
    ):
        super(EvoDecoder, self).__init__()

        ## Parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.smaat_bilinear = smaat_bilinear
        self.base_c = base_c
        self.dec_kernelsPL = dec_kernelsPL
        factor = 2 if self.smaat_bilinear else 1

        ## SPADE in bottleneck layer
        self.spade_bottle = SPADE(
            rain_channels=self.base_c * 32 // factor,
            cond_channels=self.n_classes,
            n_hidden=self.base_c * 4,
        )

        ## Decoder with SPADE
        self.up1 = UpDSwithSPADE(
            in_channels=self.base_c * 32,
            out_channels=self.base_c * 16 // factor,
            n_hidden=self.base_c * 4,
            cond_channels=self.n_classes,
            bilinear=self.smaat_bilinear,
            kernels_per_layer=dec_kernelsPL,
        )
        self.up2 = UpDS(
            in_channels=self.base_c * 16,
            out_channels=self.base_c * 8 // factor,
            bilinear=self.smaat_bilinear,
            kernels_per_layer=dec_kernelsPL,
        )
        self.up3 = UpDS(
            in_channels=self.base_c * 8,
            out_channels=self.base_c * 4 // factor,
            bilinear=self.smaat_bilinear,
            kernels_per_layer=dec_kernelsPL,
        )
        self.up4 = UpDSwithSPADE(
            in_channels=self.base_c * 4,
            out_channels=self.base_c * 2,
            n_hidden=self.base_c * 2,
            cond_channels=self.n_classes,
            bilinear=self.smaat_bilinear,
            kernels_per_layer=dec_kernelsPL,
        )

        self.outc = OutConv(in_channels=base_c * 2, out_channels=self.n_classes)

    def forward(self, x1Att, x2Att, x3Att, x4Att, x5Att, evo_result, evo3, evo4):
        # Up
        x5Att = self.spade_bottle(x5Att, evo4)
        x = self.up1(x5Att, x4Att, evo3)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att, evo_result)
        logits = self.outc(x)
        return logits


class madsmaat_evo(nn.Module):
    def __init__(self, hparams):
        super(madsmaat_evo, self).__init__()

        ## Parameters
        self.n_channels = hparams.n_channels  # Input length of rain data
        self.n_classes = hparams.n_classes  # Prediction length
        self.img_height = hparams.img_height
        self.img_width = hparams.img_width
        self.evo_bilinear = (
            hparams.evo_bilinear
        )  # Bilinear interpolation is used or not
        self.smaat_bilinear = (
            hparams.smaat_bilinear
        )  # Bilinear interpolation is used or not
        self.base_c = hparams.base_c  # Base number of channels (32)
        self.rain_kernelsPL = hparams.rain_kernelsPL  # kernels per layer
        self.rain_reduc_ratio = hparams.rain_reduc_ratio
        self.dec_kernelsPL = hparams.dec_kernelsPL

        ## Evolution Network from NowcastNet for physically consistent rain predictions
        self.evo_net = EvoNet(
            self.n_channels,
            self.n_classes,
            self.img_height,
            self.img_width,
            self.base_c,
            self.evo_bilinear,
        )

        ## Rain encoder
        self.rain_encoder = UnetEncoder(
            self.n_channels,
            self.n_classes,
            self.rain_kernelsPL,
            self.rain_reduc_ratio,
            self.base_c,
            self.smaat_bilinear,
        )

        self.maxp3 = nn.MaxPool2d(8)
        self.maxp4 = nn.MaxPool2d(16)

        ## Decoder
        self.decoder = EvoDecoder(
            self.n_channels,
            self.n_classes,
            self.dec_kernelsPL,
            self.base_c,
            self.smaat_bilinear,
        )

    def forward(self, x):  # x = rain data

        # Evolution Network
        evo_result = self.evo_net(x)

        # Maxpool evo_result so that it can be added to each decoder step
        evo3 = self.maxp3(evo_result)
        evo4 = self.maxp4(evo_result)

        # Down rain
        x1Att, x2Att, x3Att, x4Att, x5Att = self.rain_encoder(x)

        # Up
        logits = self.decoder(x1Att, x2Att, x3Att, x4Att, x5Att, evo_result, evo3, evo4)
        return logits
