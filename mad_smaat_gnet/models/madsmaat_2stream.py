"""SmaAt-UNet with 2-stream: the SmaAt-UNet model extended with an encoder for inputs of other variables."""

# Base model taken from: https://github.com/HansBambel/SmaAt-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mad_smaat_gnet.models.madsmaat_components import (
    TwoStreamEncoder,
    SPADE,
    UpDSwithSPADE,
    UnetEncoder,
)
from smaat.unet_parts_depthwise_separable import UpDS
from smaat.unet_parts import OutConv


class TwoStreamDecoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        two_stream_channels: int,
        dec_kernelsPL: int,
        base_c: int = 16,
        smaat_bilinear: bool = True,
    ):
        super(TwoStreamDecoder, self).__init__()

        ## Parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.smaat_bilinear = smaat_bilinear
        self.base_c = base_c
        self.two_stream_channels = two_stream_channels
        self.dec_kernelsPL = dec_kernelsPL
        factor = 2 if self.smaat_bilinear else 1

        ## SPADE in bottleneck layer
        self.spade_bottle = SPADE(
            rain_channels=self.base_c * 32 // factor,
            cond_channels=self.base_c * 64 // factor,
            n_hidden=self.base_c * 4,
        )

        ## Decoder with SPADE
        self.up1 = UpDSwithSPADE(
            in_channels=self.base_c * 32,
            out_channels=self.base_c * 16 // factor,
            n_hidden=self.base_c * 4,
            cond_channels=self.base_c * 64 // factor,
            bilinear=self.smaat_bilinear,
            kernels_per_layer=dec_kernelsPL,
        )
        self.up2 = UpDSwithSPADE(
            in_channels=self.base_c * 16,
            out_channels=self.base_c * 8 // factor,
            n_hidden=self.base_c * 4,
            cond_channels=self.base_c * 32 // factor,
            bilinear=self.smaat_bilinear,
            kernels_per_layer=dec_kernelsPL,
        )
        self.up3 = UpDSwithSPADE(
            in_channels=self.base_c * 8,
            out_channels=self.base_c * 4 // factor,
            n_hidden=self.base_c * 4,
            cond_channels=self.base_c * 16 // factor,
            bilinear=self.smaat_bilinear,
            kernels_per_layer=dec_kernelsPL,
        )
        self.up4 = UpDSwithSPADE(
            in_channels=self.base_c * 4,
            out_channels=self.base_c * 2,
            n_hidden=self.base_c * 2,
            cond_channels=self.base_c * 4,
            bilinear=self.smaat_bilinear,
            kernels_per_layer=dec_kernelsPL,
        )

        self.outc = OutConv(in_channels=base_c * 2, out_channels=self.n_classes)

    def forward(
        self, x1Att, x2Att, x3Att, x4Att, x5Att, z1Att, z2Att, z3Att, z4Att, z5Att
    ):
        # Up
        x5Att = self.spade_bottle(x5Att, z5Att)
        x = self.up1(x5Att, x4Att, z4Att)
        x = self.up2(x, x3Att, z3Att)
        x = self.up3(x, x2Att, z2Att)
        x = self.up4(x, x1Att, z1Att)
        logits = self.outc(x)
        return logits


class madsmaat_2stream(nn.Module):
    def __init__(self, hparams):
        super(madsmaat_2stream, self).__init__()

        ## Parameters
        self.n_channels = hparams.n_channels  # Input length of rain data
        self.n_classes = hparams.n_classes  # Prediction length
        self.two_stream_channels = (
            hparams.two_stream_channels
        )  # Input length of harmonie data
        self.smaat_bilinear = (
            hparams.smaat_bilinear
        )  # Bilinear interpolation is used or not
        self.base_c = hparams.base_c  # Base number of channels (32)
        self.rain_kernelsPL = hparams.rain_kernelsPL  # kernels per layer
        self.rain_reduc_ratio = hparams.rain_reduc_ratio
        self.var_kernelsPL = hparams.var_kernelsPL
        self.var_reduc_ratio = hparams.var_reduc_ratio
        self.dec_kernelsPL = hparams.dec_kernelsPL

        ## HARMONIE encoder
        self.two_stream_encoder = TwoStreamEncoder(
            self.two_stream_channels,
            self.var_kernelsPL,
            self.var_reduc_ratio,
            self.base_c,
            self.smaat_bilinear,
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

        ## Decoder
        self.two_stream_decoder = TwoStreamDecoder(
            self.n_channels,
            self.n_classes,
            self.two_stream_channels,
            self.dec_kernelsPL,
            self.base_c,
            self.smaat_bilinear,
        )

    def forward(self, x, z):  # x = rain data, z = harmonie data
        # Down harmonie
        z1Att, z2Att, z3Att, z4Att, z5Att = self.two_stream_encoder(z)

        # Down rain
        x1Att, x2Att, x3Att, x4Att, x5Att = self.rain_encoder(x)

        # Up
        logits = self.two_stream_decoder(
            x1Att, x2Att, x3Att, x4Att, x5Att, z1Att, z2Att, z3Att, z4Att, z5Att
        )
        return logits
