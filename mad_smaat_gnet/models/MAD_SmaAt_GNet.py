import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mad_smaat_gnet.models.madsmaat_components import (
    TwoStreamEncoder,
    Decoder,
    EvoNet,
    UnetEncoder,
)


class MAD_SmaAt_GNet(nn.Module):
    def __init__(self, hparams):
        super(MAD_SmaAt_GNet, self).__init__()
        ## Parameters
        self.n_channels = hparams.n_channels  # Input length of rain data
        self.n_classes = hparams.n_classes  # Prediction length
        self.harmo_channels = (
            hparams.two_stream_channels
        )  # Input length of harmonie data
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
        self.var_kernelsPL = hparams.var_kernelsPL
        self.var_reduc_ratio = hparams.var_reduc_ratio
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

        self.maxp3 = nn.MaxPool2d(8)
        self.maxp4 = nn.MaxPool2d(16)

        ## HARMONIE encoder
        self.two_stream_encoder = TwoStreamEncoder(
            self.harmo_channels,
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
        self.decoder = Decoder(
            self.n_channels,
            self.n_classes,
            self.dec_kernelsPL,
            self.base_c,
            self.smaat_bilinear,
        )

    def forward(self, x, z):  # x = rain data, z = harmonie data
        # Evolution Network
        evo_result = self.evo_net(x)

        # Maxpool evo_result so that it can be added to each decoder step
        evo3 = self.maxp3(evo_result)
        evo4 = self.maxp4(evo_result)

        # Down harmonie
        z1Att, z2Att, z3Att, z4Att, z5Att = self.two_stream_encoder(z)

        # Down rain
        x1Att, x2Att, x3Att, x4Att, x5Att = self.rain_encoder(x)

        # Up
        logits = self.decoder(
            x1Att,
            x2Att,
            x3Att,
            x4Att,
            x5Att,
            z1Att,
            z2Att,
            z3Att,
            z4Att,
            z5Att,
            evo_result,
            evo3,
            evo4,
        )
        return logits
