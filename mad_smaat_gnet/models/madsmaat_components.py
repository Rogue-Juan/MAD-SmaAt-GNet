"""Components for the MAD-SmaAt-GNet model and ablation study models: encoders, decoders, Evo-Net, and SPADE and up-operations with SPADE."""

# Base model taken from: https://github.com/HansBambel/SmaAt-UNet
# Evo-Net taken from: https://doi.org/10.24433/CO.0832447.v1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from smaat.unet_parts import OutConv, DoubleConv
from smaat.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from smaat.layers import CBAM
from evo_net.evolution_network import Evolution_Network
from evo_net.utils import warp, make_grid


class UnetEncoder(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        rain_kernelsPL,
        rain_reduc_ratio,
        base_c=32,
        smaat_bilinear=True,
    ):
        super(UnetEncoder, self).__init__()

        ## Parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.smaat_bilinear = smaat_bilinear
        base_c = base_c
        self.rain_kernelsPL = rain_kernelsPL
        self.rain_reduc_ratio = rain_reduc_ratio

        ## Rain encoder
        self.inc = DoubleConvDS(
            in_channels=self.n_channels,
            out_channels=base_c * 2,
            kernels_per_layer=rain_kernelsPL,
        )
        self.cbam1r = CBAM(base_c * 2, reduction_ratio=rain_reduc_ratio)
        self.down1r = DownDS(base_c * 2, base_c * 4, kernels_per_layer=rain_kernelsPL)
        self.cbam2r = CBAM(base_c * 4, reduction_ratio=rain_reduc_ratio)
        self.down2r = DownDS(base_c * 4, base_c * 8, kernels_per_layer=rain_kernelsPL)
        self.cbam3r = CBAM(base_c * 8, reduction_ratio=rain_reduc_ratio)
        self.down3r = DownDS(base_c * 8, base_c * 16, kernels_per_layer=rain_kernelsPL)
        self.cbam4r = CBAM(base_c * 16, reduction_ratio=rain_reduc_ratio)
        factor = 2 if self.smaat_bilinear else 1
        self.down4r = DownDS(
            base_c * 16, base_c * 32 // factor, kernels_per_layer=rain_kernelsPL
        )
        self.cbam5r = CBAM(base_c * 32 // factor, reduction_ratio=rain_reduc_ratio)

    def forward(self, x):

        # Down rain
        x1 = self.inc(x)
        x1Att = self.cbam1r(x1)
        x2 = self.down1r(x1)
        x2Att = self.cbam2r(x2)
        x3 = self.down2r(x2)
        x3Att = self.cbam3r(x3)
        x4 = self.down3r(x3)
        x4Att = self.cbam4r(x4)
        x5 = self.down4r(x4)
        x5Att = self.cbam5r(x5)
        return x1Att, x2Att, x3Att, x4Att, x5Att


class Decoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        dec_kernelsPL: int,
        base_c: int = 32,
        smaat_bilinear: bool = True,
    ):
        super(Decoder, self).__init__()

        ## Parameters
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.smaat_bilinear = smaat_bilinear
        self.base_c = base_c
        self.dec_kernelsPL = dec_kernelsPL
        factor = 2 if self.smaat_bilinear else 1

        ## SPADE layers in bottleneck
        self.spade_bottle_2stream = SPADE(
            rain_channels=self.base_c * 32 // factor,
            cond_channels=self.base_c * 64 // factor,
            n_hidden=self.base_c * 4,
        )
        self.spade_bottle_evo = SPADE(
            rain_channels=self.base_c * 32 // factor,
            cond_channels=self.n_classes,
            n_hidden=self.base_c * 4,
        )

        ## Decoder with SPADE
        self.up1 = UpDS_SPADE_2_inputs(
            in_channels=self.base_c * 32,
            out_channels=self.base_c * 16 // factor,
            n_hidden=self.base_c * 4,
            var_channels=self.base_c * 64 // factor,
            evo_channels=self.n_classes,
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
        self.up4 = UpDS_SPADE_2_inputs(
            in_channels=self.base_c * 4,
            out_channels=self.base_c * 2,
            n_hidden=self.base_c * 2,
            var_channels=self.base_c * 4,
            evo_channels=self.n_classes,
            bilinear=self.smaat_bilinear,
            kernels_per_layer=dec_kernelsPL,
        )

        self.outc = OutConv(in_channels=base_c * 2, out_channels=self.n_classes)

    def forward(
        self,
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
    ):
        # Up
        x5Att = self.spade_bottle_2stream(x5Att, z5Att)
        x5Att = self.spade_bottle_evo(x5Att, evo4)
        x = self.up1(x5Att, x4Att, z4Att, evo3)
        x = self.up2(x, x3Att, z3Att)
        x = self.up3(x, x2Att, z2Att)
        x = self.up4(x, x1Att, z1Att, evo_result)
        logits = self.outc(x)
        return logits


class EvoNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        height_rain: int,
        width_rain: int,
        base_c: int = 32,
        evo_bilinear: bool = True,
    ):
        super(EvoNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.height_rain = height_rain
        self.width_rain = width_rain
        self.base_c = base_c
        self.evo_bilinear = evo_bilinear

        ## Evolution Network from NowcastNet for physically consistent rain predictions
        sample_tensor = torch.zeros(1, 1, self.height_rain, self.width_rain)
        self.grid = make_grid(sample_tensor)

        self.evolution_net = Evolution_Network(
            self.n_channels, self.n_classes, base_c=base_c, bilinear=evo_bilinear
        )

    def forward(self, x):
        # Evolution Network
        batch = x.shape[
            0
        ]  # 4D tensor is assumed: batch number, channels/sequence length, height, width
        height = x.shape[2]
        width = x.shape[3]

        intensity, motion = self.evolution_net(x)
        motion_ = motion.reshape(batch, self.n_classes, 2, height, width)
        intensity_ = intensity.reshape(batch, self.n_classes, 1, height, width)
        series = []
        last_frames = x[:, (self.n_channels - 1) : self.n_channels, :, :]
        grid = self.grid.repeat(batch, 1, 1, 1)
        for i in range(self.n_classes):
            last_frames = warp(
                last_frames,
                motion_[:, i],
                grid.cuda(),
                mode="nearest",
                padding_mode="border",
            )
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)
        return evo_result


class UpDSwithSPADE(nn.Module):
    """Upscaling then double conv then SPADE"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        n_hidden: int,
        bilinear: bool = True,
        kernels_per_layer: int = 1,
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConvDS(
                in_channels, out_channels, kernels_per_layer=kernels_per_layer
            )

        self.spade = SPADE(out_channels, cond_channels, n_hidden)

    def forward(self, x1, x2, z):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.spade(x, z)
        return x


class SPADE(nn.Module):
    """Residual SPADE block"""

    def __init__(self, rain_channels: int, cond_channels: int, n_hidden: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(rain_channels, affine=False)

        self.shared = nn.Sequential(
            nn.Conv2d(cond_channels, n_hidden, kernel_size=3, padding=1), nn.ReLU()
        )
        self.gamma = nn.Conv2d(n_hidden, rain_channels, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(n_hidden, rain_channels, kernel_size=3, padding=1)

    def forward(self, x, z):
        norm_x = self.norm(x)
        z_feat = self.shared(z)
        gamma = self.gamma(z_feat)
        beta = self.beta(z_feat)
        residual = gamma * norm_x + beta
        return x + residual


class TwoStreamEncoder(nn.Module):
    def __init__(
        self,
        var_channels: int,
        var_kernelsPL: int,
        var_reduc_ratio: int,
        base_c: int = 16,
        smaat_bilinear: bool = True,
    ):
        super(TwoStreamEncoder, self).__init__()

        ## Parameters
        self.var_channels = var_channels
        self.base_c = base_c
        self.smaat_bilinear = smaat_bilinear
        self.var_kernelsPL = var_kernelsPL
        self.var_reduc_ratio = var_reduc_ratio

        self.inc = DoubleConvDS(
            self.var_channels, self.base_c * 4, kernels_per_layer=var_kernelsPL
        )
        ##### Two-stream down operations
        self.cbam1h = CBAM(self.base_c * 4, reduction_ratio=var_reduc_ratio)
        self.down1h = DownDS(
            self.base_c * 4, self.base_c * 8, kernels_per_layer=var_kernelsPL
        )
        self.cbam2h = CBAM(self.base_c * 8, reduction_ratio=var_reduc_ratio)
        self.down2h = DownDS(
            self.base_c * 8, self.base_c * 16, kernels_per_layer=var_kernelsPL
        )
        self.cbam3h = CBAM(self.base_c * 16, reduction_ratio=var_reduc_ratio)
        self.down3h = DownDS(
            self.base_c * 16, self.base_c * 32, kernels_per_layer=var_kernelsPL
        )
        self.cbam4h = CBAM(self.base_c * 32, reduction_ratio=var_reduc_ratio)
        factor = 2 if self.smaat_bilinear else 1
        self.down4h = DownDS(
            self.base_c * 32,
            self.base_c * 64 // factor,
            kernels_per_layer=var_kernelsPL,
        )
        self.cbam5h = CBAM(self.base_c * 64 // factor, reduction_ratio=var_reduc_ratio)

    def forward(self, z):
        # Down two-stream
        z1 = self.inc(z)
        z1Att = self.cbam1h(z1)
        z2 = self.down1h(z1)
        z2Att = self.cbam2h(z2)
        z3 = self.down2h(z2)
        z3Att = self.cbam3h(z3)
        z4 = self.down3h(z3)
        z4Att = self.cbam4h(z4)
        z5 = self.down4h(z4)
        z5Att = self.cbam5h(z5)
        return z1Att, z2Att, z3Att, z4Att, z5Att


class UpDS_SPADE_2_inputs(nn.Module):
    """Upscaling then double conv then SPADE for both two-stream input and evo-net input"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        var_channels: int,
        evo_channels: int,
        n_hidden: int,
        bilinear: bool = True,
        kernels_per_layer: int = 1,
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConvDS(
                in_channels, out_channels, kernels_per_layer=kernels_per_layer
            )

        self.spade_var = SPADE(out_channels, var_channels, n_hidden)
        self.spade_evo = SPADE(out_channels, evo_channels, n_hidden)

    def forward(self, x1, x2, z, evo):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.spade_var(x, z)
        x = self.spade_evo(x, evo)
        return x
