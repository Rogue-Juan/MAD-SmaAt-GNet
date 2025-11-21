from torch import nn
from smaat.unet_parts import OutConv
from smaat.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from smaat.layers import CBAM


class SmaAt_UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
        base_c=32,
    ):
        super(SmaAt_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio
        base_c = base_c

        self.inc = DoubleConvDS(
            self.n_channels, base_c * 2, kernels_per_layer=kernels_per_layer
        )
        self.cbam1 = CBAM(base_c * 2, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(base_c * 2, base_c * 4, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(base_c * 4, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(base_c * 4, base_c * 8, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(base_c * 8, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(
            base_c * 8, base_c * 16, kernels_per_layer=kernels_per_layer
        )
        self.cbam4 = CBAM(base_c * 16, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(
            base_c * 16, base_c * 32 // factor, kernels_per_layer=kernels_per_layer
        )
        self.cbam5 = CBAM(base_c * 32 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(
            base_c * 32,
            base_c * 16 // factor,
            self.bilinear,
            kernels_per_layer=kernels_per_layer,
        )
        self.up2 = UpDS(
            base_c * 16,
            base_c * 8 // factor,
            self.bilinear,
            kernels_per_layer=kernels_per_layer,
        )
        self.up3 = UpDS(
            base_c * 8,
            base_c * 4 // factor,
            self.bilinear,
            kernels_per_layer=kernels_per_layer,
        )
        self.up4 = UpDS(
            base_c * 4, base_c * 2, self.bilinear, kernels_per_layer=kernels_per_layer
        )

        self.outc = OutConv(base_c * 2, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
