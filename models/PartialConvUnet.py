""" Full assembly of the parts to form the complete network """
import torch

from .pd_unet_parts import *
from .generate_mask import maskGen

class PartialConvUnet(nn.Module):
    def __init__(self, features=16, n_channels=2, n_classes=1, bilinear=False):
        super(PartialConvUnet, self).__init__()
        self.mask = nn.Parameter(torch.from_numpy(maskGen()), requires_grad=False)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features, features*2)
        self.down2 = Down(features*2, features*4)
        self.down3 = Down(features*4, features*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(features*8, features*16 // factor)
        self.up1 = Up(features*16, features*8 // factor, bilinear)
        self.up2 = Up(features*8, features*4 // factor, bilinear)
        self.up3 = Up(features*4, features*2 // factor, bilinear)
        self.up4 = Up(features*2, features, bilinear)
        self.outc = OutConv(features, n_classes)

    def forward(self, x, l):
        x_input = x
        x = torch.cat([x,l],dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        x = x_input + self.mask * logits
        return x