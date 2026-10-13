import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    groups = min(4, outputs)
    while outputs % groups:
        groups -= 1
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(groups, outputs), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(groups, outputs), nn.ReLU())


class UNet(nn.Module):
    """Two-level U-Net; optional independently encoded RGB/high-pass stems."""

    def __init__(self, base_channels=16, variant='rgb'):
        super().__init__()
        if variant not in ('rgb', 'residual') or base_channels < 2 or base_channels % 2:
            raise ValueError('Choose rgb/residual and a positive even base width >= 2')
        self.variant = variant
        if variant == 'rgb':
            self.stem = block(3, base_channels)
        else:
            self.rgb_stem = block(3, base_channels // 2)
            self.residual_stem = block(3, base_channels // 2)
        self.encoder = block(base_channels, base_channels * 2)
        self.bottom = block(base_channels * 2, base_channels * 4)
        self.decoder2 = block(base_channels * 6, base_channels * 2)
        self.decoder1 = block(base_channels * 3, base_channels)
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, rgb):
        if self.variant == 'rgb':
            level1 = self.stem(rgb)
        else:
            smooth = F.avg_pool2d(F.pad(rgb, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
            level1 = torch.cat([self.rgb_stem(rgb), self.residual_stem(rgb - smooth)], 1)
        level2 = self.encoder(F.max_pool2d(level1, 2))
        bottom = self.bottom(F.max_pool2d(level2, 2))
        up2 = F.interpolate(bottom, size=level2.shape[-2:], mode='bilinear', align_corners=False)
        up2 = self.decoder2(torch.cat([up2, level2], 1))
        up1 = F.interpolate(up2, size=level1.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.decoder1(torch.cat([up1, level1], 1)))


def segmentation_loss(logits, target, valid):
    valid = valid.float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * valid).sum() / valid.sum().clamp_min(1)
    probabilities = logits.sigmoid() * valid
    target = target * valid
    dice = (2 * (probabilities * target).sum((1, 2, 3)) + 1) / (probabilities.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + (1 - dice).mean()
