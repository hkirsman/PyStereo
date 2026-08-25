"""AOT-GAN InpaintGenerator — vendored from researchmm/AOT-GAN-for-Inpainting.

Original repository: https://github.com/researchmm/AOT-GAN-for-Inpainting
License: Apache-2.0
Paper: "Aggregated Contextual Transformations for High-Resolution Image
Inpainting" (Zeng et al., TVCG 2023, arXiv:2104.01431)

Only the generator is included (discriminator not needed for inference).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseNetwork(nn.Module):
    def init_weights(self, init_type: str = "normal", gain: float = 0.02) -> None:
        def _init(m: nn.Module) -> None:
            cn = m.__class__.__name__
            if cn.find("InstanceNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (
                cn.find("Conv") != -1 or cn.find("Linear") != -1
            ):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(_init)


def _layer_norm(feat: torch.Tensor) -> torch.Tensor:
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    return 5.0 * (2.0 * (feat - mean) / std - 1.0)


class _UpConv(nn.Module):
    def __init__(self, inc: int, outc: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(
            F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True),
        )


class _AOTBlock(nn.Module):
    def __init__(self, dim: int, rates: tuple[int, ...]) -> None:
        super().__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.add_module(
                f"block{i:02d}",
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // len(rates), 3, padding=0, dilation=rate),
                    nn.ReLU(True),
                ),
            )
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1),
        )
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [getattr(self, f"block{i:02d}")(x) for i in range(len(self.rates))]
        out_cat = self.fuse(torch.cat(out, 1))
        mask = torch.sigmoid(_layer_norm(self.gate(x)))
        return x * (1 - mask) + out_cat * mask


class InpaintGenerator(_BaseNetwork):
    """AOT-GAN generator for image inpainting.

    Takes ``(image, mask)`` where image is ``(B, 3, H, W)`` in ``[-1, 1]``
    and mask is ``(B, 1, H, W)`` in ``{0, 1}`` (1 = hole).  Returns
    ``(B, 3, H, W)`` in ``[-1, 1]`` (full image, not just the masked region).
    """

    RATES: tuple[int, ...] = (1, 2, 4, 8)
    BLOCK_NUM: int = 8

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.middle = nn.Sequential(
            *[_AOTBlock(256, self.RATES) for _ in range(self.BLOCK_NUM)],
        )
        self.decoder = nn.Sequential(
            _UpConv(256, 128),
            nn.ReLU(True),
            _UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
        )
        self.init_weights()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return torch.tanh(x)
