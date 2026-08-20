"""Neural network architectures for context-aware layered depth inpainting.

Ported from "3D Photography using Context-aware Layered Depth Inpainting"
(Shih et al., CVPR 2020).  Three specialised networks:

- **Edge network** (GAN encoder–decoder with spectral norm):
  Hallucinate depth-edge continuations in masked regions.
- **Depth network** (U-Net with partial convolutions):
  Inpaint depth guided by hallucinated edges.
- **Color network** (U-Net with partial convolutions):
  Inpaint RGB colour guided by edges and depth.

All three expose a :meth:`forward_3P` method that pads input to a
power-of-two multiple, runs inference, and crops back.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _spectral_norm(module: nn.Module, mode: bool = True) -> nn.Module:
    if mode:
        return nn.utils.spectral_norm(module)
    return module


def _weights_init_kaiming(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
        m, "weight"
    ):
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# ── Partial Convolution building blocks ───────────────────────────────


class PartialConv(nn.Module):
    """Partial convolution (Liu et al., ECCV 2018).

    Multiplies the input by the mask before convolution, then re-normalises
    by the fraction of valid (mask > 0) pixels in the receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.mask_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False
        )
        self.input_conv.apply(_weights_init_kaiming)
        self.slide_winsize = in_channels * kernel_size * kernel_size

        nn.init.constant_(self.mask_conv.weight, 1.0)
        for p in self.mask_conv.parameters():
            p.requires_grad = False

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.input_conv(input * mask)
        output_bias = (
            self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
            if self.input_conv.bias is not None
            else torch.zeros_like(output)
        )

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        output_pre = (output - output_bias) * self.slide_winsize / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        return output, new_mask


class PCBActiv(nn.Module):
    """Partial-conv → optional BatchNorm → optional activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bn: bool = True,
        sample: str = "none-3",
        activ: str | None = "relu",
        conv_bias: bool = False,
    ) -> None:
        super().__init__()
        if sample == "down-5":
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == "down-7":
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == "down-3":
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(
        self, input: torch.Tensor, input_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, "bn"):
            h = self.bn(h)
        if hasattr(self, "activation"):
            h = self.activation(h)
        return h, h_mask


# ── Residual block for the edge GAN ──────────────────────────────────


class _ResnetBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            _spectral_norm(nn.Conv2d(dim, dim, 3, padding=0, dilation=dilation, bias=False)),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ReflectionPad2d(1),
            _spectral_norm(nn.Conv2d(dim, dim, 3, padding=0, dilation=1, bias=False)),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


# ── Edge Inpainting Network (GAN encoder–decoder) ────────────────────


class InpaintEdgeNet(nn.Module):
    """Hallucinate depth-edge continuations inside masked (synthesis) regions.

    Input channels (``forward_3P``):
      RGB (3) + normalised disparity (1) + edge (1) + context (1) + mask (1) = 7
    Output: single-channel edge probability map.
    """

    def __init__(self, residual_blocks: int = 8) -> None:
        super().__init__()
        in_ch, out_ch = 7, 1

        self.encoder_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            _spectral_norm(nn.Conv2d(in_ch, 64, 7, padding=0)),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
        )
        self.encoder_1 = nn.Sequential(
            _spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
        )
        self.encoder_2 = nn.Sequential(
            _spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )

        self.middle = nn.Sequential(*[_ResnetBlock(256, 2) for _ in range(residual_blocks)])

        self.decoder_0 = nn.Sequential(
            _spectral_norm(nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
        )
        self.decoder_1 = nn.Sequential(
            _spectral_norm(nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
        )
        self.decoder_2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64 + 64, out_ch, 7, padding=0),
        )

        self._init_weights()

    def _init_weights(self, gain: float = 0.02) -> None:
        for m in self.modules():
            classname = m.__class__.__name__
            if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
                nn.init.normal_(m.weight.data, 0.0, gain)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif "BatchNorm2d" in classname:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder_0(x)
        x2 = self.encoder_1(x1)
        x3 = self.encoder_2(x2)
        x4 = self.middle(x3)
        x5 = self.decoder_0(torch.cat((x4, x3), dim=1))
        x6 = self.decoder_1(torch.cat((x5, x2), dim=1))
        x7 = self.decoder_2(torch.cat((x6, x1), dim=1))
        return torch.sigmoid(x7)

    @torch.no_grad()
    def forward_3P(
        self,
        mask: torch.Tensor,
        context: torch.Tensor,
        rgb: torch.Tensor,
        disp: torch.Tensor,
        edge: torch.Tensor,
        unit_length: int = 128,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        disp_norm = disp / disp.max().clamp(min=1e-8)
        inp = torch.cat((rgb, disp_norm, edge, context, mask), dim=1)
        n, c, h, w = inp.shape
        pad_h = int(np.ceil(h / unit_length) * unit_length - h)
        pad_w = int(np.ceil(w / unit_length) * unit_length - w)
        ah, aw = pad_h // 2, pad_w // 2
        padded = torch.zeros((n, c, h + pad_h, w + pad_w), device=device)
        padded[..., ah : ah + h, aw : aw + w] = inp
        out = self.forward(padded)
        return out[..., ah : ah + h, aw : aw + w]


# ── Depth Inpainting Network (partial-conv U-Net) ────────────────────


class InpaintDepthNet(nn.Module):
    """Inpaint depth using hallucinated edges as guidance.

    Input channels (``forward_3P``):
      depth (1) + edge (1) + context (1) + mask (1) = 4
    Output: single-channel inpainted depth.
    """

    def __init__(self, layer_size: int = 7, upsampling_mode: str = "nearest") -> None:
        super().__init__()
        in_channels = 4
        out_channels = 1
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size

        self.enc_1 = PCBActiv(in_channels, 64, bn=False, sample="down-7", conv_bias=True)
        self.enc_2 = PCBActiv(64, 128, sample="down-5", conv_bias=True)
        self.enc_3 = PCBActiv(128, 256, sample="down-5")
        self.enc_4 = PCBActiv(256, 512, sample="down-3")
        for i in range(4, self.layer_size):
            setattr(self, f"enc_{i + 1}", PCBActiv(512, 512, sample="down-3"))

        for i in range(4, self.layer_size):
            setattr(self, f"dec_{i + 1}", PCBActiv(512 + 512, 512, activ="leaky"))
        self.dec_4 = PCBActiv(512 + 256, 256, activ="leaky")
        self.dec_3 = PCBActiv(256 + 128, 128, activ="leaky")
        self.dec_2 = PCBActiv(128 + 64, 64, activ="leaky")
        self.dec_1 = PCBActiv(64 + in_channels, out_channels, bn=False, activ=None, conv_bias=True)

    def forward(self, input_feat: torch.Tensor) -> torch.Tensor:
        inp = input_feat
        input_mask = (inp[:, -2:-1] + inp[:, -1:]).clamp(0, 1).repeat(1, inp.shape[1], 1, 1)

        h_dict: dict[str, torch.Tensor] = {"h_0": inp}
        h_mask_dict: dict[str, torch.Tensor] = {"h_0": input_mask}

        h_key_prev = "h_0"
        for i in range(1, self.layer_size + 1):
            l_key = f"enc_{i}"
            h_key = f"h_{i}"
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev]
            )
            h_key_prev = h_key

        h = h_dict[f"h_{self.layer_size}"]
        h_mask = h_mask_dict[f"h_{self.layer_size}"]

        for i in range(self.layer_size, 0, -1):
            enc_h_key = f"h_{i - 1}"
            dec_l_key = f"dec_{i}"
            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode="nearest")
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

        return h

    @torch.no_grad()
    def forward_3P(
        self,
        mask: torch.Tensor,
        context: torch.Tensor,
        depth: torch.Tensor,
        edge: torch.Tensor,
        unit_length: int = 128,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        inp = torch.cat((depth, edge, context, mask), dim=1)
        n, c, h, w = inp.shape
        pad_h = int(np.ceil(h / unit_length) * unit_length - h)
        pad_w = int(np.ceil(w / unit_length) * unit_length - w)
        ah, aw = pad_h // 2, pad_w // 2
        padded = torch.zeros((n, c, h + pad_h, w + pad_w), device=device)
        padded[..., ah : ah + h, aw : aw + w] = inp
        out = self.forward(padded)
        return out[..., ah : ah + h, aw : aw + w]


# ── Color Inpainting Network (partial-conv U-Net) ────────────────────


class InpaintColorNet(nn.Module):
    """Inpaint RGB colour guided by edges.

    Input channels (``forward_3P``):
      RGB (3) + edge (1) + context (1) + mask (1) = 6
    Output: 3-channel inpainted RGB.
    """

    def __init__(self, layer_size: int = 7, upsampling_mode: str = "nearest") -> None:
        super().__init__()
        in_channels = 6
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size

        self.enc_1 = PCBActiv(in_channels, 64, bn=False, sample="down-7")
        self.enc_2 = PCBActiv(64, 128, sample="down-5")
        self.enc_3 = PCBActiv(128, 256, sample="down-5")
        self.enc_4 = PCBActiv(256, 512, sample="down-3")
        self.enc_5 = PCBActiv(512, 512, sample="down-3")
        self.enc_6 = PCBActiv(512, 512, sample="down-3")
        self.enc_7 = PCBActiv(512, 512, sample="down-3")

        self.dec_7 = PCBActiv(512 + 512, 512, activ="leaky")
        self.dec_6 = PCBActiv(512 + 512, 512, activ="leaky")
        self.dec_5A = PCBActiv(512 + 512, 512, activ="leaky")
        self.dec_4A = PCBActiv(512 + 256, 256, activ="leaky")
        self.dec_3A = PCBActiv(256 + 128, 128, activ="leaky")
        self.dec_2A = PCBActiv(128 + 64, 64, activ="leaky")
        self.dec_1A = PCBActiv(64 + in_channels, 3, bn=False, activ=None, conv_bias=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        input_mask = (inp[:, -2:-1] + inp[:, -1:]).clamp(0, 1)
        f_0, h_0 = inp, input_mask.repeat(1, inp.shape[1], 1, 1)
        f_1, h_1 = self.enc_1(f_0, h_0)
        f_2, h_2 = self.enc_2(f_1, h_1)
        f_3, h_3 = self.enc_3(f_2, h_2)
        f_4, h_4 = self.enc_4(f_3, h_3)
        f_5, h_5 = self.enc_5(f_4, h_4)
        f_6, h_6 = self.enc_6(f_5, h_5)
        f_7, h_7 = self.enc_7(f_6, h_6)

        def _up(feat: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                F.interpolate(feat, scale_factor=2, mode=self.upsampling_mode),
                F.interpolate(mask, scale_factor=2, mode="nearest"),
            )

        def _cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.cat((a, b), dim=1)

        o_7, k_7 = _up(f_7, h_7)
        o_6, k_6 = self.dec_7(_cat(o_7, f_6), _cat(k_7, h_6))
        o_6, k_6 = _up(o_6, k_6)
        o_5, k_5 = self.dec_6(_cat(o_6, f_5), _cat(k_6, h_5))
        o_5, k_5 = _up(o_5, k_5)

        o_4, k_4 = self.dec_5A(_cat(o_5, f_4), _cat(k_5, h_4))
        o_4, k_4 = _up(o_4, k_4)
        o_3, k_3 = self.dec_4A(_cat(o_4, f_3), _cat(k_4, h_3))
        o_3, k_3 = _up(o_3, k_3)
        o_2, k_2 = self.dec_3A(_cat(o_3, f_2), _cat(k_3, h_2))
        o_2, k_2 = _up(o_2, k_2)
        o_1, k_1 = self.dec_2A(_cat(o_2, f_1), _cat(k_2, h_1))
        o_1, k_1 = _up(o_1, k_1)
        o_0, _ = self.dec_1A(_cat(o_1, f_0), _cat(k_1, h_0))

        return torch.sigmoid(o_0)

    @torch.no_grad()
    def forward_3P(
        self,
        mask: torch.Tensor,
        context: torch.Tensor,
        rgb: torch.Tensor,
        edge: torch.Tensor,
        unit_length: int = 128,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        inp = torch.cat((rgb, edge, context, mask), dim=1)
        n, c, h, w = inp.shape
        pad_h = int(np.ceil(h / unit_length) * unit_length - h)
        pad_w = int(np.ceil(w / unit_length) * unit_length - w)
        ah, aw = pad_h // 2, pad_w // 2
        padded = torch.zeros((n, c, h + pad_h, w + pad_w), device=device)
        padded[..., ah : ah + h, aw : aw + w] = inp
        out = self.forward(padded)
        return out[..., ah : ah + h, aw : aw + w]
