"""Frozen ECAPA-TDNN (wespeaker ECAPA_TDNN_GLOB_c512, VoxCeleb, 192-d).

Vendored model definition so the checkpoint loads without installing
wespeaker. ``load_ecapa`` loads ``avg_model.pt`` strictly (classifier head
dropped). Input: 16 kHz waveform -> 80-dim Kaldi fbank (hamming, x32768,
no dither) -> per-utterance CMN.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from torch import Tensor, nn


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList(
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.width) for _ in range(self.nums))

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for index, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if index >= 1:
                sp = sp + spx[index]
            sp = bn(F.relu(conv(sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class SEConnect(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, bottleneck)
        self.linear2 = nn.Linear(bottleneck, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = torch.sigmoid(self.linear2(F.relu(self.linear1(out))))
        return x * out.unsqueeze(2)


class SERes2Block(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1),
            Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
            Conv1dReluBn(channels, channels, kernel_size=1),
            SEConnect(channels),
        )

    def forward(self, x):
        return x + self.se_res2block(x)


class ASTP(nn.Module):
    """Attentive statistics pooling with global context."""

    def __init__(self, in_dim, bottleneck=128):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim * 3, bottleneck, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck, in_dim, kernel_size=1)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True).expand_as(x)
        std = torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-7).expand_as(x)
        alpha = torch.tanh(self.linear1(torch.cat((x, mean, std), dim=1)))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mu = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * x * x, dim=2) - mu * mu
        return torch.cat([mu, torch.sqrt(var.clamp(min=1e-7))], dim=1)


class ECAPATDNN(nn.Module):
    def __init__(self, channels=512, feat_dim=80, embed_dim=192):
        super().__init__()
        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SERes2Block(channels, 3, 1, 2, 2, 8)
        self.layer3 = SERes2Block(channels, 3, 1, 3, 3, 8)
        self.layer4 = SERes2Block(channels, 3, 1, 4, 4, 8)
        self.conv = nn.Conv1d(channels * 3, 1536, kernel_size=1)
        self.pool = ASTP(1536)
        self.bn = nn.BatchNorm1d(1536 * 2)
        self.linear = nn.Linear(1536 * 2, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, feats: Tensor) -> Tensor:
        """feats: [B, T, 80] -> [B, 192]."""
        x = feats.permute(0, 2, 1)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.relu(self.conv(torch.cat([out2, out3, out4], dim=1)))
        return self.linear(self.bn(self.pool(out)))


def compute_fbank(wav: Tensor, sample_rate: int = 16000) -> Tensor:
    """[T] float waveform in [-1, 1] -> [frames, 80] with CMN."""
    wav = wav.float().reshape(1, -1) * (1 << 15)
    feats = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=sample_rate,
        window_type="hamming",
        use_energy=False,
    )
    return feats - feats.mean(dim=0, keepdim=True)


def load_ecapa(model_dir: str) -> ECAPATDNN:
    path = Path(model_dir)
    ckpt = path / "avg_model.pt" if path.is_dir() else path
    state = torch.load(str(ckpt), map_location="cpu")
    state = {k: v for k, v in state.items() if not k.startswith("projection.")}
    model = ECAPATDNN()
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def embed_fbanks(model: ECAPATDNN, fbanks: Sequence[Tensor], device) -> Tensor:
    """Per-utterance forward (no padding into the statistics pooling)."""
    out = []
    for feats in fbanks:
        out.append(model(feats.to(device=device, dtype=torch.float32)[None])[0])
    return torch.stack(out)
