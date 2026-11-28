"""Adapter module of train_adapter.py (shared with decode_adapter.py)."""
import torch, torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim)); self.register_buffer("std", torch.ones(dim))
        self.inp = nn.Linear(dim, 768); self.pos = nn.Embedding(4096, 768)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(768, 12, 3072, 0.1, batch_first=True, norm_first=True), 4)
        self.norm = nn.LayerNorm(768); self.out = nn.Linear(768, dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x, pad=None):  # x: (B, T, dim) raw features -> raw features
        z = (x - self.mean) / self.std
        h = self.inp(z) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return x + self.out(self.norm(self.enc(h, src_key_padding_mask=pad))) * self.std

