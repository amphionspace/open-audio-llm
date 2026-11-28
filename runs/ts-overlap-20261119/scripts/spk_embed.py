"""Frozen CAM++ (zh-cn, 192-d) speaker embedding on GPU: Kaldi fbank 80 (per-utterance mean removed), as the
ModelScope wrapper. Batched: (B, N) waveforms at 16 kHz -> (B, 192)."""
import torch, torchaudio.compliance.kaldi as kaldi
from campplus import CAMPPlus
W = "/workspace/data/ts-overlap-20261119/task19-tse/campplus/campplus_cn_common.bin"


class SpeakerEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = CAMPPlus(80, 192); self.net.load_state_dict(torch.load(W, map_location="cpu"), strict=True)
        self.net.eval()
        for p in self.net.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, wav):
        feats = []
        for w in wav:
            f = kaldi.fbank(w[None] * 32768, num_mel_bins=80, sample_frequency=16000)
            feats.append(f - f.mean(0, keepdim=True))
        return self.net(torch.stack(feats))

    def train(self, mode=True):  # always eval: frozen
        return super().train(False)
