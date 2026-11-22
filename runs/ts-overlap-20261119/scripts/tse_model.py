"""Enrollment-conditioned mask estimator (SpeakerBeam-style). STFT 512 / hop 128 (32 ms / 8 ms at 16 kHz, as the
oracle masks of task 18). Inputs are RMS-normalised, features are log power. Enrollment: Linear + 2-layer BLSTM,
mean+std pooling -> 256-d speaker vector. Mixture: Linear(257 -> 1024), FiLM by the speaker vector, 2-layer BLSTM,
FiLM again, 2-layer BLSTM, Linear -> sigmoid mask per time-frequency bin. Output = mask x mixture STFT, mixture phase."""
import torch, torch.nn as nn
NFFT, HOP = 512, 128
F = NFFT // 2 + 1


def rms(x, lens=None):
    if lens is None:
        return x.pow(2).mean(-1, keepdim=True).clamp_min(1e-10).sqrt()
    mask = (torch.arange(x.shape[-1], device=x.device)[None] < lens[:, None]).float()
    return ((x.pow(2) * mask).sum(-1, keepdim=True) / lens[:, None].clamp_min(1)).clamp_min(1e-10).sqrt()


class TSE(nn.Module):
    def __init__(self, hid=1024, emb=256, ext_emb=0):
        """ext_emb > 0: forward takes an external speaker vector (e.g. frozen CAM++, 192-d) added to the learned one."""
        super().__init__()
        self.ext = nn.Sequential(nn.Linear(ext_emb, emb), nn.ReLU(), nn.Linear(emb, emb)) if ext_emb else None
        self.register_buffer("win", torch.hann_window(NFFT), persistent=False)
        self.e_in = nn.Sequential(nn.Linear(F, 512), nn.ReLU())
        self.e_lstm = nn.LSTM(512, 256, 2, batch_first=True, bidirectional=True)
        self.e_out = nn.Linear(1024, emb)
        self.m_in = nn.Linear(F, hid)
        self.film1, self.film2 = nn.Linear(emb, 2 * hid), nn.Linear(emb, 2 * hid)
        self.lstm1 = nn.LSTM(hid, hid // 2, 2, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hid, hid // 2, 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hid, F)

    def spec(self, x):
        return torch.stft(x, NFFT, HOP, window=self.win, return_complex=True)  # (B, F, T)

    @staticmethod
    def logpow(X):
        return torch.log(X.real ** 2 + X.imag ** 2 + 1e-8).transpose(1, 2)  # (B, T, F)

    def embed(self, e):
        h, _ = self.e_lstm(self.e_in(self.logpow(self.spec(e / rms(e)))))
        return self.e_out(torch.cat([h.mean(1), h.std(1)], -1))

    def forward(self, e, m, lens=None, spk=None):
        """Mask (B, F, T) and the mixture STFT of m / rms(m) (B, F, T)."""
        Y = self.spec(m / rms(m, lens)); z = self.embed(e)
        if self.ext is not None:
            z = z + self.ext(nn.functional.normalize(spk, dim=-1))
        h = self.m_in(self.logpow(Y))
        g, b = self.film1(z).chunk(2, -1); h, _ = self.lstm1(h * (1 + g[:, None]) + b[:, None])
        g, b = self.film2(z).chunk(2, -1); h, _ = self.lstm2(h * (1 + g[:, None]) + b[:, None])
        return torch.sigmoid(self.out(h)).transpose(1, 2), Y

    @torch.no_grad()
    def extract(self, e, m, spk=None):
        """Waveform of the extracted target at the mixture's level, for one (1, N) mixture."""
        M, Y = self(e, m, spk=spk)
        return torch.istft(M * Y, NFFT, HOP, window=self.win, length=m.shape[-1]) * rms(m)
