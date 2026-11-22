"""v3 extractor: MossFormer2_SS_16K (ClearerVoice-Studio 6b3774d, Apache-2.0, pretrained two-speaker separator)
conditioned on a frozen CAM++ speaker vector by FiLM on the encoder output before the mask net (zero-initialised, so
it starts as the pretrained separator). Output 0 = the enrolled speaker, output 1 = everything else. Inference runs
4 s windows with 3 s stride and cross-fades the 1 s overlaps (output 0 is the target in every window)."""
import sys, types
from pathlib import Path
import torch, torch.nn as nn
EXT = Path(__file__).resolve().parent.parent / "ext"
sys.path.insert(0, str(EXT / "pylib")); sys.path.insert(0, str(EXT / "ClearerVoice-Studio/clearvoice/clearvoice/models"))
from mossformer2_ss.mossformer2 import MossFormer2_SS_16K  # noqa: E402
CKPT = EXT / "ckpt/last_best_checkpoint.pt"
SR, WIN, HOPW = 16000, 4 * 16000, 3 * 16000


DEEP = (5, 11, 17)  # FiLM after these of the 24 FLASH + gated-FSMN layers of the mask net (deep=True)


class MF2TSE(nn.Module):
    def __init__(self, pretrained=True, spk_dim=192, deep=False):
        super().__init__()
        self._z = None
        args = types.SimpleNamespace(encoder_embedding_dim=512, mossformer_sequence_dim=512, num_mossformer_layer=24,
                                     encoder_kernel_size=16, num_spks=2)
        self.mf = MossFormer2_SS_16K(args).model
        if pretrained:
            state = torch.load(CKPT, map_location="cpu"); state = state.get("model", state); own = self.mf.state_dict()
            for k in own:
                for c in (k, k.replace("module.", ""), "module." + k):
                    if c in state and state[c].shape == own[k].shape:
                        own[k] = state[c]; break
                else:
                    raise KeyError(k)
            self.mf.load_state_dict(own)
        self.film = nn.Sequential(nn.Linear(spk_dim, 512), nn.ReLU(), nn.Linear(512, 1024))
        nn.init.zeros_(self.film[2].weight); nn.init.zeros_(self.film[2].bias)
        self.film_deep = nn.ModuleList()
        if deep:
            fsmn = self.mf.mask_net.mdl.intra_mdl.mossformerM.fsmn
            for i in DEEP:
                f = nn.Sequential(nn.Linear(spk_dim, 512), nn.ReLU(), nn.Linear(512, 1024))
                nn.init.zeros_(f[2].weight); nn.init.zeros_(f[2].bias); self.film_deep.append(f)
                fsmn[i].register_forward_hook(self._hook(len(self.film_deep) - 1))

    def _hook(self, j):
        def hook(module, inputs, output):  # output (B, L, 512)
            g, b = self.film_deep[j](self._z).chunk(2, -1)
            return output * (1 + g[:, None, :]) + b[:, None, :]
        return hook

    def forward(self, mix, spk):
        """mix (B, N) at unit RMS, spk (B, 192) -> (target (B, N), rest (B, N))."""
        mf = self.mf; x = mf.enc(mix)  # (B, 512, L)
        self._z = nn.functional.normalize(spk, dim=-1).to(x.dtype) if not torch.is_autocast_enabled() else nn.functional.normalize(spk, dim=-1)
        g, b = self.film(self._z).chunk(2, -1)
        mask = mf.mask_net(x * (1 + g[..., None]) + b[..., None])
        outs = [mf.dec(x * mask[i]) for i in range(2)]
        outs = [o[None] if o.dim() == 1 else o for o in outs]  # the decoder squeezes a batch of one
        n = mix.shape[-1]
        outs = [nn.functional.pad(o, (0, n - o.shape[-1])) if o.shape[-1] < n else o[:, :n] for o in outs]
        return outs[0], outs[1]

    @torch.no_grad()
    def extract(self, mix, spk):
        """(1, N) mixture at any level -> (1, N) target at the mixture's level, windowed."""
        scale = mix.pow(2).mean().clamp_min(1e-10).sqrt(); x = mix / scale; n = x.shape[-1]
        if n <= WIN:
            return self(nn.functional.pad(x, (0, WIN - n)), spk)[0][:, :n] * scale
        total = WIN + -(-(n - WIN) // HOPW) * HOPW; x = nn.functional.pad(x, (0, total - n))
        out = torch.zeros(1, total, device=x.device); ov = WIN - HOPW
        fade = torch.linspace(0, 1, ov, device=x.device)
        for s in range(0, total - WIN + 1, HOPW):
            y = self(x[:, s:s + WIN], spk)[0]
            if s:
                out[:, s:s + ov] = out[:, s:s + ov] * (1 - fade) + y[:, :ov] * fade; out[:, s + ov:s + WIN] = y[:, ov:]
            else:
                out[:, :WIN] = y
        return out[:, :n] * scale
