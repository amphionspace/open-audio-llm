"""LongEnrollTSASR: Qwen3-ASR-1.7B + long-enrollment conditioning, trained end to end.

Two encoder paths:

  mixture (trainable, global attention, one pass):
    mel -> conv (frozen) + ECAPA bias on every frame
        -> layers [0, S)          (S = first GCA layer) -- depth K states -> 24 local queries
                                  (K = mix_condition_layer, 12 by default; 0 = conv output)
        -> layers [S, 24) with GCA(32 tokens) after self-attention
        -> ln_post/proj -> Qwen3 LLM -> CE on the answer
  enrollment (frozen, no_grad):
    whole-enrollment mel -> conv (frozen, shared, computed per 1 s)
        -> frozen copy of the original layers [0, D), attention inside 30 s windows
        -> per-window pooling -> KV memory -> 8 global + 24 local = 32 tokens

The enrollment features never change during training (like the fixed RVQ
codes of the long-prompt TTS). The conv is frozen on both paths; the mixture
encoder layers, the LLM and the new modules are trained, ECAPA is frozen.
Each sample is either BOTH (TS-ASR: ECAPA + 32 tokens)
or NONE (plain / hotword ASR, no enrollment). Both paths are always computed
and multiplied by the activity flags, so every trainable parameter takes part
in each step (DDP-safe without find_unused_parameters).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import (
    AUDIO_TOKENS_PER_SEC,
    conv_frontend,
    length_mask,
    run_encoder_layers,
    tower_head,
)
from .ecapa import embed_fbanks
from .enroll_encoder import EnrollMemory, LongEnrollmentEncoder, window_bounds
from .gca import EncoderGCA, SpeakerPath, normalize_layers

BOTH, NONE = 0, 1
MODE_NAMES = {"both": BOTH, "none": NONE}

# Eval-time ablations: (mode, visible tokens). "global"/"local" mask the other
# token group inside GCA; "none" is plain ASR on the mixture.
ABLATIONS = {
    "both": (BOTH, None),
    "global": (BOTH, "global"),
    "local": (BOTH, "local"),
    "none": (NONE, None),
}


@dataclass
class EnrollmentState:
    """Per-speaker cache: depends only on enrollment audio."""

    memory: EnrollMemory
    global_tokens: Tensor  # [B, 8, H]
    ecapa: Tensor          # [B, 192], L2-normalized


def build_conditioning(module: nn.Module, tower: nn.Module, mcfg) -> None:
    """Attach the enrollment encoder, its frozen layer copy, GCA, gates and the ECAPA path to
    ``module`` under the attribute names checkpoints use. Shared by LongEnrollTSASR (training)
    and LongEnrollFrontend (inference), so both load the same weights."""
    hidden = int(tower.config.d_model)
    module.use_long = bool(mcfg.gca.get("enabled", True))
    module.use_speaker = bool(mcfg.speaker.get("enabled", True))
    module.injection_layers = normalize_layers(mcfg.gca.layers, len(tower.layers))
    module.shared_depth = min(module.injection_layers)
    module.enroll_encoder = LongEnrollmentEncoder(hidden, mcfg.enroll_encoder)
    # Enrollment features: frozen copy of the original layers [0, D); the
    # conv in front of it is the frozen conv shared with the mixture path.
    module.enroll_tower = nn.Module()
    module.enroll_tower.layers = nn.ModuleList(
        copy.deepcopy(tower.layers[i]) for i in range(max(module.enroll_encoder.feature_depths)))
    module.enroll_tower.requires_grad_(False)
    ecfg = mcfg.enroll_encoder
    module.window_tokens = round(float(ecfg.get("window_sec", 30.0)) * AUDIO_TOKENS_PER_SEC)
    module.min_tail_tokens = round(float(ecfg.get("min_tail_sec", 5.0)) * AUDIO_TOKENS_PER_SEC)
    module.mix_condition_layer = int(ecfg.mix_condition_layer)
    if module.mix_condition_layer > module.shared_depth:
        raise ValueError("mix_condition_layer must be <= first GCA layer (single encoder pass)")
    module.gca = nn.ModuleDict({
        str(i): EncoderGCA(hidden, int(mcfg.gca.num_heads), float(mcfg.gca.dropout))
        for i in module.injection_layers
    })
    module.gca_gates = nn.ParameterDict({
        str(i): nn.Parameter(torch.tensor(float(mcfg.gca.gate_init))) for i in module.injection_layers
    })
    module.speaker = SpeakerPath(int(mcfg.speaker.ecapa_dim), hidden, float(mcfg.speaker.gate_init))


class LongEnrollTSASR(nn.Module):
    def __init__(self, asr_model: nn.Module, ecapa: nn.Module, cfg):
        super().__init__()
        mcfg = cfg.model
        self.asr = asr_model
        self.ecapa = ecapa
        self.ecapa.requires_grad_(False)
        self.thinker = asr_model.thinker
        self.tower = self.thinker.audio_tower
        # Train the audio tower (except the conv), the LLM and the LM head.
        self.asr.requires_grad_(False)
        for module in (self.tower, self.thinker.model, self.thinker.lm_head):
            module.requires_grad_(True)
        for module in (self.tower.conv2d1, self.tower.conv2d2, self.tower.conv2d3, self.tower.conv_out):
            module.requires_grad_(False)
        self.audio_token_id = int(self.thinker.config.audio_token_id)
        build_conditioning(self, self.tower, mcfg)

        tcfg = cfg.get("train") or {}
        if bool(tcfg.get("gradient_checkpointing", True)):
            text_cfg = self.thinker.config.text_config
            if float(getattr(text_cfg, "attention_dropout", 0.0) or 0.0) != 0.0:
                raise ValueError("LLM checkpointing runs the text model in train mode; needs dropout 0")
            self.thinker.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self._llm_checkpointing = True
        else:
            self._llm_checkpointing = False

    # ------------------------------------------------------------ bookkeeping
    def train(self, mode: bool = True):
        super().train(mode)
        self.asr.eval()  # no dropout in the backbone
        self.ecapa.eval()
        self.enroll_tower.eval()
        if mode and self._llm_checkpointing:
            self.thinker.model.train()  # HF only checkpoints in train mode; dropout is 0
        return self

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return {n: p.detach().cpu().contiguous() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, state: dict[str, Tensor], strict: bool = True) -> None:
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        missing = sorted(set(params) - set(state))
        unexpected = sorted(set(state) - set(params))
        if strict and (missing or unexpected):
            raise RuntimeError(f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        with torch.no_grad():
            for name, param in params.items():
                if name in state:
                    param.copy_(state[name].to(param.device, param.dtype))

    def optimizer_groups(self, lr: float, backbone_lr: float, gate_lr_mult: float, weight_decay: float):
        """New modules, gates and backbone. Biases and normalization weights (every 1-D
        parameter) take no weight decay, like the Hugging Face Trainer behind AmphionASR v3/v4;
        its name patterns would miss the enrollment ``depth_norms``, the shape rule does not.
        The backbone group stays last: training logs its learning rate."""
        groups = {key: [] for key in ("new", "new_no_decay", "gates", "backbone_no_decay", "backbone")}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("gca_gates.") or name == "speaker.gate":
                key = "gates"
            else:
                key = "backbone" if name.startswith("asr.") else "new"
                if param.ndim < 2:
                    key += "_no_decay"
            groups[key].append(param)
        settings = {
            "new": (lr, weight_decay),
            "new_no_decay": (lr, 0.0),
            "gates": (lr * gate_lr_mult, 0.0),
            "backbone_no_decay": (backbone_lr, 0.0),
            "backbone": (backbone_lr, weight_decay),
        }
        return [{"name": key, "params": params, "lr": settings[key][0], "weight_decay": settings[key][1]}
                for key, params in groups.items() if params]

    def gate_values(self) -> dict[str, float]:
        out = {f"gca_{k}": float(torch.tanh(v.detach())) for k, v in self.gca_gates.items()}
        out["spk"] = float(torch.tanh(self.speaker.gate.detach()))
        return out

    @property
    def device(self) -> torch.device:
        return self.tower.conv2d1.weight.device

    # ------------------------------------------------------------ encoder parts
    @torch.no_grad()
    def enrollment_features(self, mels: Sequence[Tensor], window_batch: int = 64):
        """Frozen enrollment encoder. Whole-enrollment mels -> conv (computed per
        1 s) -> 30 s windows -> frozen layer copy, attention inside each window.
        Returns {depth: [N, T, H]} over the N windows of all enrollments, the
        window lengths [N] and the enrollment index of each window."""
        hidden, lengths = conv_frontend(self.tower, mels)
        windows, owners = [], []
        for index, length in enumerate(lengths.tolist()):
            for start, end in window_bounds(length, self.window_tokens, self.min_tail_tokens):
                windows.append(hidden[index, start:end])
                owners.append(index)
        depths = self.enroll_encoder.feature_depths
        outs: dict[int, list[Tensor]] = {d: [] for d in depths}
        for start in range(0, len(windows), window_batch):
            group = windows[start:start + window_batch]
            lens = torch.tensor([w.size(0) for w in group], device=hidden.device)
            x = nn.utils.rnn.pad_sequence(group, batch_first=True)
            _, states = run_encoder_layers(self.enroll_tower, x, length_mask(lens, x.size(1)), 0,
                                           len(self.enroll_tower.layers), collect=depths)
            for d in depths:
                outs[d].append(states[d])
        max_len = max(t.size(1) for t in outs[depths[0]])
        feats = {d: torch.cat([F.pad(t, (0, 0, 0, max_len - t.size(1))) for t in outs[d]]) for d in depths}
        window_lengths = torch.tensor([w.size(0) for w in windows], device=hidden.device)
        return feats, window_lengths, owners

    def encode_enrollment(self, mels: Sequence[Tensor], fbanks: Sequence[Tensor]) -> EnrollmentState:
        feats, lengths, owners = self.enrollment_features(mels)
        keys, values = self.enroll_encoder.compress_chunks(feats, lengths)
        memory = LongEnrollmentEncoder.gather_memory(keys, values, owners, len(mels))
        global_tokens = self.enroll_encoder.global_tokens(memory)
        ecapa = F.normalize(embed_fbanks(self.ecapa, fbanks, self.device), dim=-1)
        return EnrollmentState(memory, global_tokens, ecapa)

    def encode_mix(self, mix_mels: Sequence[Tensor], enroll: EnrollmentState, modes: Tensor,
                   token_mask: Optional[Tensor] = None) -> Tensor:
        """Conditioned encoder pass. Returns packed audio embeddings [sum(lengths), 2048]."""
        long_active = modes.eq(BOTH).float() * float(self.use_long)
        spk_active = modes.eq(BOTH).float() * float(self.use_speaker)
        hidden, lengths = conv_frontend(self.tower, mix_mels)
        mask = length_mask(lengths, hidden.size(1))
        cond = hidden  # depth 0: conv output, before the ECAPA bias
        spk_bias = spk_active[:, None] * self.speaker(enroll.ecapa.float())
        hidden = hidden + spk_bias[:, None, :].to(hidden.dtype)
        depth = self.mix_condition_layer
        hidden, states = run_encoder_layers(self.tower, hidden, mask, 0, self.shared_depth,
                                            collect=[depth] if depth > 0 else [])
        if depth > 0:
            cond = states[depth]
        local = self.enroll_encoder.local_tokens(enroll.memory, cond, mask)
        tokens = torch.cat([enroll.global_tokens.to(local.dtype), local], dim=1)

        def hook(layer_index: int, h: Tensor) -> Tensor:
            key = str(layer_index)
            if key not in self.gca:
                return torch.zeros_like(h)
            # In the GCA weights' own precision: fp32 master weights keep it fp32 (as before);
            # bf16 weights (DeepSpeed bf16) run it in bf16.
            dtype = self.gca_gates[key].dtype
            with torch.autocast(device_type=h.device.type, enabled=False):
                gca = self.gca[key](h.to(dtype), tokens.to(dtype), token_mask)
                out = torch.tanh(self.gca_gates[key]) * long_active[:, None, None].to(dtype) * gca
            return out.to(h.dtype)

        hidden, _ = run_encoder_layers(self.tower, hidden, mask, self.shared_depth, len(self.tower.layers),
                                       post_attention_hook=hook)
        embeds = tower_head(self.tower, hidden)
        return embeds[mask]

    # ------------------------------------------------------------ LLM
    def _inputs_embeds(self, input_ids: Tensor, audio_embeds: Tensor) -> Tensor:
        embeds = self.thinker.get_input_embeddings()(input_ids)
        audio_mask = input_ids.eq(self.audio_token_id)
        if int(audio_mask.sum()) != audio_embeds.size(0):
            raise RuntimeError(f"audio pads {int(audio_mask.sum())} != audio frames {audio_embeds.size(0)}")
        return embeds.masked_scatter(audio_mask[..., None].expand_as(embeds), audio_embeds.to(embeds.dtype))

    def llm_ce(self, input_ids: Tensor, attention_mask: Tensor, labels: Tensor,
               audio_embeds: Tensor) -> tuple[Tensor, Tensor]:
        """Returns per-sample summed CE [B] and label-token counts [B]."""
        embeds = self._inputs_embeds(input_ids, audio_embeds)
        position_ids, _ = self.thinker.get_rope_index(attention_mask)
        hidden = self.thinker.model(inputs_embeds=embeds, attention_mask=attention_mask,
                                    position_ids=position_ids, use_cache=False).last_hidden_state
        targets = labels[:, 1:]
        valid = targets.ne(-100)
        logits = self.thinker.lm_head(hidden[:, :-1][valid]).float()
        ce = F.cross_entropy(logits, targets[valid], reduction="none")
        rows = torch.nonzero(valid, as_tuple=False)[:, 0]
        sums = torch.zeros(labels.size(0), device=ce.device, dtype=ce.dtype).index_add(0, rows, ce)
        return sums, valid.sum(dim=1)

    # ------------------------------------------------------------ training
    def forward(self, batch: dict) -> dict[str, Tensor]:
        device = self.device
        modes = batch["mode"].to(device)
        # bf16 autocast here, not only at the call site: DeepSpeed's engine switches off an
        # outer autocast, and the fp32 norms / softmax below rely on it to meet bf16 weights.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            enroll = self.encode_enrollment(batch["enroll_mels"], batch["ecapa_fbank"])
            audio = self.encode_mix(batch["mix_mels"], enroll, modes)
            ce_sum, n_tok = self.llm_ce(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                        batch["labels"].to(device), audio)
        out = {"loss": ce_sum.sum() / n_tok.sum().clamp_min(1), "n_tokens": n_tok.sum().detach().float(),
               # Per-sample sums for per-task logging; they carry no gradient.
               "sample_ce": ce_sum.detach(), "sample_tokens": n_tok.detach()}
        for name, code in MODE_NAMES.items():
            out[f"mode_{name}"] = modes.eq(code).sum().detach().float()
        return out

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, mix_mels: Sequence[Tensor], prompt_ids: Sequence[Sequence[int]],
                 enroll: EnrollmentState, ablation: str = "both", max_new_tokens: int = 256,
                 pad_id: int = 151643, eos_ids: Sequence[int] = (151643, 151645)) -> list[list[int]]:
        mode, visible = ABLATIONS[ablation]
        device = self.device
        bsz = len(mix_mels)
        token_mask = None
        if visible is not None:
            g = self.enroll_encoder.global_count
            token_mask = torch.zeros(bsz, g + self.enroll_encoder.local_count, dtype=torch.bool, device=device)
            if visible == "global":
                token_mask[:, :g] = True
            else:
                token_mask[:, g:] = True
        modes = torch.full((bsz,), mode, device=device, dtype=torch.long)
        audio = self.encode_mix(mix_mels, enroll, modes, token_mask)
        return self.generate_from_audio(audio, prompt_ids, max_new_tokens, pad_id, eos_ids)

    @torch.no_grad()
    def generate_from_audio(self, audio: Tensor, prompt_ids: Sequence[Sequence[int]], max_new_tokens: int = 256,
                            pad_id: int = 151643, eos_ids: Sequence[int] = (151643, 151645)) -> list[list[int]]:
        """Greedy decoding from packed audio embeddings [sum(n_audio), 2048] (the encode_mix output);
        each prompt holds its n_audio <|audio_pad|> tokens."""
        device = self.device
        bsz = len(prompt_ids)
        max_len = max(len(p) for p in prompt_ids)
        ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        for row, prompt in enumerate(prompt_ids):  # left padding
            ids[row, max_len - len(prompt):] = torch.tensor(prompt, device=device)
            attn[row, max_len - len(prompt):] = 1
        embeds = self._inputs_embeds(ids, audio)
        out = self.thinker.generate(
            inputs_embeds=embeds, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, eos_token_id=list(eos_ids), pad_token_id=pad_id,
        )
        results = []
        for row in out.tolist():
            seq = []
            for token in row:
                if token in eos_ids:
                    break
                seq.append(token)
            results.append(seq)
        return results
