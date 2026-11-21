"""Replace text placeholder tokens with projected audio embeddings."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from open_audio_llm.constants import IGNORE_TOKEN_ID
from .special_tokens import token_id_is_set


class SlotMerger(nn.Module):
    """Multi-audio slot merger ported from the legacy `src/model.py` behavior."""

    def __init__(self, config, hidden_size: int):
        super().__init__()
        self.config = config
        self.prompt_embedding = nn.Embedding(config.num_prompt_tokens, hidden_size)

    def forward(
        self,
        audio_features: list[torch.Tensor],
        audio_lengths: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not audio_features:
            raise ValueError("At least one audio feature slot is required")
        batch_size = audio_features[0].shape[0]
        final_embeds = []
        final_masks = []
        final_labels = []

        for i in range(batch_size):
            active = attention_mask[i].bool()
            active_ids = input_ids[i][active]
            active_embeds = inputs_embeds[i][active].clone()
            active_labels = labels[i][active] if labels is not None else None

            self._inject_prompt_embeddings(active_ids, active_embeds)
            speech_pos = (
                active_ids == self.config.default_speech_token_id
            ).nonzero(as_tuple=True)[0]
            n_speech = int(speech_pos.numel())
            if n_speech < 1:
                raise ValueError("Prompt must contain at least one <speech> token")
            if n_speech > len(audio_features):
                raise ValueError(
                    f"Prompt has {n_speech} <speech> tokens but only "
                    f"{len(audio_features)} audio feature slots were provided."
                )

            selected_features = audio_features[-n_speech:]
            selected_lengths = audio_lengths[-n_speech:]
            embed_parts = []
            label_parts = [] if labels is not None else None
            current = 0

            for slot_idx, pos in enumerate(torch.sort(speech_pos)[0].tolist()):
                feat_len = int(selected_lengths[slot_idx][i].item())
                embed_parts.append(active_embeds[current:pos])
                embed_parts.append(selected_features[slot_idx][i, :feat_len])
                if label_parts is not None and active_labels is not None:
                    label_parts.append(active_labels[current:pos])
                    label_parts.append(
                        torch.full(
                            (feat_len,),
                            IGNORE_TOKEN_ID,
                            dtype=active_labels.dtype,
                            device=active_labels.device,
                        )
                    )
                current = pos + 1

            embed_parts.append(active_embeds[current:])
            merged = torch.cat(embed_parts, dim=0)
            final_embeds.append(merged.flip(dims=[0]))
            final_masks.append(
                torch.ones(
                    merged.shape[0],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ).flip(dims=[0])
            )
            if label_parts is not None and active_labels is not None:
                label_parts.append(active_labels[current:])
                final_labels.append(torch.cat(label_parts, dim=0).flip(dims=[0]))

        padded_embeds = pad_sequence(final_embeds, batch_first=True, padding_value=0.0)
        padded_masks = pad_sequence(final_masks, batch_first=True, padding_value=False)
        padded_embeds = padded_embeds.flip(dims=[1])
        padded_masks = padded_masks.flip(dims=[1])

        if labels is None:
            return padded_embeds, padded_masks, None
        padded_labels = pad_sequence(
            final_labels,
            batch_first=True,
            padding_value=IGNORE_TOKEN_ID,
        ).flip(dims=[1])
        return padded_embeds, padded_masks, padded_labels

    def _inject_prompt_embeddings(
        self,
        active_ids: torch.Tensor,
        active_embeds: torch.Tensor,
    ) -> None:
        token_to_prompt = [
            (self.config.start_text_token_id, 0),
            (self.config.end_text_token_id, 1),
            (self.config.start_speech_token_id, 2),
            (self.config.end_speech_token_id, 3),
        ]
        for token_id, prompt_idx in token_to_prompt:
            if not token_id_is_set(token_id):
                continue
            positions = (active_ids == token_id).nonzero(as_tuple=True)[0]
            for pos in positions:
                active_embeds[pos] = self.prompt_embedding.weight[prompt_idx].to(
                    active_embeds.dtype
                )
