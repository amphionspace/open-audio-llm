"""GRPO rollouts decoded by an in-process vLLM engine instead of transformers ``generate``.

The policy's own audio tower turns each prompt's audio into embeddings (the same template
encoding, SEP handling and encoder code as the training forward), and vLLM only decodes text
from those embeddings with ``Qwen3ASRTargetSOTForVLLM``. Prompt token ids come from the training
template, so the scored prompt is token-for-token the rollout prompt. vLLM's text weights are
reloaded from the policy whenever the optimizer has stepped.

The engine replaces ms-swift's ``TransformersEngine`` (``trainer.engine``) and keeps its
``infer`` contract: one ``ChatCompletionResponse`` per request, with token ids, sampled-token
logprobs and the finish reason. Identical prompts (a GRPO group) become one vLLM request with
``n`` samples, so their prefill is shared.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

AUDIO_PAD = "<|audio_pad|>"
AUDIO_TOKENS = ("<|audio_start|>", "<|audio_end|>", AUDIO_PAD)  # never valid in a completion
# The vLLM decoder never runs the audio tower: embeddings come from the policy.
AUDIO_PREFIX = "thinker.audio_tower."
# Activation-checkpoint wrappers rename parameters; vLLM expects checkpoint names.
WRAPPER_PREFIXES = ("_checkpoint_wrapped_module.",)


def collapse_audio_pads(input_ids: list[int], pad_id: int, expected: int) -> list[int]:
    """Replace the template's expanded run of audio pads with the single placeholder vLLM expands."""
    positions = [i for i, token in enumerate(input_ids) if token == pad_id]
    if len(positions) != expected:
        raise ValueError(f"prompt has {len(positions)} audio pads but the encoder produced {expected} frames")
    if positions and positions[-1] - positions[0] + 1 != len(positions):
        raise ValueError("audio pads must form one contiguous run")
    if not positions:
        return list(input_ids)
    return [*input_ids[: positions[0]], pad_id, *input_ids[positions[-1] + 1:]]


def prompt_key(encoded: dict) -> str:
    """Identity of a rendered prompt: its token ids, audio features and enrollment split."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(torch.as_tensor(encoded["input_ids"]).numpy().tobytes())
    features = encoded.get("input_features")
    if features is not None:
        digest.update(features.detach().cpu().contiguous().float().numpy().tobytes())
    digest.update(str(int(encoded.get("enroll_n_frames") or 0)).encode())
    return digest.hexdigest()


def group_prompts(encoded: list[dict]) -> "OrderedDict[str, list[int]]":
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, item in enumerate(encoded):
        groups.setdefault(prompt_key(item), []).append(index)
    return groups


def policy_weights(model):
    """Text-model parameters of the policy under checkpoint names."""
    for name, param in model.named_parameters():
        for prefix in WRAPPER_PREFIXES:
            name = name.replace(prefix, "")
        # The joint MTP module (joint_mtp.py) is not part of the served policy.
        if not name.startswith(AUDIO_PREFIX) and not name.startswith("mtp."):
            yield name, param.detach()


@contextmanager
def _expandable_segments(enabled: bool):
    from swift.rlhf_trainers.utils import set_expandable_segments

    set_expandable_segments(enabled)
    try:
        yield
    finally:
        set_expandable_segments(not enabled)


class VllmAudioRolloutEngine:
    def __init__(self, hf_engine, model_dir: str, *, gpu_memory_utilization: float, max_model_len: int,
                 max_num_seqs: int, seed: int, external_launcher: bool = True):
        os.environ["OPEN_AUDIO_LLM_TARGET_SOT_VLLM"] = "1"
        # Weight sync reaches the model through the in-process executor.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        if os.environ.get("AMPHION_TSASR_INSERT_SEP", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise ValueError("AMPHION_TSASR_INSERT_SEP would make training mel extraction split clips")
        from open_audio_llm.integrations.vllm.plugin import register

        register()
        from vllm import LLM

        self.hf_engine = hf_engine
        self.tokenizer = hf_engine.template.tokenizer
        self.pad_id = self.tokenizer.convert_tokens_to_ids(AUDIO_PAD)
        # A sampled audio placeholder makes the scoring forward see more audio positions than features
        # (masked_scatter assert; GRPO round 9 crashed at step 273 after "<|audio_start|>" completions).
        # Rollout logprobs are processed_logprobs, so they renormalize over the rest: these tokens' mass is ~0.
        audio_ids = self.tokenizer.convert_tokens_to_ids(list(AUDIO_TOKENS))
        if self.tokenizer.unk_token_id is not None and self.tokenizer.unk_token_id in audio_ids:
            raise ValueError("audio special tokens missing from the tokenizer")
        self.blocked = {token: -100.0 for token in audio_ids}  # vLLM clamps logit_bias to [-100, 100]
        kwargs = {"distributed_executor_backend": "external_launcher"} if external_launcher else {}
        with _expandable_segments(False):
            self.llm = LLM(
                model=model_dir,
                hf_overrides={"architectures": ["Qwen3ASRTargetSOTForVLLM"]},
                enable_mm_embeds=True,
                limit_mm_per_prompt={"audio": 1},
                dtype="bfloat16",
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                enable_prefix_caching=False,
                logprobs_mode="processed_logprobs",
                seed=seed,
                **kwargs,
            )
        self.synced_step = None

    @property
    def model(self):
        # ms-swift reads engine.model in a few places; it is the policy, not vLLM's copy.
        return self.hf_engine.model

    def _vllm_model(self):
        return self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model

    @torch.no_grad()
    def sync_weights(self, model) -> int:
        vllm_model = self._vllm_model()
        loaded = vllm_model.load_weights(policy_weights(model))
        if self.synced_step is None:
            # The first load must cover every text-model parameter, or vLLM would silently
            # decode with its own startup copy of what was not loaded.
            expected = {name for name, _ in vllm_model.named_parameters() if not name.startswith("audio_tower.")}
            missing = expected - set(loaded)
            if missing:
                raise ValueError(f"policy weights did not cover vLLM parameters: {sorted(missing)[:10]}")
            logger.warning("vLLM rollout: first weight sync loaded %d tensors", len(loaded))
        self.llm.reset_prefix_cache()
        return len(loaded)

    @torch.no_grad()
    def _audio_embeddings(self, encoded: list[dict]) -> list[torch.Tensor]:
        from swift.utils import to_device

        model = self.hf_engine.model
        template = self.hf_engine.template
        batch = to_device(template.data_collator([dict(item) for item in encoded]), model.device)
        thinker = model.thinker if hasattr(model, "thinker") else model
        kwargs = {"feature_attention_mask": batch["feature_attention_mask"]}
        if "enroll_n_frames" in batch:
            kwargs["enroll_n_frames"] = batch["enroll_n_frames"]
        flat = thinker.get_audio_features(batch["input_features"].to(model.dtype), **kwargs)
        counts = [list(item["input_ids"]).count(self.pad_id) for item in encoded]
        if sum(counts) != flat.shape[0]:
            raise ValueError(f"encoder produced {flat.shape[0]} frames for {sum(counts)} audio pads")
        return list(flat.split(counts))

    def infer(self, infer_requests, request_config=None, metrics=None, *, use_tqdm=None, adapter_request=None):
        from swift.infer_engine.protocol import (
            ChatCompletionResponse,
            ChatCompletionResponseChoice,
            ChatMessage,
            UsageInfo,
        )
        from vllm import SamplingParams

        if adapter_request is not None or (request_config is not None and request_config.stream):
            raise NotImplementedError("vLLM audio rollout supports full-model, non-streaming generation")
        if not infer_requests:
            return []
        template = self.hf_engine.template
        if template.use_model:
            template.model = self.hf_engine.model
        template.set_mode("transformers")
        encoded, _ = self.hf_engine._batch_encode(infer_requests, strict=True)
        template_inputs = [item.pop("template_inputs") for item in encoded]
        groups = group_prompts(encoded)
        firsts = [indices[0] for indices in groups.values()]
        embeddings = self._audio_embeddings([encoded[i] for i in firsts])
        prompts, params = [], []
        max_model_len = self.llm.llm_engine.model_config.max_model_len
        for first, embedding, indices in zip(firsts, embeddings, groups.values()):
            ids = collapse_audio_pads(list(encoded[first]["input_ids"]), self.pad_id, len(embedding))
            # One audio item of shape (1, frames, hidden); a 2-D tensor would count every frame as an item.
            item = {"audio_embeds": embedding.unsqueeze(0).cpu()}
            prompts.append({"prompt_token_ids": ids, "multi_modal_data": {"audio": item}})
            # Without a limit the transformers engine fills the context; vLLM's default is 16 tokens.
            max_tokens = request_config.max_tokens or max_model_len - len(encoded[first]["input_ids"])
            params.append(SamplingParams(
                n=len(indices),
                temperature=request_config.temperature if request_config.temperature is not None else 1.0,
                top_p=request_config.top_p if request_config.top_p is not None else 1.0,
                top_k=request_config.top_k if request_config.top_k not in (None, 0) else -1,
                min_p=request_config.min_p or 0.0,
                repetition_penalty=request_config.repetition_penalty or 1.0,
                stop=request_config.stop or None,
                max_tokens=max_tokens,
                logprobs=0,
                skip_special_tokens=False,
                logit_bias=self.blocked,
            ))
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        pad_token = self.tokenizer.pad_token_id
        results = [None] * len(encoded)
        for output, indices in zip(outputs, groups.values()):
            expected = len(encoded[indices[0]]["input_ids"])
            if len(output.prompt_token_ids) != expected:
                raise ValueError(f"vLLM prompt has {len(output.prompt_token_ids)} tokens, template {expected}")
            for index, completion in zip(indices, output.outputs, strict=True):
                # ms-swift's transformers engine drops pad ids (<|endoftext|> doubles as pad and eos).
                kept = [(t, lp) for t, lp in zip(completion.token_ids, completion.logprobs) if t != pad_token]
                token_ids = [t for t, _ in kept]
                logprobs = {"content": [
                    {"token": self.tokenizer.convert_ids_to_tokens(t), "logprob": lp[t].logprob,
                     "bytes": None, "top_logprobs": []}
                    for t, lp in kept
                ]}
                text = template.decode_generate_ids(token_ids, template_inputs=template_inputs[index])
                results[index] = ChatCompletionResponse(
                    model="vllm-rollout",
                    choices=[ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=text),
                        finish_reason=completion.finish_reason,
                        logprobs=logprobs,
                        token_ids=token_ids,
                    )],
                    usage=UsageInfo(prompt_tokens=expected, completion_tokens=len(token_ids),
                                    total_tokens=expected + len(token_ids)),
                    prompt_token_ids=list(encoded[index]["input_ids"]),
                )
        return results


def install_vllm_rollout(trainer, options: dict) -> VllmAudioRolloutEngine:
    """Swap the GRPO rollout engine; reload vLLM text weights after each optimizer step."""
    if getattr(trainer.args, "use_vllm", False):
        raise ValueError("rollout.engine=vllm replaces ms-swift's own vLLM path; keep use_vllm false")
    unknown = set(options) - {"engine", "gpu_memory_utilization", "max_model_len"}
    if unknown or options.get("engine") != "vllm":
        raise ValueError(f"rollout supports engine=vllm, gpu_memory_utilization, max_model_len; got {options}")
    args = trainer.args
    per_rank = args.per_device_train_batch_size * getattr(args, "steps_per_generation", 1)
    engine = VllmAudioRolloutEngine(
        trainer.engine,
        trainer.model.model_dir,
        gpu_memory_utilization=float(options.get("gpu_memory_utilization", 0.25)),
        max_model_len=int(options.get("max_model_len", 4096)),
        max_num_seqs=max(per_rank, 8),
        seed=trainer.accelerator.process_index,
    )
    trainer.engine = engine
    logger.warning("vLLM rollout engine ready on rank %d", trainer.accelerator.process_index)
    rollout = trainer._rollout_samples

    def rollout_with_current_weights(inputs):
        if engine.synced_step != trainer.state.global_step:
            engine.sync_weights(trainer.model)
            engine.synced_step = trainer.state.global_step
        return rollout(inputs)

    trainer._rollout_samples = rollout_with_current_weights
    return engine
