"""Where does a GRPO step on target-absent prompts move the presence decision, and through which tokens?

From the start checkpoint S of GRPO v23-v25 (SFT 073 avg3) this rebuilds the training signal of one
GRPO step and splits it, then asks what each part does to held-out evaluation samples.

1. Rollouts: prompts drawn from the v25 data config's own plans (``GRPOPromptPlan``, the exact audio
   the trainer would see): look-alike target-absent (v25 plan), v15 target-absent, v15 target-present.
   ``--generations`` samples each at temperature 1 (HF generate on S), scored with the training reward
   functions and weights, presence contrast applied, advantages z-scored per group (ms-swift
   ``scale_rewards: group``).
2. Gradient parts: at S the step is on-policy (ratio 1, KL gradient 0), so the GRPO ascent direction of a
   group is ``sum_i A_i / L_i * sum_t grad log p(token_t)`` (``loss_type: grpo``, per-sequence token mean).
   It is accumulated separately per prompt type and per token part: ``decision`` = the language-name
   token (`` None`` vs a language; the presence decision), ``rest`` = every other completion token
   (``language``, ``<asr_text>``, the transcript, the end token).
3. Held-out: ts-asr-benchmark v12 samples of chosen families (hard / easy target-absent, target-present).
   Per sample, the gradient of the decision margin ``m = log p(None) - max(log p(Chinese), log p(English))``
   is dotted with each part (unit-normalised), per module group, giving the first-order change of m per
   unit step along that part. The same dot with realised updates (``--delta NAME=CKPT``, e.g. v25 avg3 - S)
   and the actual margins of those checkpoints check what training really did.

A part that raises m on hard negatives as much as on positives only moves the threshold; a useful part
raises m on negatives and not on positives. Adam rescales coordinates, so raw gradient directions are
compared with the realised deltas rather than trusted alone.

Usage (oal-train env, PYTHONPATH=src, 2 GPUs):
  grad_attribution.py --out DIR [--prompts-per-type 64] [--generations 8] [--per-family 40]
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

DATA_CONFIG = "/workspace/workspace/mingdong/repo/oal-grpo/runs/amphionasr2-grpo-20261101/configs/train-data-grpo-v25.yaml"
START = "/workspace/data/ae-runs/ts-robust-073-avg3/checkpoint-avg3"
V12_RUN = "/workspace/data/ae-runs/ts-robust-073-avg3/runs/run-20261107T024705Z-ff0a6030"  # S on ts-asr-benchmark v12
PROMPT_TYPES = {  # name -> GRPOPromptPlan.origin
    "absent-lookalike": "38-grpo-v25-lookalike-mined-from-sft073/plan-ts-absent",
    "absent-v15": "28-grpo-v15-newprompt-from-sft073/plan-ts-absent",
    "present-v15": "28-grpo-v15-newprompt-from-sft073/plan-ts-present",
}
PARTS = ("decision", "rest")
REWARD_WEIGHTS = {"accuracy": 1.0, "presence": 1.0, "hotword": 1.0, "format": 0.25}  # train-grpo-v25.yaml
FAMILIES = {  # v12 split -> role
    **{f: "hard-neg" for f in ("short-hard-neg", "abs-hard", "abs-hard-k1", "realreg3-hard-neg", "enroll-short1-neg",
                               "enroll-contam0-neg")},
    **{f: "easy-neg" for f in ("short-rand-neg", "abs-rand-k1", "abs-k2", "enroll-clean-neg")},
    **{f: "pos" for f in ("short-k1-pos", "dyn-full-k1", "dyn-burst-k1", "dyn-overlap-k1", "dyn-alternate-k2",
                          "enroll-clean-pos", "enroll-short1-pos", "enroll-contam0-pos")},
}
DEFAULT_DELTAS = ("v15=/workspace/data/ae-runs/amphionasr2-grpo-v15avg3-ckpt400/checkpoint",
                  "v23=/workspace/data/ae-runs/amphionasr2-grpo-v23avg3-ckpt400/checkpoint",
                  "v24=/workspace/data/ae-runs/amphionasr2-grpo-v24avg3-ckpt400/checkpoint",
                  "v25=/workspace/data/ae-runs/amphionasr2-grpo-v25avg3-ckpt400/checkpoint")
ADV_EPS = 1e-4  # ms-swift adds 1e-4 to the group std


def module_group(name):
    m = re.search(r"audio_tower\.layers\.(\d+)\.", name)
    if m:
        return f"audio.{int(m.group(1)) // 8 * 8:02d}-{int(m.group(1)) // 8 * 8 + 7:02d}"
    if "audio_tower" in name:
        return "audio.other"
    m = re.search(r"model\.layers\.(\d+)\.", name)
    if m:
        i = int(m.group(1))
        return "llm.00-09" if i < 10 else "llm.10-19" if i < 20 else "llm.20-27"
    if "lm_head" in name:
        return "lm_head"
    return "llm.other"


def load_model(checkpoint):
    from swift.model import get_model_processor
    from swift.template import get_template

    import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401

    model, processor = get_model_processor(checkpoint, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16,
                                           attn_impl="sdpa", device_map="cuda:0")
    model.eval()
    model.thinker.audio_tower.config._attn_implementation = "sdpa"  # one global window, as training and serving
    from safetensors import safe_open

    with safe_open(str(Path(checkpoint) / "model.safetensors"), framework="pt") as f:
        sep = f.get_tensor("thinker.audio_tower.sep_token.weight")
    tower_sep = model.thinker.audio_tower.sep_token.weight
    assert torch.equal(tower_sep.detach().cpu(), sep.to(tower_sep.dtype)), "SEP not loaded"
    template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b")
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return model, template, tokenizer


def to_device(batch):
    batch = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in batch.items()}
    batch["input_features"] = batch["input_features"].to(torch.bfloat16)
    return batch


def rewards(output, solution, messages):
    from open_audio_llm.integrations.ms_swift.rewards import qwen3_asr

    hot = qwen3_asr.hotword(output, solution, qwen3_asr.prompt_hotwords(messages))
    parts = {"accuracy": qwen3_asr.accuracy(output, solution), "presence": qwen3_asr.presence(output, solution),
             "hotword": hot, "format": float(qwen3_asr.well_formed(output))}
    return parts, sum(REWARD_WEIGHTS[k] * v for k, v in parts.items() if v is not None)


def build_prompts(args):
    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
    from open_audio_llm.integrations.ms_swift.grpo_prompts import GRPOPromptPlan

    config = read_data_config(DATA_CONFIG)
    dataset = CatalogSwiftDataset(config, encode=None, grpo=True, message_format="qwen3_asr")
    plan = GRPOPromptPlan(dataset, **config["grpo"])
    rng = random.Random(args.seed)
    chosen = []
    for kind, origin in PROMPT_TYPES.items():
        indices = [i for i, o in enumerate(plan.origin) if o == origin]
        chosen += [(kind, i) for i in rng.sample(indices, args.prompts_per_type)]
    return dataset, plan, chosen


def raw_prompt(dataset, plan, index):
    sample = dataset[tuple(plan.plan[index])]
    return {"messages": sample["messages"], "audios": sample["audios"], "mix_wav": sample["mix_wav"],
            "solution": sample["solution"], "record_id": dataset.records[plan.plan[index][1]].record.id,
            "mix_info": sample.get("online_mix_info", {})}


def rollouts(args, model, template, tokenizer, dataset, plan, chosen):
    """Sample, score and z-score like one GRPO step at S; cached in rollouts.jsonl."""
    from open_audio_llm.integrations.ms_swift.grpo_contrast import contrast_indices

    path = Path(args.out) / "rollouts.jsonl"
    if path.exists():
        return [json.loads(line) for line in open(path)]
    template.set_mode("transformers")
    groups, start = [], time.time()
    for n, (kind, index) in enumerate(chosen):
        prompt = raw_prompt(dataset, plan, index)
        encoded = template.encode({k: prompt[k] for k in ("messages", "audios", "mix_wav")})
        batch = to_device(template.data_collator([encoded] * args.generations))
        torch.manual_seed(args.seed * 100003 + index)
        with torch.inference_mode():
            out = model.generate(**batch, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=512, use_cache=True)
        out = getattr(out, "sequences", out)
        prompt_len = batch["input_ids"].shape[1]
        if out.shape[1] < prompt_len or not torch.equal(out[:, :prompt_len], batch["input_ids"]):
            prompt_len = 0  # the model returned only the new tokens
        # Cut at the first end token and keep tags such as <asr_text> (special tokens to the tokenizer).
        ends = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id, tokenizer.pad_token_id}
        outputs = []
        for row in out[:, prompt_len:].tolist():
            cut = next((i for i, t in enumerate(row) if t in ends), len(row))
            outputs.append(tokenizer.decode(row[:cut], skip_special_tokens=False).strip())
        contrast = contrast_indices(outputs, [prompt["solution"]] * len(outputs), len(outputs))
        if contrast:
            outputs[-1] = prompt["solution"]
        scored = [rewards(o, prompt["solution"], prompt["messages"]) for o in outputs]
        total = np.array([r for _, r in scored])
        std = total.std(ddof=1)
        advantages = (total - total.mean()) / (std + ADV_EPS)
        groups.append({"kind": kind, "plan_index": index, "record_id": prompt["record_id"],
                       "solution": prompt["solution"], "mix_info": prompt["mix_info"], "contrast": bool(contrast),
                       "outputs": outputs, "rewards": [p for p, _ in scored], "reward": total.tolist(),
                       "advantage": advantages.tolist()})
        if (n + 1) % 16 == 0:
            print(f"rollouts {n + 1}/{len(chosen)} {time.time() - start:.0f}s", flush=True)
    template.set_mode("train")
    with open(path, "w") as handle:
        for group in groups:
            handle.write(json.dumps(group, ensure_ascii=False) + "\n")
    return groups


def decision_index(labels, input_ids, language_id):
    """Position whose logits predict the language name: the first assistant token, ``language``."""
    first = int((labels != -100).nonzero()[0])
    assert int(input_ids[first]) == language_id
    return first


def accumulate(args, model, template, tokenizer, dataset, plan, groups, params):
    """Per prompt type and token part, the GRPO ascent direction of the sampled step (fp32, cuda:1)."""
    language_id = tokenizer.convert_tokens_to_ids("language")
    buffers = {(kind, part): [torch.zeros(p.shape, dtype=torch.float32, device="cuda:1") for _, p in params]
               for kind in PROMPT_TYPES for part in PARTS}
    stats = {kind: {"groups": 0, "signal_groups": 0, "contrast": 0, "tokens": {p: 0.0 for p in PARTS}}
             for kind in PROMPT_TYPES}
    template.set_mode("train")
    start = time.time()
    for n, group in enumerate(groups):
        stats[group["kind"]]["groups"] += 1
        stats[group["kind"]]["contrast"] += group["contrast"]
        advantages = np.array(group["advantage"])
        if np.abs(advantages).max() < 1e-6:
            continue
        stats[group["kind"]]["signal_groups"] += 1
        prompt = raw_prompt(dataset, plan, group["plan_index"])
        rows = [{"messages": [*prompt["messages"], {"role": "assistant", "content": o}],
                 "audios": prompt["audios"], "mix_wav": prompt["mix_wav"]} for o in group["outputs"]]
        batch = template.data_collator([template.encode(r) for r in rows])
        labels = batch.pop("labels")
        batch = to_device(batch)
        logits = model(**batch, use_cache=False).logits.float()
        logp = torch.log_softmax(logits[:, :-1], -1)
        target = batch["input_ids"][:, 1:]
        mask = (labels[:, 1:] != -100).to("cuda:0")
        token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1) * mask
        decision = torch.zeros_like(mask)
        for b in range(len(rows)):
            decision[b, decision_index(labels[b], batch["input_ids"][b].cpu(), language_id)] = True  # predicts name
        lengths = mask.sum(-1).clamp(min=1)
        weight = torch.tensor(advantages, device="cuda:0", dtype=torch.float32) / lengths
        objectives = {"decision": (weight * (token_logp * decision).sum(-1)).sum(),
                      "rest": (weight * (token_logp * (mask & ~decision)).sum(-1)).sum()}
        for part in PARTS:
            stats[group["kind"]]["tokens"][part] += float(((mask & ~decision) if part == "rest" else decision).sum())
            model.zero_grad(set_to_none=True)
            objectives[part].backward(retain_graph=part != PARTS[-1])
            for buf, (_, p) in zip(buffers[(group["kind"], part)], params):
                buf.add_(p.grad.to("cuda:1", torch.float32))
        model.zero_grad(set_to_none=True)
        if (n + 1) % 16 == 0:
            print(f"grads {n + 1}/{len(groups)} {time.time() - start:.0f}s", flush=True)
    return buffers, stats


def held_out(args):
    predictions = {json.loads(line)["id"] for line in open(Path(args.v12_run) / "predictions.jsonl")}
    by_family = {}
    for line in open(Path(args.v12_run) / "samples.jsonl"):
        s = json.loads(line)
        family = s["source"].split(":", 1)[1].split("/", 1)[0]
        if family in FAMILIES and s["id"] in predictions:
            by_family.setdefault(family, []).append(s)
    rng = random.Random(args.seed)
    return [s for f in sorted(by_family) for s in rng.sample(by_family[f], min(args.per_family, len(by_family[f])))]


def margin_batch(sample, template, resolver):
    from open_audio_llm.data.qwen3_asr import prepare_ts_clips

    import extract_states as ex  # same directory

    slots = {s["name"]: s["ref"] for s in sample["record"]["audio_slots"]}
    enroll = ex.load_ref(resolver, slots["enrollment"], ex.ENROLL_SECONDS).astype(np.float32) / 32768
    mix = ex.load_ref(resolver, slots["mixture"]).astype(np.float32) / 32768
    enroll, mix = prepare_ts_clips([enroll, mix], ex.SR)
    # The assistant text only has to start with ``language``: the margin is read where the name is predicted.
    row = {"messages": [{"role": "system", "content": sample["context"]}, {"role": "user", "content": "<audio>"},
                        {"role": "assistant", "content": "language None<asr_text>"}], "audios": [enroll], "mix_wav": mix}
    return template.data_collator([template.encode(row)])


def margin(model, batch, labels, ids, language_id):
    d = decision_index(labels[0], batch["input_ids"][0].cpu(), language_id)
    logp = torch.log_softmax(model(**batch, use_cache=False).logits[0, d].float(), -1)
    return logp[ids[0]] - torch.maximum(logp[ids[1]], logp[ids[2]])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts-per-type", type=int, default=64)
    parser.add_argument("--generations", type=int, default=8)  # train-grpo-v25.yaml --num_generations
    parser.add_argument("--per-family", type=int, default=40)
    parser.add_argument("--delta", action="append", default=None, help="NAME=CHECKPOINT; realised update CKPT - S")
    parser.add_argument("--start", default=START, help="checkpoint the step is taken from (S)")
    parser.add_argument("--v12-run", default=V12_RUN, help="AmphionEval v12 run of --start (held-out samples)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    deltas = dict(d.split("=", 1) for d in (args.delta or DEFAULT_DELTAS))

    from safetensors.torch import load_file

    from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

    model, template, tokenizer = load_model(args.start)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = ("embed_tokens", "conv2d1", "conv2d2", "conv2d3", "conv_out")  # --freeze_parameters of v25
    for n, p in params:
        if any(f in n for f in frozen):
            p.requires_grad_(False)
    params = [(n, p) for n, p in params if p.requires_grad]
    groups_of = [module_group(n) for n, _ in params]
    group_names = sorted(set(groups_of))

    dataset, plan, chosen = build_prompts(args)
    groups = rollouts(args, model, template, tokenizer, dataset, plan, chosen)
    buffers, stats = accumulate(args, model, template, tokenizer, dataset, plan, groups, params)
    norms = {f"{k}/{p}": float(sum(b.double().pow(2).sum() for b in bufs).sqrt()) for (k, p), bufs in buffers.items()}
    print(json.dumps(norms, indent=1), flush=True)
    scale = {key: 1.0 / max(norms[f"{key[0]}/{key[1]}"], 1e-30) for key in buffers}

    # Realised updates as bf16 deltas on cuda:1 (S's own weights are kept on CPU to restore after margins).
    start_state = {n: p.detach().to("cpu", copy=True) for n, p in params}
    delta_buffers = {}
    for name, ckpt in deltas.items():
        weights = {}
        for f in sorted(Path(ckpt).glob("*.safetensors")):
            weights.update(load_file(str(f)))
        delta_buffers[name] = [(weights[n].float() - start_state[n].float()).to("cuda:1", torch.bfloat16) for n, _ in params]
        del weights

    samples = held_out(args)
    resolver = LhotseCatalogAudioResolver("/workspace/data/registry/catalog", "/workspace/data/registry/roots.json")
    language_id = tokenizer.convert_tokens_to_ids("language")
    ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in (" None", " Chinese", " English")]
    components = [f"{k}/{p}" for k, p in buffers] + [f"delta/{n}" for n in delta_buffers]
    dots = np.zeros((len(samples), len(components), len(group_names)), np.float64)
    margins = {"S": np.zeros(len(samples))}
    group_index = [group_names.index(g) for g in groups_of]
    batches = []
    start = time.time()
    for s_i, sample in enumerate(samples):
        batch = margin_batch(sample, template, resolver)
        labels = batch.pop("labels")
        batch = to_device(batch)
        batches.append((batch, labels))
        model.zero_grad(set_to_none=True)
        m = margin(model, batch, labels, ids, language_id)
        m.backward()
        margins["S"][s_i] = float(m.detach())
        acc = torch.zeros(len(components), len(group_names), dtype=torch.float64, device="cuda:1")
        for j, ((n, p), gi) in enumerate(zip(params, group_index)):
            g = p.grad.to("cuda:1", torch.float32)
            for c, key in enumerate(buffers):
                acc[c, gi] += (g * buffers[key][j]).sum().double() * scale[key]
            for c, name in enumerate(delta_buffers, start=len(buffers)):
                acc[c, gi] += (g * delta_buffers[name][j].float()).sum().double()
        dots[s_i] = acc.cpu().numpy()
        if (s_i + 1) % 50 == 0:
            print(f"held-out {s_i + 1}/{len(samples)} {time.time() - start:.0f}s", flush=True)
    model.zero_grad(set_to_none=True)
    for name, delta in delta_buffers.items():  # actual margins of S + delta (bf16 as the checkpoint)
        with torch.no_grad():
            for (n, p), d in zip(params, delta):
                p.copy_((start_state[n].float() + d.float().cpu()).to(p.device, p.dtype))
            margins[name] = np.array([float(margin(model, b, lab, ids, language_id)) for b, lab in batches])
            for n, p in params:
                p.copy_(start_state[n].to(p.device))

    np.savez(out / "attribution.npz", dots=dots, components=np.array(components), module_groups=np.array(group_names),
             family=np.array([s["source"].split(":", 1)[1].split("/", 1)[0] for s in samples]),
             role=np.array([FAMILIES[s["source"].split(":", 1)[1].split("/", 1)[0]] for s in samples]),
             ids=np.array([s["id"] for s in samples]), **{f"margin_{k}": v for k, v in margins.items()})
    (out / "summary.json").write_text(json.dumps({"norms": norms, "stats": stats, "deltas": deltas,
                                                  "start": args.start, "data_config": DATA_CONFIG, "v12_run": args.v12_run,
                                                  "args": vars(args)}, indent=1, ensure_ascii=False))
    print("done", flush=True)


if __name__ == "__main__":
    main()
