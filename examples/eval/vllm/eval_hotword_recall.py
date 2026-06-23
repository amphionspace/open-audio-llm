#!/usr/bin/env python3
"""Evaluate phase-1/phase-2 hotword retrieve recall without main ASR scoring.

Usage:
    python examples/eval/vllm/eval_hotword_recall.py \
        --cuts /path/to/hotword_cuts.jsonl.gz \
        --port 8001 --model open-audio-llm \
        --top-k 10 --workers 64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "integrations" / "vllm"))

from compute_wer import get_normalizer, tokenize as _tokenize  # noqa: E402
from retrieve_hotwords import retrieve_hotwords  # noqa: E402
from test_vllm_inference import (  # noqa: E402
    _load_item_audios,
    build_base_url,
    call_vllm_api,
    load_lhotse_cuts,
    parse_asr_output,
)

logger = logging.getLogger(__name__)


def hotword_in_text(hotword: str, hyp: str, language: str) -> bool:
    norm = get_normalizer(language)
    hotword_tokens = _tokenize(hotword, norm)
    hyp_tokens = _tokenize(hyp, norm)
    if not hotword_tokens or len(hotword_tokens) > len(hyp_tokens):
        return False
    width = len(hotword_tokens)
    return any(hyp_tokens[i : i + width] == hotword_tokens for i in range(len(hyp_tokens) - width + 1))


def filter_valid_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        clean = item.get("clean") or {}
        if isinstance(clean, dict) and clean.get("pass") is False:
            continue
        if item.get("hotwords"):
            out.append(item)
    return out


def make_phase1_args(port: int, model: str, max_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        base_url=build_base_url(port),
        model=model,
        max_tokens=max_tokens,
        prompt=None,
        prompt_style="swift",
    )


def transcribe_one(item: dict[str, Any], args: SimpleNamespace) -> tuple[str, str]:
    audios_b64, _eval_dur = _load_item_audios(item, has_enrollment=False)
    item_no_hotwords = dict(item)
    item_no_hotwords["hotwords"] = []
    hyp_raw, _usage = call_vllm_api(
        audios_b64,
        item_no_hotwords,
        no_enrollment=True,
        no_language=True,
        no_hotwords=True,
        language_override=None,
        custom_prompt=args.prompt,
        task="asr",
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        prompt_style=args.prompt_style,
        spec_task="asr",
    )
    _detected, hyp_text = parse_asr_output(hyp_raw)
    return item["id"], hyp_text


def retrieve_one(work: tuple[str, str, list[str], list[str], int, str]) -> dict[str, Any]:
    utt_id, hyp, gt_hotwords, pool, top_k, language = work
    retrieved = retrieve_hotwords(hotwords=pool, ctc_text=hyp, top_k=top_k, language=language)
    retrieved_set = set(retrieved)
    gt_set = set(gt_hotwords)
    hit_count = sum(1 for hotword in gt_set if hotword in retrieved_set)
    return {
        "id": utt_id,
        "hyp": hyp,
        "gt": sorted(gt_set),
        "retrieved": list(retrieved),
        "missed": sorted(gt_set - retrieved_set),
        "gt_count": len(gt_set),
        "hit_count": hit_count,
    }


def run_phase1(items: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, str]:
    phase_args = make_phase1_args(args.port, args.model, args.max_tokens)
    hypotheses: dict[str, str] = {}
    errors = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(transcribe_one, item, phase_args): item for item in items}
        for idx, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                utt_id, hyp = future.result()
                hypotheses[utt_id] = hyp
            except Exception as exc:  # noqa: BLE001
                hypotheses[item["id"]] = ""
                errors += 1
                if errors <= 3:
                    logger.warning("phase1 error on %s: %s", item["id"], exc)
            elapsed = max(time.time() - t0, 1e-6)
            speed = idx / elapsed
            eta = (len(items) - idx) / speed if speed > 0 else 0
            sys.stdout.write(
                f"\r  [phase1] [{idx}/{len(items)}] {speed:.1f} utt/s errors={errors} ETA={eta:.0f}s"
            )
            sys.stdout.flush()
    print()
    logger.info("phase1 done in %.1fs (errors=%d)", time.time() - t0, errors)
    return hypotheses


def run_phase2(
    items: list[dict[str, Any]],
    hypotheses: dict[str, str],
    pool: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    work = [
        (item["id"], hypotheses.get(item["id"], ""), list(item["hotwords"]), pool, args.top_k, args.language)
        for item in items
    ]
    if args.retrieve_workers <= 1:
        return [retrieve_one(item) for item in work]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.retrieve_workers) as executor:
        futures = {executor.submit(retrieve_one, item): item for item in work}
        for idx, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if idx % 200 == 0 or idx == len(work):
                elapsed = max(time.time() - t0, 1e-6)
                speed = idx / elapsed
                eta = (len(work) - idx) / speed if speed > 0 else 0
                sys.stdout.write(f"\r  [phase2] [{idx}/{len(work)}] {speed:.1f} it/s ETA={eta:.0f}s")
                sys.stdout.flush()
    print()
    logger.info("phase2 done in %.1fs", time.time() - t0)
    return results


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    total_gt = sum(row["gt_count"] for row in results)
    total_hit = sum(row["hit_count"] for row in results)
    n_perfect = sum(1 for row in results if row["hit_count"] == row["gt_count"])
    n_eval = len(results)
    missed_cases = [row for row in results if row["hit_count"] != row["gt_count"]]
    missed_cases.sort(key=lambda row: row["hit_count"] / max(row["gt_count"], 1))
    missed_counter: Counter[str] = Counter()
    for row in missed_cases:
        missed_counter.update(row["missed"])

    micro_recall = total_hit / total_gt if total_gt else 0.0
    prrr = n_perfect / n_eval if n_eval else 0.0
    print("\n===== Hotword Recall Summary =====")
    print(f"Cuts file       : {args.cuts}")
    print(f"Top-K           : {args.top_k}")
    print(f"Tasks evaluated : {n_eval}")
    print(f"Micro recall    : {total_hit}/{total_gt} = {micro_recall:.2%}")
    print(f"PrRR (perfect)  : {n_perfect}/{n_eval} = {prrr:.2%}")
    for hotword, count in missed_counter.most_common(args.top_missed):
        print(f"  {count:>4}x  {hotword}")

    return {
        "summary": {
            "n_evaluated": n_eval,
            "micro_recall": micro_recall,
            "prrr": prrr,
            "total_gt": total_gt,
            "total_hit": total_hit,
            "n_perfect": n_perfect,
            "n_missed_cases": len(missed_cases),
        },
        "top_missed_hotwords": missed_counter.most_common(args.top_missed),
        "missed_cases": missed_cases[:500],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", required=True, help="Lhotse cuts manifest with hotwords")
    parser.add_argument("--port", type=int, default=8000, help="vLLM service port")
    parser.add_argument("--model", default="qwen3-omni", help="vLLM served-model-name")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=64, help="phase1 thread workers")
    parser.add_argument("--retrieve-workers", type=int, default=8, help="phase2 process workers")
    parser.add_argument("--language", default="auto", choices=["auto", "zh", "en"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--top-missed", type=int, default=30)
    return parser


def main() -> None:
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)
    args = build_parser().parse_args()
    items = load_lhotse_cuts(args.cuts, limit=args.num_samples)
    items = [{**item, "hotwords": item.get("hotwords") or []} for item in items]
    valid = filter_valid_items(items)
    logger.info("valid hotword items: %d / %d", len(valid), len(items))
    if not valid:
        raise SystemExit("no items with hotwords found")

    pool = sorted({hotword.strip() for item in valid for hotword in item["hotwords"] if hotword.strip()})
    logger.info("hotword pool: %d unique terms", len(pool))
    hypotheses = run_phase1(valid, args)
    results = run_phase2(valid, hypotheses, pool, args)
    payload = {
        "args": vars(args),
        "pool_size": len(pool),
        **summarize(results, args),
    }
    output_path = Path(args.output) if args.output else Path("tmp/hotword_recall_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDetail saved to {output_path}")


if __name__ == "__main__":
    main()
