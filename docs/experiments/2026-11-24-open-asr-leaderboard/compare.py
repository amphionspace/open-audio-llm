"""Compare our leaderboard-style scores with the Open ASR Leaderboard (v1.16.2, 2026-11-20).

Usage: compare.py <board-public7.json> <reports-dir> > reports/compare.md
reports/<model>__<part>/open-asr-leaderboard.json come from ae open-asr-leaderboard score.
Board numbers of Qwen/Qwen3-ASR-1.7B-hf were read from the Space (English main table and the
per-language dataset breakdown).
"""
import json
import sys
from pathlib import Path

BOARD_QWEN3 = {
    "ami_cleaned": 8.31, "Earnings22-Cleaned-AA-chunked": 5.84, "gigaspeech_cleaned": 7.22,
    "spgispeech": 2.57, "urgent2024": 9.93, "Voice Arena Monsoon": 3.5, "voxpopuli_cleaned_aa": 2.85,
    "fleurs_zh": 2.72, "aishell1_zh": 1.51, "fleurs_de": 3.16, "mcv_de": 4.6,
    "fleurs_fr": 3.88, "mcv_fr": 7.85, "mls_fr": 5.17, "fleurs_it": 2.15, "mcv_it": 4.62, "mls_it": 9.5,
    "fleurs_es": 2.61, "mcv_es": 4.04, "mls_es": 4.46, "fleurs_pt": 3.69, "mls_pt": 8.63,
    "fleurs_nl": 6.42, "mcv_nl": 5.46, "mls_nl": 11.79,
}
BOARD_LANG = {"zh": 2.12, "de": 3.88, "fr": 5.63, "it": 5.42, "es": 3.7, "pt": 6.16, "nl": 7.89}
# Chinese column of the other models in the multilingual table (CER, FLEURS + AISHELL-1).
BOARD_ZH = {"microsoft/azure-speech-09-2026": 1.89, "sprag/symphony": 1.94, "Qwen/Qwen3-ASR-1.7B-hf": 2.12,
            "modulate/multilingual": 2.6, "Qwen/Qwen3-ASR-0.6B-hf": 2.66, "yoob/yoob-asr-pro-v1": 2.79,
            "openai/gpt-transcribe": 3.04, "assemblyai/universal-3-5-pro": 3.09,
            "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B": 3.16, "openai/whisper-large-v3": 4.78}


def load(reports, job):
    path = reports / job / "open-asr-leaderboard.json"
    return json.loads(path.read_text()) if path.exists() else None


def main():
    board = json.loads(Path(sys.argv[1]).read_text())
    reports = Path(sys.argv[2])
    ours = {}
    for model in ("qwen3-asr-1.7b", "amphionasr-1117"):
        merged = {"datasets": {}, "languages": {}}
        for part in ("en", "ml"):
            result = load(reports, f"{model}__{part}")
            if result:
                merged["datasets"].update(result["datasets"])
                merged["languages"].update(result["languages"])
                if part == "en":
                    merged["english"] = result["english_public_average"]
        ours[model] = merged
    q, a = ours["qwen3-asr-1.7b"], ours["amphionasr-1117"]
    out = ["# Open ASR Leaderboard 同口径对照", "",
           "## 逐数据集（错误率 %；中文为 CER）", "",
           "| 数据集 | 榜单 Qwen3-ASR-1.7B-hf | 我们测 Qwen3-ASR-1.7B | 差 | 我们测 11.17 | 11.17 − Qwen3 |",
           "| --- | --- | --- | --- | --- | --- |"]
    for name, ref in BOARD_QWEN3.items():
        qv = q["datasets"].get(name, {}).get("error_rate")
        av = a["datasets"].get(name, {}).get("error_rate")
        fmt = lambda v: "—" if v is None else f"{v:.2f}"
        diff = "—" if qv is None else f"{qv - ref:+.2f}"
        gap = "—" if qv is None or av is None else f"{av - qv:+.2f}"
        out.append(f"| {name} | {ref:.2f} | {fmt(qv)} | {diff} | {fmt(av)} | {gap} |")
    out += ["", "## 语言分与英文平均", "",
            "| 项目 | 榜单 Qwen3 | 我们测 Qwen3 | 我们测 11.17 |", "| --- | --- | --- | --- |"]
    board_q = next(r for r in board if r["model"] == "Qwen/Qwen3-ASR-1.7B-hf")
    eq, ea = q.get("english", {}).get("score"), a.get("english", {}).get("score")
    out.append(f"| 英文 7 个公开集平均 | {board_q['public7']:.2f} | {eq} | {ea} |")
    for lang, ref in BOARD_LANG.items():
        out.append(f"| {lang} | {ref:.2f} | {q['languages'].get(lang, {}).get('score')} | {a['languages'].get(lang, {}).get('score')} |")
    if ea is not None:
        # Rank 11.17 among board entries by the same 7-set public average (private sets excluded).
        shift = (eq - board_q["public7"]) if eq is not None else 0.0
        adjusted = ea - shift
        better = [r for r in board if r["public7"] < ea]
        better_adj = [r for r in board if r["public7"] < adjusted]
        out += ["", "## 英文公开集排名（按 7 个公开集平均，不含私有集）", "",
                f"- 11.17 原值 {ea:.2f}：排第 {len(better) + 1}（共 {len(board)} 个有完整 7 集成绩的条目），"
                f"开源模型中第 {sum(r['open'] for r in better) + 1}。",
                f"- 按我们与榜单在 Qwen3 上的差（{shift:+.2f}）校正后 {adjusted:.2f}：排第 {len(better_adj) + 1}，"
                f"开源模型中第 {sum(r['open'] for r in better_adj) + 1}。"]
    zh = a["languages"].get("zh", {}).get("score")
    if zh is not None:
        ranks = sorted(list(BOARD_ZH.items()) + [("AmphionASR 11.17（我们测）", zh)], key=lambda x: x[1])
        out += ["", "## 中文列（FLEURS + AISHELL-1，CER）", "", "| 模型 | 中文 |", "| --- | --- |"]
        out += [f"| {m} | {v:.2f} |" for m, v in ranks]
    print("\n".join(out))


if __name__ == "__main__":
    main()
