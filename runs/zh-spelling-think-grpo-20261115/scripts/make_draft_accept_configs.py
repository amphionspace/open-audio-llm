"""07-draft-accept: configs/eval-grpo-avg3-eagle3-k{2,3}/ from 04's configs/eval-grpo-avg3-mtp/.

Only the draft changes (06's EAGLE-3 head, method eagle3, k=2 or 3), plus GPUs, ports, served names, output roots and
attempts dirs, so the two lanes run side by side on one node and 04's numbers stay comparable.
usage: make_draft_accept_configs.py <draft dir>
"""
import shutil
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "configs/eval-grpo-avg3-mtp"
SUITES = ["context-correction", "context-correction-think"]
LANES = {3: {"asr": (0, 18980), "tsasr": (1, 18981)}, 2: {"asr": (2, 18982), "tsasr": (3, 18983)}}


def sub(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    draft = sys.argv[1]
    for k, lanes in LANES.items():
        name = f"grpo-avg3-eagle3-k{k}"
        dst = EXP / f"configs/eval-{name}"
        dst.mkdir(exist_ok=True)
        shutil.copy(SRC / "handoff.json", dst / "handoff.json")
        for m, (gpu, port) in lanes.items():
            s = (SRC / f"serve-{m}.yaml").read_text()
            s = sub(s, s.splitlines()[0], f"# 07-draft-accept：avg3 挂 06 训练的 EAGLE-3 草稿头（k={k}），"
                    f"由 scripts/make_draft_accept_configs.py 从 configs/eval-grpo-avg3-mtp/serve-{m}.yaml 生成，只换草稿、GPU、端口和记录目录。")
            s = sub(s, "  gpus:\n  - 5\n", f"  gpus:\n  - {gpu}\n")
            s = sub(s, f"    --port: {18970 if m == 'asr' else 18971}\n", f"    --port: {port}\n")
            s = sub(s, f"--served-model-name: grpo-avg3-mtp-{m}\n", f"--served-model-name: {name}-{m}\n")
            s = sub(s, "      method: eagle\n      model: /workspace/workspace/mingdong/repo/oal-joint-mtp/runs/"
                    "joint-mtp-spec-20261107/shared/draft-v16avg3\n      num_speculative_tokens: 2\n",
                    f"      method: eagle3\n      model: {draft}\n      num_speculative_tokens: {k}\n")
            s = sub(s, f"../../tasks/04-mtp-borrow/local-serve-{m}/attempts",
                    f"../../tasks/07-draft-accept/serve-k{k}-{m}/attempts")
            (dst / f"serve-{m}.yaml").write_text(s)
        for suite in SUITES:
            s = (SRC / f"after-training-{suite}.yaml").read_text()
            think = "-think" if suite.endswith("think") else ""
            s = sub(s, f"model_name: grpo-avg3-mtp{think}\n", f"model_name: {name}{think}\n")
            s = sub(s, "server_url: http://127.0.0.1:18970\n", f"server_url: http://127.0.0.1:{lanes['asr'][1]}\n")
            s = sub(s, "tsasr_server_url: http://127.0.0.1:18971\n", f"tsasr_server_url: http://127.0.0.1:{lanes['tsasr'][1]}\n")
            s = sub(s, "served_model: grpo-avg3-mtp-asr\n", f"served_model: {name}-asr\n")
            s = sub(s, "tsasr_served_model: grpo-avg3-mtp-tsasr\n", f"tsasr_served_model: {name}-tsasr\n")
            s = sub(s, f"spelling-think-grpo-20261115-grpo-avg3-mtp{think}\n", f"spelling-think-grpo-20261115-{name}{think}\n")
            (dst / f"after-training-{suite}.yaml").write_text(s)
        print(dst)


if __name__ == "__main__":
    main()
