"""Real-speech TS configs for one checkpoint (question 8): configs/eval-tsreal-<name>/ with one TS server and two
suites of the 11.06 release's gate, think off.

Why: in question 7, J left more context-correction TS items empty than template, nearly all of them items whose TTS
clone sounds little like the real speaker enrolled (ECAPA cosine < 0.5). These two sets say whether J also misses
real target speech:
- gate-v12: open-audio-llm/ts-asr-benchmark@v12-20261022, 26,750 items, mixtures of real test-set speech (aishell,
  aishell2/3, kespeech, magicdata) and real recordings (AliMeeting, AISHELL-4, RAMC, business feedback);
- gate-first-other: open-audio-llm/ts-asr-first-other@v3-20261105, 9,030 items, AliMeeting / AISHELL-4 meetings,
  the target present or absent among other talkers.
Each suite copies the release run's recorded config (as 08's make_gate_configs.py: runs/zh-spelling-think-grpo-20261115
on oal-think-exp) and changes only the model name, the TS server and the output directory. The server is the scene-TS
one (make_scene_ts_configs.write_serve: vLLM 0.32, code-v1, the checkpoint's own mtp/, greedy). Run with ae 0.14
(new TS prompt) for both models, so template and J are compared on the same client; the release numbers are not
the baseline.
usage: make_ts_real_configs.py <name> <checkpoint> <ts gpu> <ts port>
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_scene_ts_configs import write_serve  # noqa: E402

RELEASE_RUNS = Path("/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/runs")
GATES = {
    "v12": "run-20261105T182838Z-fd77aed9",  # 中文 TS-ASR 测试集 ts-asr-benchmark@v12-20261022
    "first-other": "run-20261105T185559Z-9b91bc4f",  # 跟错人 ts-asr-first-other@v3-20261105
}


def main():
    name, checkpoint, gpu, port = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    model = f"tsreal-{name}"
    dst = write_serve(model, checkpoint, gpu, port,
                      "# 由 scripts/make_ts_real_configs.py 从 configs/eval-clue-target-implicit-noexplain/serve-tsasr.yaml"
                      " 生成：只换模型（草稿随之换成它自带的 mtp/）、GPU、端口、服务名和记录目录。\n")
    for key, run in GATES.items():
        config = json.loads((RELEASE_RUNS / run / "run.json").read_text())["config"]
        assert config.get("eval_set") and not config.get("server_url"), key
        config.pop("data_config", None)  # resolved copy of the eval set; the name resolves it again
        config.update(model_name=f"ctx-clue-target-{model}", tsasr_server_url=f"http://127.0.0.1:{port}",
                      tsasr_served_model=f"{model}-tsasr",
                      output_root=f"/workspace/data/ae-runs/ctx-clue-target-20261117-{model}",
                      wandb={"entity": "1016097967-amphion", "project": "open-audio-llm"}, publish={"enabled": False})
        assert not config["parameters"].get("think"), key
        header = f"# 真人语音 TS（think 关）：交付版 {run} 的配置，只换模型、服务与输出目录（scripts/make_ts_real_configs.py）。\n"
        (dst / f"after-training-gate-{key}.yaml").write_text(header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print(dst)


if __name__ == "__main__":
    main()
