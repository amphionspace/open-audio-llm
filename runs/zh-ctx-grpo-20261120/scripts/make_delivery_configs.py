"""Delivery evaluation configs (README "交付评估"): the 27 suites of the 11.17 delivery package's own validation
(AmphionEval runs of whitebox-merge-v18a-es116inc on 2026-11-17, the package's model weights), for one checkpoint on two
server pairs, so avg3 and 11.17 run on the same node, code (code-v3) and client (ae 0.14) and compare pairwise.

Each suite copies the 11.17 run's recorded config (as make_gate_configs.py does with the 11.06 runs) and changes only
the model name, the servers and the output root (ctx-grpo-20261120-delivery-<name>). The suites are split into two
lists of about equal work (pair a takes asrreg, pair b hotword-bench); each pair's servers come from make_eval_configs.py.
usage: make_delivery_configs.py <name> <checkpoint> <gpu a asr> <gpu a ts> <port a> <gpu b asr> <gpu b ts> <port b>
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
RUNS_1117 = Path("/workspace/data/ae-runs/whitebox-merge-v18a-es116inc/runs")
# (key, run id suffix, pair). Keys name the suite; pair a / b balances the two ASR and TS lanes.
SUITES = [
    ("v12", "961b82b3", "a"), ("es-asr", "917e5efb", "a"), ("ts-es-v1", "8692ce9c", "b"), ("babble4", "3ef2545a", "a"),
    ("babble3", "ef3a9d87", "a"), ("customer-ts", "746d0006", "a"), ("customer-asr", "f566d145", "b"),
    ("ts-hotwords", "e4675684", "b"), ("ts-en", "8319f035", "a"), ("first-other", "644cd966", "a"),
    ("asr-benchmark", "7f90ddc5", "a"), ("hotword-bench", "b7aff370", "b"), ("police", "1febeaee", "a"),
    ("police-single", "0a023b5f", "a"), ("crosslingual", "bb51fe81", "a"), ("scene-asr", "2f8539df", "a"),
    ("crosslingual-en", "95a755ec", "a"), ("scene-ts", "47fe0ed3", "a"), ("ts-es-v2", "c9da9e0c", "b"),
    ("ctx-v1", "8a7c2040", "b"), ("babble-es", "ccec188c", "b"), ("hotword-bench-es", "d030821d", "b"),
    ("chime4", "0659a8a2", "b"), ("customer-hotwords", "50ee3911", "b"), ("badcase", "7909a758", "b"),
    ("ts-hotwords-es", "22a47c06", "b"), ("crosslingual-es", "1966a0be", "b"),
]


def run_dir(suffix):
    found = [p for p in RUNS_1117.glob(f"run-20261117T*-{suffix}")]
    assert len(found) == 1, suffix
    return found[0]


def main():
    name, checkpoint = sys.argv[1], sys.argv[2]
    pairs = {"a": (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])), "b": (int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]))}
    for pair, (asr_gpu, ts_gpu, port) in pairs.items():
        model = f"delivery-{name}-{pair}"
        subprocess.run([sys.executable, str(HERE / "scripts/make_eval_configs.py"), model, checkpoint, str(asr_gpu), str(ts_gpu), str(port)],
                       check=True, stdout=subprocess.DEVNULL)
        out = HERE / "configs" / f"eval-{model}"
        serve = {route: yaml.safe_load((out / f"serve-{route}.yaml").read_text())["task"]["arguments"] for route in ("asr", "tsasr")}
        for key, suffix, p in SUITES:
            if p != pair:
                continue
            run = run_dir(suffix)
            config = json.loads((run / "run.json").read_text())["config"]
            if config.get("eval_set"):
                config.pop("data_config", None)  # resolved copy of the eval set; the name resolves it again
            for route, url_key, model_key in (("asr", "server_url", "served_model"), ("tsasr", "tsasr_server_url", "tsasr_served_model")):
                if config.get(url_key):
                    config[url_key] = f"http://127.0.0.1:{serve[route]['--port']}"
                    config[model_key] = serve[route]["--served-model-name"]
            assert not (config.get("parameters") or {}).get("think"), key
            config.update(model_name=f"ctx-grpo-delivery-{name}", output_root=f"/workspace/data/ae-runs/ctx-grpo-20261120-delivery-{name}",
                          wandb={"entity": "1016097967-amphion", "project": "open-audio-llm"}, publish={"enabled": False})
            header = f"# 交付评估：11.17 交付包验证 {run.name} 的配置，只换模型、服务与输出目录（scripts/make_delivery_configs.py）。\n"
            (out / f"after-training-delivery-{key}.yaml").write_text(header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        print(out)


if __name__ == "__main__":
    main()
