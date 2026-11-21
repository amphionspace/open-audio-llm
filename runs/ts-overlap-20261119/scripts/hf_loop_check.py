"""tasks/11 HF check (the training and scoring tower): load models with open_audio_llm's load_qwen3_asr and compare
audio-tower outputs on the same random TS input (5 s enrollment + 10 s mixture, SEP between).
Pass: loop-init == 11.17 exactly, trueloop == deep exactly, trueloop != 11.17.
usage: hf_loop_check.py OUT_JSON NAME=MODEL_DIR ...      (feat/ts-audio-loop source on PYTHONPATH)
"""
import itertools
import json
import sys

import torch

from open_audio_llm.tsasr.sep_token import load_qwen3_asr


def main():
    out, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    torch.manual_seed(0)
    features = torch.randn(128, 1500)
    outputs, info = {}, {}
    for name, path in specs:
        model = load_qwen3_asr(path).to("cuda").eval()
        tower = model.thinker.audio_tower
        info[name] = {"layers": len(tower.layers), "loop_layers": len(getattr(tower, "loop_layers", []))}
        with torch.no_grad():
            outputs[name] = tower(features.to("cuda", torch.bfloat16), feature_lens=torch.tensor([1500]),
                                  enroll_n_frames=500).last_hidden_state.float().cpu()
        del model
        torch.cuda.empty_cache()
    result = {"models": info, "pairs": {}}
    for a, b in itertools.combinations(outputs, 2):
        diff = (outputs[a] - outputs[b]).abs()
        result["pairs"][f"{a}|{b}"] = {"identical": bool(torch.equal(outputs[a], outputs[b])),
                                       "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}
    json.dump(result, open(out, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
