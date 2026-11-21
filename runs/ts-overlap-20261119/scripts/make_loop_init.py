"""tasks/11: build the looped start model from a source checkpoint (no training).

Hard-links every file of SRC into OUT, adds ``amphion_audio_loop = {start, end}`` to config.json's audio config,
writes identity-initialised copies of audio layers [START, END) to model-loop.safetensors (open_audio_llm
tsasr/audio_loop.py: zero attention-output and fc2 weights and biases), and an index over both weight files.
The result computes exactly what SRC computes; training then moves only the copies.
usage: make_loop_init.py SRC OUT START END     (needs the feat/ts-audio-loop source on PYTHONPATH)
"""
import json
import os
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from open_audio_llm.tsasr.audio_loop import LOOP_KEY, init_loop_state

TOWER = "thinker.audio_tower."


def main():
    src, out, start, end = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    weights = sorted(src.glob("*.safetensors"))
    if [p.name for p in weights] != ["model.safetensors"] or (src / "model.safetensors.index.json").exists():
        raise ValueError(f"{src} must hold one unsharded model.safetensors")
    out.mkdir(parents=True)
    for path in src.iterdir():
        if path.name != "config.json":
            os.link(path, out / path.name)
    config = json.loads((src / "config.json").read_text())
    audio = config["thinker_config"]["audio_config"]
    if LOOP_KEY in audio:
        raise ValueError(f"{src} already has {LOOP_KEY}")
    audio[LOOP_KEY] = {"start": start, "end": end}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    with safe_open(str(src / "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        tower = {k[len(TOWER):]: f.get_tensor(k) for k in keys if k.startswith(f"{TOWER}layers.")}
    loop = init_loop_state(tower, start, end)
    save_file({k: v.contiguous() for k, v in loop.items()}, str(out / "model-loop.safetensors"), metadata={"format": "pt"})
    weight_map = {k: "model.safetensors" for k in keys} | {k: "model-loop.safetensors" for k in loop}
    total = (src / "model.safetensors").stat().st_size + (out / "model-loop.safetensors").stat().st_size
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1) + "\n")
    print(json.dumps({"out": str(out), "loop": {"start": start, "end": end}, "loop_tensors": len(loop),
                      "loop_params": sum(v.numel() for v in loop.values())}))


if __name__ == "__main__":
    main()
