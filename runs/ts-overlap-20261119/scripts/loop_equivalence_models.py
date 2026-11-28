"""tasks/11 serving check: two debug models that must compute the same thing (no training).

A (OUT/trueloop): SRC + amphion_audio_loop {START, END} whose copies are the unmodified layers, i.e. a real
  weight-tied loop: layers[:END] + layers[START:END] + layers[END:].
B (OUT/deep): the same computation without the loop feature: a plain tower with encoder_layers + (END - START)
  layers stored in that order.
pack-loop serving A and the unmodified pack-mtp serving B must give identical outputs; that checks the loop's
config, weight names, loading and layer order in the served model.
usage: loop_equivalence_models.py SRC OUT START END
"""
import json
import os
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

LAYER = re.compile(r"^thinker\.audio_tower\.layers\.(\d+)\.(.+)$")


def link_rest(src, out):
    out.mkdir(parents=True)
    for path in src.iterdir():
        if path.name not in ("config.json", "model.safetensors"):
            os.link(path, out / path.name)


def main():
    src, out, start, end = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    config = json.loads((src / "config.json").read_text())
    n = config["thinker_config"]["audio_config"]["encoder_layers"]
    with safe_open(str(src / "model.safetensors"), framework="pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}

    loop_dir = out / "trueloop"
    link_rest(src, loop_dir)
    os.link(src / "model.safetensors", loop_dir / "model.safetensors")
    loop = {}
    for key, value in state.items():
        m = LAYER.match(key)
        if m and start <= int(m[1]) < end:
            loop[f"thinker.audio_tower.loop_layers.{int(m[1]) - start}.{m[2]}"] = value
    save_file(loop, str(loop_dir / "model-loop.safetensors"), metadata={"format": "pt"})
    weight_map = {k: "model.safetensors" for k in state} | {k: "model-loop.safetensors" for k in loop}
    (loop_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}) + "\n")
    c = json.loads(json.dumps(config))
    c["thinker_config"]["audio_config"]["amphion_audio_loop"] = {"start": start, "end": end}
    (loop_dir / "config.json").write_text(json.dumps(c, indent=2) + "\n")

    deep_dir = out / "deep"
    link_rest(src, deep_dir)
    order = list(range(end)) + list(range(start, end)) + list(range(end, n))
    deep = {k: v for k, v in state.items() if not LAYER.match(k)}
    for position, source in enumerate(order):
        prefix = f"thinker.audio_tower.layers.{source}."
        for key, value in state.items():
            if key.startswith(prefix):
                deep[f"thinker.audio_tower.layers.{position}.{key[len(prefix):]}"] = value.clone()
    save_file({k: v.contiguous() for k, v in deep.items()}, str(deep_dir / "model.safetensors"), metadata={"format": "pt"})
    c = json.loads(json.dumps(config))
    c["thinker_config"]["audio_config"]["encoder_layers"] = len(order)
    c["thinker_config"]["audio_config"]["num_hidden_layers"] = len(order)
    (deep_dir / "config.json").write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps({"trueloop": str(loop_dir), "deep": str(deep_dir), "deep_order": order}))


if __name__ == "__main__":
    main()
