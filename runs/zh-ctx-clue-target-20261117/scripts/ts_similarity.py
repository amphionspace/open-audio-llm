"""ECAPA cosine between the enrollment and the mixture of every TS-mode item of the context-correction eval v2.

Why (question 7): J left more of these items empty than template. Each item is a real speaker's enrollment and a
Qwen3-TTS clone of that speaker alone, so the cosine says how much the clone sounds like the person enrolled.
Embedder: WeSpeaker ECAPA-TDNN512-LM (ONNX), the same front end as audio-data-contract
scripts/ts_benchmark/speaker_embeddings.py. Needs onnxruntime, torchaudio and soundfile (the amphion-tse-open-pvad
env has them); reads the registered audio of the run's sample records.
usage: python -I ts_similarity.py RUN_DIR OUT_JSON   (RUN_DIR: any context-correction v2 run, e.g. J's)
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL = "/workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx"
ROOTS = json.loads(Path("/workspace/data/registry/roots.json").read_text())


def make_embedder():
    import onnxruntime as ort
    import torch
    from torchaudio.compliance import kaldi

    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def embed(w):
        feats = kaldi.fbank(torch.from_numpy(w).unsqueeze(0) * 32768, num_mel_bins=80, frame_length=25, frame_shift=10,
                            dither=0, sample_frequency=16000, window_type="hamming", use_energy=False)
        feats = feats - feats.mean(0, keepdim=True)
        e = session.run(None, {name: feats.unsqueeze(0).numpy()})[0].reshape(-1)
        return e / np.linalg.norm(e)

    return embed


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    index, cache, embed = {}, {}, make_embedder()

    def vec(ref):
        key = (ref["dataset_id"], ref["version"])
        if key not in index:
            f = Path(f"/workspace/data/datasets/{key[0]}/versions/{key[1]}/audio-index.jsonl.gz")
            index[key] = {r["cut_id"]: r for r in map(json.loads, gzip.open(f, "rt"))}
        row = index[key][ref["cut_id"]]
        path = Path(ROOTS[row.get("root_alias") or "aidc_data"]) / row["relative_path"]
        if path not in cache:
            w, sr = sf.read(path, dtype="float32")
            assert sr == 16000, path
            cache[path] = embed(w if w.ndim == 1 else w.mean(1))
        return cache[path]

    sims = {}
    for line in open(run / "samples.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s.get("task") == "ts_asr":
            slots = {a["purpose"]: a["ref"] for a in s["record"]["audio_slots"]}
            sims[s["id"]] = round(float(vec(slots["enrollment"]) @ vec(slots["mixture"])), 4)
    out.write_text(json.dumps(sims, ensure_ascii=False, indent=0) + "\n")
    print(len(sims), "items, mean", round(float(np.mean(list(sims.values()))), 3))


if __name__ == "__main__":
    main()
