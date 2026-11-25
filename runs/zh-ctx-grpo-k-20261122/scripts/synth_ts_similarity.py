"""README "数据增量诊断" static check: how close the enrolled voice is to the target in the synthetic TS training items,
against the real TS eval where the merged models newly miss. ECAPA cosine (wespeaker ECAPA-TDNN512-LM, the front end of
audio-data-contract scripts/ts_benchmark/speaker_embeddings.py, the model behind the eval labels' ``target_sim``) between
enrollment and mixture of every present item of context_correction_synth_zh ``train_ts`` (one TTS voice per mixture, so
mixture = target speech) and ``train_ts_twins`` (its mixture slot is the target clip; the loader renders the scene from
metadata.scene, so this is also enrollment vs target speech: the present twins reuse train_ts's 2,917 target clips).
Prints quantiles next to ``labels.target_sim`` of ts-asr-first-other@v3 present items (enrollment vs the target's solo
speech in the far-field mixture). Writes tasks/08-datainc/synth-ts-similarity.json.
usage: python -I synth_ts_similarity.py   (env with onnxruntime, torchaudio, soundfile)
"""
import glob
import gzip
import json
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL = "/workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx"
SYNTH = "/workspace/data/datasets/context_correction_synth_zh/versions"
SPLITS = (("train-v1-20261114", "train_ts"), ("ts-twins-v1-20261121", "train_ts_twins"))
EVAL_RUNS = "/workspace/data/ae-runs/ctx-grpo-20261120-delivery-r1117/runs/run-*"
OUT = Path(__file__).resolve().parents[1] / "tasks/08-datainc/synth-ts-similarity.json"
ROOTS = json.loads(Path("/workspace/data/registry/roots.json").read_text())
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def make_embedder():
    import onnxruntime as ort
    import torch
    from torchaudio.compliance import kaldi
    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def embed(wave):
        feats = kaldi.fbank(torch.from_numpy(wave).unsqueeze(0) * 32768, num_mel_bins=80, frame_length=25,
                            frame_shift=10, dither=0, sample_frequency=16000, window_type="hamming", use_energy=False)
        feats = feats - feats.mean(0, keepdim=True)
        e = session.run(None, {name: feats.unsqueeze(0).numpy()})[0].reshape(-1)
        return e / np.linalg.norm(e)
    return embed


def summary(values):
    v = np.asarray(values)
    return {"n": int(v.size), **{f"q{int(q * 100)}": round(float(np.quantile(v, q)), 3) for q in QUANTILES},
            "share_below_0.5": round(float((v < 0.5).mean()), 3)}


def main():
    embed = make_embedder()
    result = {}
    for version, split in SPLITS:
        index = {r["cut_id"]: r for r in map(json.loads, gzip.open(f"{SYNTH}/{version}/audio-index.jsonl.gz", "rt"))}
        cache = {}

        def vec(ref):
            r = index[ref["cut_id"]]
            if ref["cut_id"] not in cache:
                wave, sr = sf.read(Path(ROOTS[r.get("root_alias") or "aidc_data"]) / r["relative_path"], dtype="float32")
                assert sr == 16000, sr
                cache[ref["cut_id"]] = embed(wave if wave.ndim == 1 else wave.mean(1))
            return cache[ref["cut_id"]]
        sims = []
        for line in gzip.open(f"{SYNTH}/{version}/records/{split}.jsonl.gz", "rt"):
            record = json.loads(line)
            if not record["target"].strip():
                continue
            slots = {a["purpose"]: a["ref"] for a in record["audio_slots"]}
            sims.append(float(vec(slots["enrollment"]) @ vec(slots["mixture"])))
        result[split] = summary(sims)
        print(split, result[split], flush=True)
    run = [r for r in sorted(glob.glob(EVAL_RUNS))
           if "ts-asr-first-other" in (json.load(open(r + "/run.json"))["config"].get("eval_set") or "")][-1]
    labels = [s["record"]["labels"].get("target_sim") for s in map(json.loads, open(run + "/samples.jsonl"))
              if s["reference"].strip()]
    result["first-other target_sim"] = summary([x for x in labels if x is not None])
    print("first-other target_sim", result["first-other target_sim"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
