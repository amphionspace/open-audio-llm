"""tasks/29: speaker embeddings of the simulator's training speakers, for its hard (similar-voice) negatives.

The v12 generator picks a hard negative's enrollment among the 5 pool speakers closest to an interferer
(mixer.plan_scene, `hard: true`), by the cosine of the benchmark's speaker model (WeSpeaker ECAPA-TDNN512-LM, ONNX;
adc scripts/ts_benchmark/speaker_embeddings.py). The benchmark embedded its test speakers; training needs the same
for the train pools of tse_data.setup(POOLS). Same rule as the benchmark: unit-norm mean of up to 3 utterances per
speaker (whole utterances), drawn with the recipe seed. Embedder: the benchmark model through adc-meeting-fa
voice_check.Embedder (as meeting_train_plan.py).
usage (oal-train python, PYTHONPATH as job_meeting_train.sh): train_speaker_embeddings.py OUT.npz [WORKERS] [POOLS]"""
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np

SIM = Path("/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/"
           "ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag")  # the simulator copy tasks 23-28 trained with
sys.path.insert(0, str(SIM / "tse"))
sys.path.insert(0, "/workspace/workspace/mingdong/repo/adc-meeting-fa/scripts/ts_real")
EMBED_PY = Path("/workspace/workspace/mingdong/miniconda3/envs/k2/bin/python")
EMBED_MODEL = Path("/workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx")
UTTERANCES_PER_SPEAKER = 3  # the benchmark's rule


def work(job):
    import tse_data
    import voice_check as voice
    rank, items = job
    emb = voice.Embedder(EMBED_PY, EMBED_MODEL, threads=2); out = []
    for k, (speaker, paths) in enumerate(items):
        vs = [emb(tse_data.build.read_audio(p)) for p in paths]  # mono 16 kHz (AISHELL-3 is 44.1 kHz)
        m = np.mean(vs, 0); out.append((speaker, m / np.linalg.norm(m)))
        if rank == 0 and k % 500 == 0:
            print(f"worker 0: {k}/{len(items)}", flush=True)
    emb.close()
    return out


def main():
    out, workers = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 24
    pools_mode = sys.argv[3] if len(sys.argv) > 3 else "train_final2"
    import tse_data
    recipe, pools, _, n_spk = tse_data.setup(pools_mode)
    print("pools", n_spk, flush=True)
    rng = random.Random(recipe["seed"])
    todo = []
    for speaker, utts in sorted(pools["mixed"]["speakers"].items()):
        chosen = rng.sample(utts, min(UTTERANCES_PER_SPEAKER, len(utts)))
        todo.append((speaker, [u.path for u in chosen]))
    jobs = [(r, todo[r::workers]) for r in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = [x for part in pool.map(work, jobs) for x in part]
    rows.sort()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, keys=np.array([k for k, _ in rows]), vectors=np.stack([v for _, v in rows]).astype(np.float32))
    print(f"{len(rows)} speakers -> {out}", flush=True)


if __name__ == "__main__":
    main()
