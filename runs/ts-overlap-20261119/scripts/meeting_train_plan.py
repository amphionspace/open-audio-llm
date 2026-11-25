"""tasks/28: the plan of the deployment-shaped meeting TS stream, built from the real-meeting TS training records of
11.17's recipe (ts_real_train alimeeting / aishell4 / ramc, final text, the recipe's versions and excluded records).

Task 27 located the meeting false alarms: with similar voices (ECAPA >= 0.4) an enrollment from the same recording is
accepted 3-6x as often as an equally similar one from another recording, because in training an enrollment from the
same recording is almost always the target (same-recording negatives are 1.6% of TS, with random, mostly dissimilar
people). This plan lets the training stream (ft_meeting.py) send every window of those records the way speaker-enroll
does, to the window's own target or to another participant of the same recording, picked by voice similarity.

Per corpus, from the recipe's sources (REPLAY_CONFIG entries of ts_real_train with these splits):
- windows: each target-present record (mixture span, target, final punctuated text, the window's utterances), the
  text split into the target's utterances (punctuation stripped it equals the utterances' text in every record);
- spans: every enrollment span whose recording is the mixture's (the recipe's same-recording enrollments, positive
  and negative records), per (recording, speaker); spans overlapping a window are not offered for it; every other
  participant's spans are offered as negatives (the stream keeps those of people silent in the audio it sends: the
  whole window, or one of the target's utterances, where people who talk elsewhere in the window are silent too);
- ECAPA (the benchmark's speaker model, voice_check.Embedder) of each span's first 5 s on channel 0 (what the client
  sends); a speaker's profile is the normalized mean of its spans; per window and offered negative span the cosine with
  the profile of each talker of the window (task 27's impostor_sim is the max over talkers);
- AliMeeting: each speaker's headset file (Train_Ali_near) for the cross-channel enrollments.

    python meeting_train_plan.py REPLAY_CONFIG OUT_DIR [WORKERS]   (oal-train python)
"""
import collections, gzip, json, multiprocessing as mp, sys, unicodedata
from pathlib import Path
import numpy as np, soundfile as sf, yaml
from audio_data_contract import load_catalog, resolve_split

sys.path.insert(0, "/workspace/workspace/mingdong/repo/adc-meeting-fa/scripts/ts_real")
import voice_check as voice  # noqa: E402

CORPORA = ("alimeeting", "aishell4", "ramc")
SR, CROP = 16000, 5  # the client sends the first 5 s of an enrollment
NEAR = Path("/workspace/data/datasets/alimeeting/source/openslr-119-local-20260826/Train_Ali_near/audio_dir")
EMBED_PY = Path("/workspace/workspace/mingdong/miniconda3/envs/k2/bin/python")
EMBED_MODEL = Path("/workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx")
ROOTS = "/workspace/data/registry/roots.json"


def strip(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LN")


def pieces(target, lengths):
    """The punctuated target cut after each utterance's characters (trailing punctuation stays with the piece)."""
    out, i, n = [], 0, len(target)
    for k, length in enumerate(lengths):
        j, seen = i, 0
        while j < n and seen < length:
            seen += unicodedata.category(target[j])[0] in "LN"; j += 1
        if k < len(lengths) - 1:
            while j < n and unicodedata.category(target[j])[0] not in "LN":
                j += 1
        else:
            j = n
        out.append(target[i:j].strip()); i = j
    return out


def read_records(cfg):
    cat, roots = load_catalog(cfg["catalog"]), json.loads(Path(ROOTS).read_text())
    for src in cfg["train"]:
        if src["dataset_id"] != "ts_real_train" or src["split"] not in CORPORA:
            continue
        ver, split = src["version"], src["split"]
        excluded = set(Path(src["exclude_records"]).read_text().split()) if src.get("exclude_records") else set()
        index = {}
        for p in resolve_split(cat, "ts_real_train", ver, split, "audio_index", roots):
            for line in gzip.open(p, "rt"):
                row = json.loads(line); index[row["cut_id"]] = (str(Path(roots[row["root_alias"]]) / row["relative_path"]), row["channels"])
        n = 0
        for p in resolve_split(cat, "ts_real_train", ver, split, "records", roots):
            for line in gzip.open(p, "rt"):
                r = json.loads(line)
                if r["id"] not in excluded:
                    n += 1; yield split, ver, r, index
        print(f"{split} {ver}: {n} records ({len(excluded)} excluded)", flush=True)


def embed_spans(job):
    rank, items = job
    emb = voice.Embedder(EMBED_PY, EMBED_MODEL, threads=4); out = {}
    for k, (path, channels, start, dur) in enumerate(items):
        n = int(min(dur, CROP) * SR)
        w, sr = sf.read(path, start=int(round(start * SR)), frames=n, dtype="float32", always_2d=True)
        assert sr == SR
        out[(path, start, dur)] = emb(w[:, 0])
        if rank == 0 and k % 2000 == 0:
            print(f"embed worker 0: {k}/{len(items)}", flush=True)
    emb.close()
    return out


def main():
    cfg_path, out = Path(sys.argv[1]), Path(sys.argv[2]); workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    cfg = yaml.safe_load(cfg_path.read_text()); out.mkdir(parents=True, exist_ok=True)
    windows = {c: [] for c in CORPORA}; spans = collections.defaultdict(set); sources = []
    for split, ver, r, index in read_records(cfg):
        if (split, ver) not in sources: sources.append((split, ver))
        e, m = r["audio_slots"][0]["ref"], r["audio_slots"][1]["ref"]
        path, channels = index[m["cut_id"]]
        rec = (split, path)
        if e["cut_id"] == m["cut_id"]:
            spans[rec].add((r["metadata"]["target_speaker"], round(e["start"], 3), round(e["duration"], 3), channels))
        if not r["labels"]["target_present"]:
            continue
        target = r["metadata"]["target_speaker"]
        utts = [(u["speaker"], round(u["start"], 3), round(u["duration"], 3)) for u in r["metadata"]["window_utts"]]
        own = [u for u in r["metadata"]["window_utts"] if u["speaker"] == target]
        lengths = [len(strip(u["text"])) for u in own]
        assert strip(r["target"]).lower() == "".join(strip(u["text"]) for u in own).lower(), r["id"]
        windows[split].append({"id": r["id"], "path": path, "channels": channels, "start": round(m["start"], 3),
                               "duration": round(m["duration"], 3), "target": target, "text": r["target"], "utts": utts,
                               "pieces": pieces(r["target"], lengths)})
    todo = sorted({(rec[1], ch, s, d) for rec, ss in spans.items() for _, s, d, ch in ss})
    print(f"{len(todo)} enrollment spans to embed with {workers} workers", flush=True)
    chunks = [(i, todo[i::workers]) for i in range(workers)]
    E = {}
    with mp.Pool(workers) as pool:
        for part in pool.imap_unordered(embed_spans, chunks):
            E.update(part)
    plan = {"sources": sources, "recordings": {}, "windows": {}}
    stats = {}
    for corpus in CORPORA:
        recs, prof, vecs = {}, {}, []
        for (split, path), ss in spans.items():
            if split != corpus:
                continue
            by = collections.defaultdict(list)
            for spk, s, d, _ in sorted(ss):
                v = E[(path, s, d)]; vecs.append(v); by[spk].append((s, d, len(vecs) - 1))
            near = {}
            if corpus == "alimeeting":
                rid = Path(path).stem.rsplit("_", 1)[0]  # R1019_M1963_MS116 -> R1019_M1963
                for spk in by:
                    hits = sorted(NEAR.glob(f"{rid}_*_SPK{spk.split(':')[1]}.wav"))
                    if len(hits) == 1: near[spk] = str(hits[0])
            recs[path] = {"spans": dict(by), "near": near}
            for spk, items in by.items():
                p = np.mean([vecs[i] for _, _, i in items], 0); prof[(path, spk)] = p / np.linalg.norm(p)
        V = np.stack(vecs) if vecs else np.zeros((0, 1))
        ws, silent, cropped = [], [], []
        for w in windows[corpus]:
            rec = recs.get(w["path"])
            if rec is None or w["target"] not in rec["spans"]:
                continue
            lo, hi = w["start"], w["start"] + w["duration"]
            free = lambda s, d: s + d <= lo or s >= hi
            talkers = sorted({u[0] for u in w["utts"]})
            pos = [i for s, d, i in rec["spans"][w["target"]] if free(s, d)]
            if not pos or any((w["path"], t) not in prof for t in talkers):
                continue
            neg = []
            for spk, items in rec["spans"].items():
                if spk == w["target"]:
                    continue
                for s, d, i in items:
                    if free(s, d):
                        neg.append((spk, i, [round(float(V[i] @ prof[(w["path"], t)]), 4) for t in talkers]))
            w = dict(w, talkers=talkers, pos=pos, neg=neg); ws.append(w)
            silent += [max(x[2]) for x in neg if x[0] not in talkers]  # negatives for the whole window
            cropped += [x[2][talkers.index(w["target"])] for x in neg]  # vs the target, for a crop of its utterance
        np.save(out / f"emb-{corpus}.npy", V.astype(np.float32))
        plan["recordings"][corpus] = recs; plan["windows"][corpus] = ws
        def summary(x):
            x = np.array(x)
            return {"offers": len(x), "sim_quantiles": [round(float(q), 3) for q in np.quantile(x, [0.1, 0.5, 0.9])] if len(x) else [],
                    "sim_bins": {b: int(((x >= lo_) & (x < hi_)).sum()) for b, lo_, hi_ in
                                 (("<0.2", -1, 0.2), ("0.2-0.3", 0.2, 0.3), ("0.3-0.4", 0.3, 0.4), ("0.4-0.5", 0.4, 0.5), (">=0.5", 0.5, 2))}}
        stats[corpus] = {"recordings": len(recs), "windows": len(ws), "spans": len(vecs),
                         "near_speakers": sum(len(r["near"]) for r in recs.values()), "speakers": sum(len(r["spans"]) for r in recs.values()),
                         "windows_with_silent_participant": sum(any(x[0] not in w["talkers"] for x in w["neg"]) for w in ws),
                         "windows_with_other_participant": sum(bool(w["neg"]) for w in ws),
                         "whole_window_negatives": summary(silent), "utterance_crop_negatives_vs_target": summary(cropped)}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False))
    (out / "stats.json").write_text(json.dumps({"replay_config": str(cfg_path), "sources": sources, "corpora": stats}, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
