"""On-the-fly target-speaker extraction scenes from the v12 generator (audio-data-contract 3f9c7ca, recipe
v4-20261022) with speech pools switched to train splits (KeSpeech without its 250-speaker cap) or, for validation,
dev splits. Each item: 5 s enrollment (the split's enrollment condition), mixture, and the target image (target with
its room response and gain, the mixture's scaling; no interferers, no noise). Target-present splits without a
nonlinear mix effect. Noise and RIR files are the archived v12 ones."""
import json, random, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch, yaml
GEN = Path(__file__).resolve().parent.parent / "gen"
sys.path.insert(0, str(GEN)); sys.path.insert(0, str(GEN.parent))
import build, mixer  # noqa: E402
from stems_lib import stems  # noqa: E402
SRC = Path("/workspace/data/datasets/ts_asr_benchmark/source/v4-20261022")
SR = mixer.SAMPLE_RATE
DEV_SOURCES = {"studio": [{"dataset": "aishell", "version": "legacy-20260804", "split": "dev"}],
               "phone": [{"dataset": "kespeech", "version": "legacy-20260804", "split": "dev"},
                         {"dataset": "magicdata", "version": "legacy-20260804", "split": "dev"}],
               "read": [{"dataset": "aishell", "version": "legacy-20260804", "split": "dev"}]}


FINAL_SOURCES = [{"dataset": "kespeech", "version": "train-punc-v1-20261102-final-v1-20261120", "split": "train"},
                 {"dataset": "aishell2", "version": "train-punc-v1-20261102-final-v1-20261120", "split": "train"}]
_CJK = "\u2e80-\u9fff\u3000-\u303f\uff00-\uffef"


def final_text(text):
    """Training-text spec rule 5 (zh-en-transcription.md): one space between English words, none between Chinese
    characters or between Chinese and English (final-v1 did not execute this rule); nothing else is changed."""
    import re
    t = re.sub(r"\s+", " ", (text or "").strip())
    return re.sub(f"(?<=[{_CJK}]) | (?=[{_CJK}])", "", t)


def _load_source_final(source, catalog, roots, rng):
    """build.load_source with final_text instead of deleting every space (that would join English words)."""
    import re
    from collections import defaultdict
    from audio_data_contract import resolve_split
    paths = {role: resolve_split(catalog, source["dataset"], source["version"], source["split"], role, roots)
             for role in ("recordings", "supervisions")}
    audio = {r["id"]: r["sources"][0]["source"] for path in paths["recordings"] for r in build._rows(path)}
    by_speaker = defaultdict(list)
    for row in (row for path in paths["supervisions"] for row in build._rows(path)):
        text = final_text(row.get("text"))
        if not text or row.get("start", 0) != 0 or row["recording_id"] not in audio:
            continue
        speaker = f"{source['dataset']}:{row['speaker']}"
        by_speaker[speaker].append(mixer.Utt(id=f"{source['dataset']}:{row['id']}", speaker=speaker, text=text,
                                             duration=float(row["duration"]), path=audio[row["recording_id"]],
                                             corpus=source["dataset"], gender=row.get("gender")))
    return {s: by_speaker[s] for s in sorted(by_speaker)}


def setup(mode):
    """mode: 'dev' (dev splits, legacy text; audio only), 'train' (train splits of the original sources, legacy text;
    tasks 19-23 before 2026-11-23), 'train_final' (every pool = KeSpeech + AISHELL-2 final-v1 train text)."""
    from audio_data_contract import load_catalog
    recipe = yaml.safe_load((GEN / "recipe.yaml").read_text())
    if mode == "train_final":
        build.load_source = _load_source_final
    for name, pool in recipe["pools"].items():
        if not pool.get("sources"):
            continue
        if mode == "dev":
            pool["sources"] = DEV_SOURCES[name]
        elif mode == "train_final":
            pool["sources"] = [dict(s) for s in FINAL_SOURCES]
        else:
            for s in pool["sources"]:
                s["split"] = "train"; s.pop("max_speakers", None)
    roots = json.loads(Path("/workspace/data/registry/roots.json").read_text())
    catalog = load_catalog(Path("/workspace/data/registry/catalog"))
    pools = build.build_pools(recipe, catalog, roots)
    pools["__enroll_files__"] = build.enrollment_files(recipe, roots)
    pools["__noise__"] = {k: build.noise_files(spec, catalog, roots) for k, spec in recipe["noise"].items()}
    extra, rir_pools = build.prepared_noise(SRC / "noise")
    pools["__noise__"].update(extra); pools["__rir_pools__"] = rir_pools
    pools["__rirs__"] = sorted(str(p) for p in (SRC / "rir").glob("*.wav"))
    pools["__rir_pools__"]["dns5_sim"] = list(pools["__rirs__"])
    pools["__rirs__"] = sorted(p for v in pools["__rir_pools__"].values() for p in v)
    build.G.update(recipe=recipe, pools=pools)
    splits = [s for s in build.expand_splits(recipe)
              if not s["name"].endswith("-neg") and s["params"].get("kind") not in ("neg", "nonspeech")
              and not s["params"].get("hard") and not s["params"].get("mix_effect")]
    n_spk = {k: (len(v["speakers"]), len(v["short"])) for k, v in pools.items() if not k.startswith("__")}
    return recipe, pools, splits, n_spk


def scene(rng, pools, splits):
    while True:
        split = rng.choice(splits)
        try:
            plan = mixer.plan_scene(rng, split["params"], pools, None)
        except (mixer.PlanError, IndexError, ValueError):  # e.g. a pool without short-utterance speakers
            continue
        if plan["params"].get("mix_effect") or not any(t.role == "target" for t in plan["tracks"]):
            continue
        t, o, n = stems(plan, build, mixer)
        load = lambda utt: build.read_audio(utt.path)
        enroll = mixer.render_enrollment(plan, load, lambda p: sf.read(p, dtype="float32")[0],
                                         lambda kind, path: build._cached_noise(path))
        c = 5 * SR
        enroll = np.pad(enroll[:c], (0, max(0, c - len(enroll)))).astype(np.float32)
        return split["name"], enroll, (t + o + n).astype(np.float32), t.astype(np.float32)


class Scenes(torch.utils.data.IterableDataset):
    def __init__(self, pools, splits, seed, max_seconds=16.0):
        self.pools, self.splits, self.seed, self.max = pools, splits, seed, int(max_seconds * SR)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(f"{self.seed}:{info.id if info else 0}")
        while True:
            _, e, m, t = scene(rng, self.pools, self.splits)
            if len(m) > self.max:  # random crop keeps training batches bounded
                s = rng.randrange(len(m) - self.max + 1); m, t = m[s:s + self.max], t[s:s + self.max]
            yield e, m, t


def collate(items):
    n = max(len(m) for _, m, _ in items)
    e = torch.stack([torch.from_numpy(x) for x, _, _ in items])
    m = torch.zeros(len(items), n); t = torch.zeros(len(items), n); lens = torch.tensor([len(x) for _, x, _ in items])
    for i, (_, a, b) in enumerate(items):
        m[i, :len(a)] = torch.from_numpy(a); t[i, :len(b)] = torch.from_numpy(b)
    return e, m, t, lens
