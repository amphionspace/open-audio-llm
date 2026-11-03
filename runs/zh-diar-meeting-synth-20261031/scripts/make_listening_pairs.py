"""Build the A/B listening spec: two labelled voices the start model merged, plus same- and different-person controls.

Run from scripts/ with shared/code-control/src and amphiondata/src on PYTHONPATH. Clips are solo stretches of the
rendered near-field window, level untouched. Item ids and order hide the group; key.json stays outside the served page.
"""
import collections, gzip, hashlib, json, random, re
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from amphiondata.review import build
from audio_data_contract import resolve_split
from open_audio_llm.data import meeting_online as mo
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

ATTEMPT = Path("../tasks/13-aishell5-listening/attempts/002/artifacts").resolve()
PREDICTIONS = "../tasks/12-aishell5-material/attempts/001/artifacts/checkpoints/step-start-aishell5/fidelity-r%d/attempts/001/artifacts/evaluation/predictions.jsonl"
AUDIO = "/workspace/data/datasets/meeting_synth_fidelity/versions/v6-20261101/syn_fill_near_aishell5"
LINE = re.compile(r"\[S(\d+)\]\[(\d+\.?\d*)-(\d+\.?\d*)\]\s*(.*)")
MIN_CLIP_SECONDS = 2.0  # shorter stretches rarely carry enough voice to compare
MAX_CLIP_SECONDS = 4.0  # keeps one item under about ten seconds
MIN_SAME_PERSON_GAP_SECONDS = 10.0  # same-person controls come from separate moments of the window
MERGED_DECODES = 2  # a pair counts as merged when at least this many of the three decodes merge it


def turns(text):
    return [(int(m[1]), float(m[2]), float(m[3])) for m in LINE.finditer(text or "")]


def overlap(a, b, x, y):
    return max(0.0, min(b, y) - max(a, x))


def solo_stretches(ref, speaker):
    """Stretches where only this labelled speaker is talking, longest first."""
    grid = np.arange(0, max(t[2] for t in ref) + 0.05, 0.05)
    own, other = np.zeros(len(grid), bool), np.zeros(len(grid), bool)
    for s, x, y in ref:
        mask = (grid >= x) & (grid < y)
        if s == speaker:
            own |= mask
        else:
            other |= mask
    solo = own & ~other
    edges = np.flatnonzero(np.diff(np.r_[False, solo, False]))
    spans = [(float(grid[a]), float(grid[b - 1]) + 0.05) for a, b in zip(edges[::2], edges[1::2])]
    return sorted((s for s in spans if s[1] - s[0] >= MIN_CLIP_SECONDS), key=lambda s: s[0] - s[1])


def centre(span):
    length = min(span[1] - span[0], MAX_CLIP_SECONDS)
    start = (span[0] + span[1] - length) / 2
    return round(start, 2), round(start + length, 2)


def mapping(ref, hyp):
    """Labelled speaker -> decoded speaker with the most shared time."""
    shared = collections.defaultdict(collections.Counter)
    for s, a, b in ref:
        for h, x, y in hyp:
            shared[s][h] += overlap(a, b, x, y)
    return {s: c.most_common(1)[0][0] if c and c.most_common(1)[0][1] > 0 else None for s, c in shared.items()}


data = yaml.safe_load(open("../configs/synth-v3-fidelity-aishell5.yaml"))
cfg = mo.OnlineMeetingConfig(**data["online_meeting"])
res = LhotseCatalogAudioResolver(data["catalog"], data["roots"])
m = cfg.material
rows = list(mo.read_material_rows(resolve_split(res.catalog, m["dataset_id"], m["version"], m["split"], "meeting_material", res.roots)))
records = [json.loads(l) for l in gzip.open("/workspace/data/datasets/meeting_selection_panel/versions/v1-20261008/records/alimeeting_dev.jsonl.gz", "rt")]
material = mo.load_material(rows, replace(cfg, corpus_weights={"aishell5": 1.0}), lambda r: "t")
preds = collections.defaultdict(list)
for r in (1, 2, 3):
    for line in open(PREDICTIONS % r):
        d = json.loads(line)
        if d["label"] == "syn_fill_near_aishell5":
            preds[d["id"].split("/")[1]].append(d)

windows = []
for rec in records:
    ds = preds[rec["id"]]
    ref = turns(ds[0]["reference"])
    speakers = sorted({t[0] for t in ref})
    if len(speakers) != 3:
        continue
    template = sorted(rec["metadata"]["turns"], key=lambda t: t["start"])
    length = round(rec["audio_slots"][0]["ref"]["duration"] * 16000) / 16000
    tpl = [[t["source_speaker"], t["start"], t["end"], len(mo.CHARS_PATTERN.sub("", t["text"]))] for t in template]
    _, placed = mo.plan_meeting(random.Random(f"fidelity-20261031:fill:aishell5:{rec['id']}"), tpl, length, material, cfg)
    time_shared = collections.defaultdict(collections.Counter)
    for s, a, b in ref:
        for begin, duration, _, voice, _ in placed:
            time_shared[s]["/".join(map(str, voice.key))] += overlap(a, b, begin, begin + duration)
    maps = [mapping(ref, turns(d.get("prediction"))) for d in ds]
    merged = collections.Counter()
    for mp in maps:
        for i, a in enumerate(speakers):
            for b in speakers[i + 1:]:
                merged[a, b] += mp[a] is not None and mp[a] == mp[b]
    windows.append(dict(
        window=rec["id"], ref=ref, speakers=speakers, merged=merged, decoded=[len({t[0] for t in turns(d.get("prediction"))}) for d in ds],
        material={s: c.most_common(1)[0][0] for s, c in time_shared.items()}, solo={s: solo_stretches(ref, s) for s in speakers},
    ))
    windows[-1]["meeting"] = {p.split("/")[1] for p in windows[-1]["material"].values()}


def pair_item(w, group, a, b):
    return dict(group=group, window=w["window"], decoded_voices=w["decoded"], a=dict(speaker=a, span=centre(w["solo"][a][0]), material=w["material"][a]),
                b=dict(speaker=b, span=centre(w["solo"][b][0]), material=w["material"][b]))


items, used, meetings = [], set(), set()
for w in windows:  # merged by the model, labelled as different people
    pairs = [p for p, n in w["merged"].most_common() if n >= MERGED_DECODES and w["solo"][p[0]] and w["solo"][p[1]]]
    if sum(n < 3 for n in w["decoded"]) == 3 and pairs and not w["meeting"] & meetings and sum(i["group"] == "merged" for i in items) < 8:
        items.append(pair_item(w, "merged", *pairs[0]))
        used.add(w["window"])
        meetings |= w["meeting"]
control_meetings = set()
for w in windows:  # the model tells these apart in every decode
    if all(n == 3 for n in w["decoded"]) and not w["meeting"] & control_meetings and sum(i["group"] == "kept_apart" for i in items) < 2:
        pairs = [(a, b) for i, a in enumerate(w["speakers"]) for b in w["speakers"][i + 1:] if w["merged"][a, b] == 0 and w["solo"][a] and w["solo"][b]]
        if pairs:
            items.append(pair_item(w, "kept_apart", *pairs[0]))
            used.add(w["window"])
            control_meetings |= w["meeting"]
control_meetings = set()
for w in windows:  # one labelled speaker at two moments
    if w["window"] in used or w["meeting"] & control_meetings or sum(i["group"] == "same_person" for i in items) >= 2:
        continue
    for s in w["speakers"]:
        pairs = [(x, y) for i, x in enumerate(w["solo"][s]) for y in w["solo"][s][i + 1:] if abs(x[0] - y[0]) >= MIN_SAME_PERSON_GAP_SECONDS]
        if pairs:
            x, y = pairs[0]
            items.append(dict(group="same_person", window=w["window"], decoded_voices=w["decoded"], a=dict(speaker=s, span=centre(x), material=w["material"][s]), b=dict(speaker=s, span=centre(y), material=w["material"][s])))
            control_meetings |= w["meeting"]
            break
print(collections.Counter(i["group"] for i in items))

random.Random("aishell5-listening-pairs-20261101").shuffle(items)
spec = dict(
    id="aishell5-voice-pairs-20261101", title="这两段是同一个人吗",
    intro=["每题约 8 秒：先放 A，一声提示音，再放 B。只判断 A、B 是不是同一个人，不用管内容。",
           "空格播放，按 1、2、3 作答，J 下一题。选完自动保存。部分片段音量偏小，未调整过音量，请把音量调大。"],
    choices=["同一个人", "不同的人", "听不出"], items=[],
)
key = []
for number, item in enumerate(items, 1):
    clips = []
    for side in "ab":
        start, end = item[side]["span"]
        audio, rate = sf.read(f"{AUDIO}/{item['window']}.wav", start=round(start * 16000), stop=round(end * 16000), dtype="float32")
        path = ATTEMPT / "sources" / f"{number:02d}-{side}.wav"
        sf.write(path, audio, rate, subtype="PCM_16")
        item[side].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rms_dbfs=round(20 * float(np.log10(np.sqrt(np.mean(audio**2)) + 1e-9)), 1))
        clips.append(dict(path=str(path)))
    spec["items"].append(dict(id=f"p{number:02d}", title="A 和 B 是同一个人吗？", audio=clips))
    key.append(dict(id=f"p{number:02d}", **item))
    print(f"p{number:02d}", item["group"], item["window"][-12:], item["decoded_voices"], item["a"]["span"], item["a"]["material"], item["a"]["rms_dbfs"], "|", item["b"]["span"], item["b"]["material"], item["b"]["rms_dbfs"])
(ATTEMPT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1))
build(spec, ATTEMPT / "review")
total = sum(sf.info(p).duration for p in (ATTEMPT / "review" / "audio").glob("*.wav"))
print("items", len(spec["items"]), "total seconds", round(total), "longest", round(max(sf.info(p).duration for p in (ATTEMPT / "review" / "audio").glob("*.wav")), 1))
