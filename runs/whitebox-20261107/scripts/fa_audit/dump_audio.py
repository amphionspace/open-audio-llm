"""Dump 5 s enrollment + mixture for negatives of the hotspot splits (plus a few others) into one npz per split."""
import json, sys, numpy as np
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
from audio_data_contract import AudioRef
SR=16000
RUN='/workspace/data/ae-runs/whitebox-merge-sum-full/runs/run-20261107T054755Z-b3a6683a'
splits=sys.argv[1].split(',')
out=sys.argv[2]
res=LhotseCatalogAudioResolver('/workspace/data/registry/catalog','/workspace/data/registry/roots.json')
def load_ref(ref, seconds=None):
    cut=res.get_cut(AudioRef(**{k: ref[k] for k in ("dataset_id","version","split","cut_id")}))
    if cut.sampling_rate!=SR: cut=cut.resample(SR)
    a=np.clip(cut.load_audio().mean(0)*32768,-32768,32767).astype(np.int16)
    if seconds is not None:
        n=int(seconds*SR); a=np.pad(a[:n],(0,max(0,n-len(a))))
    return a
for l in open(f'{RUN}/samples.jsonl'):
    s=json.loads(l); sp=s['id'].split(':')[0]
    if sp not in splits or (s["reference"].strip() and sp!="solo-short-pos"): continue
    slots={x['name']:x['ref'] for x in s['record']['audio_slots']}
    e=load_ref(slots['enrollment'],5.0); m=load_ref(slots['mixture'])
    np.savez(f"{out}/{s['id'].replace(':','__')}.npz", enroll=e, mix=m)
print('done')
