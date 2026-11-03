import collections,gzip,json,random,re,sys
from dataclasses import replace
sys.path.insert(0,'.')
import yaml, numpy as np, select_checkpoints as sc
from audio_data_contract import resolve_split
from open_audio_llm.data import meeting_online as mo
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
data=yaml.safe_load(open('../configs/synth-v3-fidelity-aishell5.yaml'))
cfg=mo.OnlineMeetingConfig(**data['online_meeting'])
res=LhotseCatalogAudioResolver(data['catalog'],data['roots']); m=cfg.material
rows=list(mo.read_material_rows(resolve_split(res.catalog,m['dataset_id'],m['version'],m['split'],'meeting_material',res.roots)))
material=mo.load_material(rows,replace(cfg,corpus_weights={'aishell5':1.0}),lambda r:'t')
sizes={c[0].key[1]:len(c) for c in material['aishell5']}
print('sizes',collections.Counter(sizes.values()))
recs=[json.loads(l) for l in gzip.open('/workspace/data/datasets/meeting_selection_panel/versions/v1-20261008/records/alimeeting_dev.jsonl.gz','rt')]
plan={}
for r in recs:
    length=round(r['audio_slots'][0]['ref']['duration']*16000)
    turns=sorted(r['metadata']['turns'],key=lambda t:t['start'])
    tpl=[[t['source_speaker'],t['start'],t['end'],len(mo.CHARS_PATTERN.sub('',t['text']))] for t in turns]
    rng=random.Random(f"fidelity-20261031:fill:aishell5:{r['id']}")
    _,placed=mo.plan_meeting(rng,tpl,length/16000,material,cfg)
    keys={}
    for begin,d,s,v,i in placed: keys[s]=v.key
    plan[r['id']]=keys
res_=collections.defaultdict(lambda:[0,0,0])
for r_ in (1,2,3):
    for l in open(f'../tasks/12-aishell5-material/attempts/001/artifacts/checkpoints/step-start-aishell5/fidelity-r{r_}/attempts/001/artifacts/evaluation/predictions.jsonl'):
        d=json.loads(l)
        k=d['label'].split('_')[2]; rid=d['id'].split('/')[1]; keys=plan[rid]
        people=[x[2] for x in keys.values()]; convs={x[1] for x in keys.values()}
        ref,hyp=sc.turns_of(d['reference']),sc.turns_of(d.get('prediction'))
        nr,nh=len({x[0] for x in ref}),len({x[0] for x in hyp})
        assert nr<=len(keys)
        kind=(k,len(keys),len(convs),'dup' if len(set(people))<len(people) else 'uniq')
        c=res_[kind]; c[0]+=1; c[1]+=nh<nr; c[2]+=nh>nr
for k,v in sorted(res_.items()): print(k,'decodes',v[0],'under%',round(100*v[1]/v[0],1),'over%',round(100*v[2]/v[0],1))
print('example',list(plan.items())[:3])
