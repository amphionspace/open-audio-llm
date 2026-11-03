import collections,gzip,json,random,sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,'.')
import numpy as np, soundfile as sf, torch, yaml
import select_checkpoints as sc
from audio_data_contract import resolve_split
from modelscope.pipelines import pipeline
from open_audio_llm.data import meeting_online as mo
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
data=yaml.safe_load(open('../configs/synth-v3-fidelity-aishell5.yaml'))
cfg=mo.OnlineMeetingConfig(**data['online_meeting'])
res=LhotseCatalogAudioResolver(data['catalog'],data['roots']); m=cfg.material
tracks={}
for path in resolve_split(res.catalog,m['dataset_id'],m['version'],m['split'],'audio_index',res.roots):
    for row in mo.read_material_rows([path]): tracks[row['cut_id']]=str(Path(res.roots[row['root_alias']])/row['relative_path'])
rows=list(mo.read_material_rows(resolve_split(res.catalog,m['dataset_id'],m['version'],m['split'],'meeting_material',res.roots)))
track_of=lambda r: tracks[f"{r['corpus']}/{r['meeting']}/{r['speaker']}"]
recs=[json.loads(l) for l in gzip.open('/workspace/data/datasets/meeting_selection_panel/versions/v1-20261008/records/alimeeting_dev.jsonl.gz','rt')]
model=pipeline(task='speaker-verification',model='/workspace/model/modelscope/models/iic--speech_eres2netv2_sv_zh-cn_16k-common/snapshots/master',device='cuda:0')
cache={}
def embed(v,i):
    k=(v.track,float(v.offsets[i]))
    if k not in cache:
        d=min(float(v.durations[i]),10.0)
        clip=sf.read(v.track,start=round(v.offsets[i]*16000),frames=round(d*16000),dtype='float32',always_2d=True)[0][:,0]
        e=np.asarray(model([clip],output_emb=True)['embs'])[0]; cache[k]=e/np.linalg.norm(e)
    return cache[k]
out={}
for name,corpus,excl,pred in (('aishell5','aishell5',cfg.exclude_meetings,'../tasks/12-aishell5-material/attempts/001/artifacts/checkpoints/step-start-aishell5/fidelity-r%d/attempts/001/artifacts/evaluation/predictions.jsonl'),('alimeeting','alimeeting',(),'../tasks/08-whole-conversations/attempts/001/artifacts/checkpoints/step-start-whole/fidelity-r%d/attempts/001/artifacts/evaluation/predictions.jsonl')):
    c=replace(cfg,corpus_weights={corpus:1.0},exclude_meetings=excl)
    mat=mo.load_material(rows,c,track_of)
    preds=collections.defaultdict(list)
    for r in (1,2,3):
        for l in open(pred%r):
            d=json.loads(l)
            if d['label']==f'syn_fill_near_{corpus}': preds[d['id'].split('/')[1]].append(d)
    W=[]
    for rec in recs:
        turns=sorted(rec['metadata']['turns'],key=lambda t:t['start'])
        L=round(rec['audio_slots'][0]['ref']['duration']*16000)/16000
        tpl=[[t['source_speaker'],t['start'],t['end'],len(mo.CHARS_PATTERN.sub('',t['text']))] for t in turns]
        _,pl=mo.plan_meeting(random.Random(f"fidelity-20261031:fill:{corpus}:{rec['id']}"),tpl,L,mat,c)
        per=collections.defaultdict(list)
        for b,d,s,v,i in pl:
            if d>=1.5 and len(per[s])<8: per[s].append((v,i))
        means={s:np.mean([embed(v,i) for v,i in x],axis=0) for s,x in per.items() if x}
        means={s:e/np.linalg.norm(e) for s,e in means.items()}
        names=sorted(means)
        if len(names)<2: continue
        pairs=[float(means[a]@means[b]) for i,a in enumerate(names) for b in names[i+1:]]
        ds=preds.get(rec['id'],[])
        nr=len({x[0] for x in sc.turns_of(ds[0]['reference'])})
        under=np.mean([len({x[0] for x in sc.turns_of(d.get('prediction'))})<nr for d in ds])
        W.append((nr,max(pairs),float(np.mean(pairs)),float(under)))
    W=np.array(W)
    for n in (2,3,4):
        w=W[W[:,0]==n]
        print(name,'voices',n,'windows',len(w),'max pair p50 %.2f p90 %.2f | mean pair %.2f | under %.1f%%'%(np.median(w[:,1]),np.percentile(w[:,1],90),w[:,2].mean(),100*w[:,3].mean()))
    w=W[W[:,0]>=3]
    for lo,hi in ((-1,.3),(.3,.4),(.4,.5),(.5,1.01)):
        s=w[(w[:,1]>=lo)&(w[:,1]<hi)]
        if len(s): print(' max pair',lo,hi,'windows',len(s),'under %.1f%%'%(100*s[:,3].mean()))
    out[name]=W.tolist()
json.dump(out,open(sys.argv[1],'w'))
