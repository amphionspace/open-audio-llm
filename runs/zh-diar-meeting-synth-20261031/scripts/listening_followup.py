import collections,glob,gzip,json,re,sys
import numpy as np, soundfile as sf
A='../tasks/13-aishell5-listening/attempts/002/artifacts'
key=json.load(open(f'{A}/key.json')); ans=json.load(open(f'{A}/review/answers.json'))['answers']
P='../tasks/12-aishell5-material/attempts/001/artifacts/checkpoints/step-start-aishell5/fidelity-r1/attempts/001/artifacts/evaluation/predictions.jsonl'
pred={}
for l in open(P):
    d=json.loads(l)
    if d['label']=='syn_fill_near_aishell5': pred[d['id'].split('/')[1]]=d
LINE=re.compile(r'\[S(\d+)\]\[(\d+\.?\d*)-(\d+\.?\d*)\]\s*(.*)')
import torch
from modelscope.pipelines import pipeline
sv=pipeline(task='speaker-verification',model='/workspace/model/modelscope/models/iic--speech_eres2netv2_sv_zh-cn_16k-common/snapshots/master',device='cpu')
def emb(x):
    e=np.asarray(sv([x],output_emb=True)['embs'])[0]; return e/np.linalg.norm(e)
AUDIO='/workspace/data/datasets/meeting_synth_fidelity/versions/v6-20261101/syn_fill_near_aishell5'
for k in key:
    ref=[(int(m[1]),float(m[2]),float(m[3]),m[4]) for m in LINE.finditer(pred[k['window']]['reference'])]
    out=[]; es=[]
    for side in 'ab':
        a,b=k[side]['span']; s=k[side]['speaker']
        text=[t[3] for t in ref if t[0]==s and t[1]<b and t[2]>a]
        out.append(side+':'+' / '.join(text)[:70])
        x,_=sf.read(f"{A}/sources/{k['id'][1:]}-{side}.wav",dtype='float32'); es.append(emb(x))
    print(k['id'],k['group'][:6],ans[k['id']]['choice'],'sv %.2f'%float(es[0]@es[1]),k['a']['material'].split('/')[-1],k['b']['material'].split('/')[-1]); print('   ',out[0]); print('   ',out[1])
