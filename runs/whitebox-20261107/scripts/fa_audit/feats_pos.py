"""WeSpeaker ECAPA-TDNN512-LM (same model the benchmark used for hard negatives) + energy VAD/levels."""
import json, glob, os, numpy as np, torch, onnxruntime as ort
from torchaudio.compliance import kaldi
SR=16000
RUN='/workspace/data/ae-runs/whitebox-merge-sum-full/runs/run-20261107T054755Z-b3a6683a'
MODEL="/workspace/model/wespeaker-ecapa-tdnn512-LM/voxceleb_ECAPA512_LM.onnx"
so=ort.SessionOptions(); so.intra_op_num_threads=16
sess=ort.InferenceSession(MODEL,sess_options=so,providers=["CPUExecutionProvider"]); nm=sess.get_inputs()[0].name
def embed(w):
    if len(w)<int(0.3*SR): w=np.pad(w,(0,int(0.3*SR)-len(w)))
    f=kaldi.fbank(torch.from_numpy(w).unsqueeze(0)*32768,num_mel_bins=80,frame_length=25,frame_shift=10,dither=0,
                  sample_frequency=16000,window_type="hamming",use_energy=False)
    f=f-f.mean(0,keepdim=True)
    e=sess.run(None,{nm:f.unsqueeze(0).numpy()})[0].reshape(-1); return e/np.linalg.norm(e)
def frames_db(w):
    hop=160; win=400; n=max(1,(len(w)-win)//hop+1)
    idx=np.arange(win)[None]+hop*np.arange(n)[:,None]
    x=w[np.minimum(idx,len(w)-1)]
    return 10*np.log10((x**2).mean(1)+1e-12)
def vad(w):
    d=frames_db(w); thr=max(-55.0, np.percentile(d,95)-25.0)
    act=d>thr
    return act, d
def active_level(w,act,d):
    return float(10*np.log10(np.mean(10**(d[act]/10)))) if act.any() else -120.0
meta={}
for l in open(f'{RUN}/samples.jsonl'):
    s=json.loads(l); meta[s['id']]=s['record']
out={}
files=sorted(glob.glob('wavpos/*.npz'))
for i,fn in enumerate(files):
    sid=os.path.basename(fn)[:-4].replace('__',':')
    z=np.load(fn); e=z['enroll'].astype(np.float32)/32768; m=z['mix'].astype(np.float32)/32768
    r=meta[sid]; md=r['metadata']
    ea,ed=vad(e); ma,mdb=vad(m)
    # enrollment embedding on its voiced part (zero padding removed), mixture per interferer on its utt spans
    mask=np.pad(np.repeat(ea,160),(0,max(0,len(e)-len(ea)*160)))[:len(e)]; ev=e[mask] if ea.sum()>30 else e[np.abs(e)>0] if (np.abs(e)>0).sum()>SR*0.3 else e
    ee=embed(ev)
    sims=[]; gains=[]
    for it in md.get('interferers',[]):
        segs=[m[int(u['start']*SR):int((u['start']+u['duration'])*SR)] for u in it['utts']]
        seg=np.concatenate(segs) if segs else m
        if len(seg)<int(0.3*SR): continue
        sims.append(float(ee@embed(seg))); gains.append(it.get('gain_db',0.0))
    whole=float(ee@embed(m))
    union=0.0
    spans=sorted((u['start'],u['start']+u['duration']) for it in md.get('interferers',[]) for u in it['utts'])
    cur=None
    for a,b in spans:
        b=min(b,len(m)/SR)
        if cur and a<=cur[1]: cur[1]=max(cur[1],b)
        else:
            if cur: union+=cur[1]-cur[0]
            cur=[a,b]
    if cur: union+=cur[1]-cur[0]
    out[sid]=dict(sim_int_max=max(sims) if sims else whole, sim_int_loudest=sims[int(np.argmax(gains))] if sims else whole,
                  sim_mix=whole, mix_dur=len(m)/SR, mix_speech_vad=float(ma.sum()*0.01), mix_speech_meta=union,
                  enroll_speech_vad=float(ea.sum()*0.01), mix_level_db=active_level(m,ma,mdb), enroll_level_db=active_level(e,ea,ed),
                  mix_peak_db=float(20*np.log10(np.abs(m).max()+1e-9)))
    if i%200==0: print(i,len(files),flush=True)
json.dump(out,open('feats_pos.json','w'))
print('ok',len(out))
