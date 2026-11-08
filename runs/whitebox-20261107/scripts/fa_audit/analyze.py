import json, re, random, collections, numpy as np
RUN='/workspace/data/ae-runs/whitebox-merge-sum-full/runs/run-20261107T054755Z-b3a6683a'
F=json.load(open('feats.json')); ST=json.load(open('states_small.json'))
S={};P={}
for l in open(f'{RUN}/samples.jsonl'):
    s=json.loads(l); S[s['id']]=s
for l in open(f'{RUN}/predictions.jsonl'):
    p=json.loads(l); P[p['id']]=p
HOT=['short-hard-neg','enroll-short1-neg','realreg3-hard-neg','abs-hard','enroll-contam0-neg','enroll-contam10-neg','short-rand-neg','abs-hard-k1','realreg3-mid2-neg','enroll-len2-neg']
CTRL=['enroll-clean-neg','realreg3-neg','abs-rand-k1','abs-k2']
def norm(t): return re.sub(r'[^\w]|_','',t.lower())
def lcs(a,b):
    if not a or not b: return 0
    prev=[0]*(len(b)+1)
    for ca in a:
        cur=[0]
        for j,cb in enumerate(b): cur.append(prev[j]+1 if ca==cb else max(prev[j+1],cur[j]))
        prev=cur
    return prev[-1]
FILL=set('嗯啊哦呃唉哎诶欸嘿哈呀吧呢嘛么喔噢哇额恩唔')
from pypinyin import lazy_pinyin
def py(t): return lazy_pinyin(t)
def auc(x,y):  # P(FA value > CR value)
    x=np.asarray(x);y=np.asarray(y)
    if len(x)==0 or len(y)==0: return None
    gt=(x[:,None]>y[None]).sum()+0.5*(x[:,None]==y[None]).sum()
    return float(gt/(len(x)*len(y)))
rows=[]
for sid,s in S.items():
    sp=sid.split(':')[0]
    if sp not in HOT+CTRL or s['reference'].strip(): continue
    r=s['record']; md=r['metadata']; lb=r['labels']; p=P[sid]
    pred=norm(p['prediction']); fa=bool(pred)
    itext=''.join(norm(u['text']) for it in md.get('interferers',[]) for u in it['utts'])
    etext=norm(md.get('enrollment_utt',{}).get('text',''))+''.join(norm(u['text']) for u in md.get('enrollment_extra_utts',[]))
    otext=''.join(norm(u['text']) for u in md.get('enrollment_other_utts',[]))
    d=dict(id=sid,split=sp,fa=fa,pred=p['prediction'],n_int=lb.get('n_interferers'),
           hard_cosine=lb.get('hard_cosine'),int_gain=lb.get('interferer_gain_db'),noise_floor_db=lb.get('noise_floor_dbfs'),
           noise_kind=lb.get('noise_floor_kind'),
           mixkey=(lb.get('pair'),tuple(u['id'] for it in md.get('interferers',[]) for u in it['utts'])),
           int_speakers=[it['speaker'] for it in md.get('interferers',[])],target_speaker=md.get('target_speaker'),
           other_speaker=(md.get('enrollment_other_utts') or [{}])[0].get('id','').rsplit('_',1)[0] or None)
    d.update(F[sid]); d.update(ST[sid])
    d['snr_db']=d['mix_level_db']-d['noise_floor_db'] if d['noise_floor_db'] is not None else None
    if fa:
        L=len(pred); ci=lcs(pred,itext)/L; ce=lcs(pred,etext)/L; co=lcs(pred,otext)/L if otext else 0
        cip=lcs(py(pred),py(itext))/L  # toneless pinyin per char: catches homophones (嘀嗒/滴答)
        d.update(cov_int=ci,cov_int_pinyin=cip,cov_enroll=ce,cov_other=co)
        if set(pred)<=FILL: cat='filler/backchannel'
        elif max(ci,cip)>=0.5 and ci>=ce: cat='interferer words'
        elif ce>=0.5 or co>=0.5: cat='enrollment words'
        elif L<=4: cat='short other (<=4 chars)'
        else: cat='unrelated/partial'
        d['cat']=cat
    rows.append(d)
R={}
by=collections.defaultdict(list)
for d in rows: by[d['split']].append(d)
# Q1 construction summary
R['construction']={sp:dict(n=len(v),fa=sum(d['fa'] for d in v),
    n_interferers=dict(collections.Counter(d['n_int'] for d in v)),
    mix_dur_median=float(np.median([d['mix_dur'] for d in v])),
    mix_speech_meta_median=float(np.median([d['mix_speech_meta'] for d in v])),
    enroll_speech_vad_median_in_5s=float(np.median([d['enroll_speech_vad'] for d in v])),
    hard_cosine_median=(float(np.median([d['hard_cosine'] for d in v if d['hard_cosine'] is not None])) if any(d['hard_cosine'] is not None for d in v) else None),
    target_in_mixture=sum(d['target_speaker'] in d['int_speakers'] for d in v),
    contam_speaker_in_mixture=sum(bool(d['other_speaker']) and d['other_speaker'] in d['int_speakers'] for d in v))
    for sp,v in by.items()}
# Q2 AUCs
FEATS=['sim_int_max','sim_mix','tower_cos24','hard_cosine','mix_dur','mix_speech_vad','mix_speech_meta','enroll_speech_vad','mix_level_db','enroll_level_db','int_gain','noise_floor_db','snr_db','margin']
R['auc_within_split']={}
pooled={f:[0,0] for f in FEATS}
for sp in HOT+CTRL:
    v=by[sp]; fa=[d for d in v if d['fa']]; cr=[d for d in v if not d['fa']]
    ent={'n_fa':len(fa),'n_cr':len(cr)}
    for f in FEATS:
        x=[d[f] for d in fa if d[f] is not None]; y=[d[f] for d in cr if d[f] is not None]
        a=auc(x,y)
        if a is None: continue
        ent[f]=dict(auc=round(a,3),fa_median=round(float(np.median(x)),3),cr_median=round(float(np.median(y)),3))
        w=len(x)*len(y); pooled[f][0]+=a*w; pooled[f][1]+=w
    R['auc_within_split'][sp]=ent
R['auc_pooled_hot']={}
for f in FEATS:
    num=den=0
    for sp in HOT:
        e=R['auc_within_split'][sp].get(f)
        if e: w=R['auc_within_split'][sp]['n_fa']*R['auc_within_split'][sp]['n_cr']; num+=e['auc']*w; den+=w
    if den: R['auc_pooled_hot'][f]=round(num/den,3)
# paired counterfactuals: same mixture, different enrollment
def pairtab(a,b):
    A={d['mixkey']:d['fa'] for d in by[a]}; B={d['mixkey']:d['fa'] for d in by[b]}
    k=set(A)&set(B); c=collections.Counter((A[x],B[x]) for x in k)
    return dict(n_common=len(k),both_fa=c[(True,True)],only_first=c[(True,False)],only_second=c[(False,True)],neither=c[(False,False)])
R['paired_same_mixture']={f'{a} vs {b}':pairtab(a,b) for a,b in [('short-hard-neg','short-rand-neg'),('enroll-short1-neg','enroll-clean-neg'),
    ('enroll-len2-neg','enroll-clean-neg'),('enroll-contam0-neg','enroll-clean-neg'),('enroll-contam10-neg','enroll-clean-neg'),
    ('realreg3-hard-neg','realreg3-neg'),('abs-hard-k1','abs-rand-k1'),('abs-hard','abs-k2')]}
# Q3 transcript categories
fas=[d for d in rows if d['fa'] and d['split'] in HOT]
R['fa_transcript_categories']={sp:dict(collections.Counter(d['cat'] for d in fas if d['split']==sp)) for sp in HOT}
R['fa_transcript_categories']['ALL_HOT']=dict(collections.Counter(d['cat'] for d in fas))
R['fa_pred_len_chars_median']=float(np.median([len(norm(d['pred'])) for d in fas]))
R['fa_cov_int_median']=float(np.median([d['cov_int'] for d in fas]))
random.seed(0)
pool=[d for d in fas if d['split'] in ('short-hard-neg','enroll-short1-neg')]
samp=sorted(random.sample(pool,min(30,len(pool))),key=lambda d:(d['split'],d['id']))
R['sample30']=[dict(id=d['id'],split=d['split'],mix_dur=round(d['mix_dur'],2),mix_speech=round(d['mix_speech_vad'],2),enroll_speech=round(d['enroll_speech_vad'],2),
    sim=round(d['sim_int_max'],3),hard_cosine=d['hard_cosine'],margin=round(d['margin'],2),cat=d['cat'],cov_int=round(d['cov_int'],2),pred=d['pred'][:30],
    int_text=''.join(u['text'] for it in S[d['id']]['record']['metadata']['interferers'] for u in it['utts'])[:30]) for d in samp]
allneg=[d for d in rows if d['split'] in HOT+CTRL]
R['auc_across_splits_all_analyzed']={f:round(auc([d[f] for d in allneg if d['fa'] and d[f] is not None],[d[f] for d in allneg if not d['fa'] and d[f] is not None]),3) for f in ['sim_int_max','tower_cos24','mix_speech_vad','enroll_speech_vad']}
R['margin_fa']=dict(median=float(np.median([d['margin'] for d in fas])),near_tie_abs_lt_2=sum(abs(d['margin'])<2 for d in fas),n=len(fas))
R['short_words_interferer_le2chars']=sum(len(norm(d['pred']))<=2 and d['cat']=='interferer words' for d in fas)
json.dump(R,open('results.json','w'),ensure_ascii=False,indent=1)
json.dump([{k:v for k,v in d.items() if k!='mixkey'} for d in rows],open('rows.json','w'),ensure_ascii=False)
print(json.dumps({k:R[k] for k in ['auc_pooled_hot','paired_same_mixture','fa_transcript_categories','fa_pred_len_chars_median','fa_cov_int_median']},ensure_ascii=False,indent=0))
