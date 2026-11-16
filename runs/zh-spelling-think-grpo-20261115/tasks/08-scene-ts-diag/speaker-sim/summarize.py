import csv, numpy as np, collections
rows=list(csv.DictReader(open('all_items.csv')))
err=[r for r in rows if r['err']]; print('errors',len(err))
g=collections.defaultdict(list); ml=collections.defaultdict(list); el=collections.defaultdict(list)
for r in rows:
    if r['err']: continue
    g[r['group']].append(float(r['sim'])); ml[r['group']].append(float(r['mix_s'])); el[r['group']].append(float(r['enroll_s']))
g['real_ts_real_train_ALL(1500)']=sum((g[k] for k in list(g) if k.startswith('real_ts_real_train_')),[])
g['real_ALL(2000)']=g['real_ts_real_train_ALL(1500)']+g['real_ts_scene_pool_zh_alt_twins']
g['scene_ts_present_ALL']=g['scene_ts_sep_present']+g['scene_ts_overlap_present']
g['scene_ts_absent_ALL']=g['scene_ts_sep_absent']+g['scene_ts_overlap_absent']
print('| group | n | mean | median | p10 | p25 | p75 | <0.4 | <0.5 | med enroll s | med mix s |\n|---|---|---|---|---|---|---|---|---|---|---|')
for k,v in g.items():
    a=np.array(v); me=np.median(el[k]) if k in el else float('nan'); mm=np.median(ml[k]) if k in ml else float('nan')
    print(f"| {k} | {len(a)} | {a.mean():.3f} | {np.median(a):.3f} | {np.percentile(a,10):.3f} | {np.percentile(a,25):.3f} | {np.percentile(a,75):.3f} | {(a<.4).mean():.1%} | {(a<.5).mean():.1%} | {me:.1f} | {mm:.1f} |")
