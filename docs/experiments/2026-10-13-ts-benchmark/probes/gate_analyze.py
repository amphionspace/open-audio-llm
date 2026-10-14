import gzip, json, glob, sys, numpy as np
from collections import defaultdict
scores = json.load(open(sys.argv[1]))
root = '/workspace/data/datasets/ts_asr_benchmark/versions/v1-20261013/records'
by = defaultdict(list)
for f in glob.glob(root + '/*.jsonl.gz'):
    for l in gzip.open(f, 'rt'):
        r = json.loads(l); by[r['labels']['scenario']].append((scores[r['id']], r['labels']['target_present']))
def vals(name): return np.array([s for s, _ in by[name]])
def auc(p, n):
    allv = np.concatenate([p, n]); ranks = allv.argsort().argsort() + 1
    return (ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
def eer(p, n):
    best = 1
    for t in np.unique(np.concatenate([p, n])):
        fr = (p < t).mean(); fa = (n >= t).mean(); best = min(best, max(fr, fa)) if abs(fr - fa) < 0.05 else best
    ts = np.unique(np.concatenate([p, n]))
    d = [(abs((p < t).mean() - (n >= t).mean()), t) for t in ts]; t = min(d)[1]
    return ((p < t).mean() + (n >= t).mean()) / 2, t
print('cos stats: target-present mean %.3f, absent mean %.3f' % (np.mean([s for v in by.values() for s, p in v if p]), np.mean([s for v in by.values() for s, p in v if not p])))
print(f"{'pos split':22s} {'neg split':20s} {'AUC':>5s} {'EER%':>5s} {'thr':>5s}")
pairs = [(f'enroll-{c}-pos', f'enroll-{c}-neg') for c in ['clean','short1','lead_silence','lp4k','noise10','reverb','contam10','contam0','quiet']]
pairs += [('dyn-full-k1','abs-k1'),('dyn-full-k4','abs-k4'),('dyn-burst-k2','abs-k2'),('dyn-burst-k4','abs-k4'),('dyn-alternate-k2','abs-k2'),('dyn-overlap-k3','abs-k2'),('dyn-short-k1','abs-k1'),('dyn-short-k3','abs-k2'),('dyn-target-quiet','abs-k2'),('dyn-full-k1','abs-hard'),
          ('dom-clean-pos','dom-clean-neg'),('dom-reverb-noise-pos','dom-reverb-noise-neg'),('dom-music-pos','dom-music-neg'),('dom-traffic-pos','dom-traffic-neg'),('dom-phone-pos','dom-phone-neg'),('dom-studio-pos','dom-studio-neg')]
res = {}
for p, n in pairs:
    pv, nv = vals(p), vals(n); e, t = eer(pv, nv)
    res[(p, n)] = (auc(pv, nv), e, t)
    print(f"{p:22s} {n:20s} {auc(pv,nv):5.2f} {100*e:5.1f} {t:5.2f}")
# fixed threshold calibrated on clean: accept 90% of clean positives
tc = np.percentile(vals('enroll-clean-pos'), 10)
print('\nthreshold %.3f (90%% of clean positives pass); rate of negatives passing (false accept) under the gate alone:' % tc)
for n in ['enroll-clean-neg','enroll-short1-neg','enroll-contam10-neg','abs-k1','abs-k2','abs-k4','abs-hard','dom-phone-neg','dom-studio-neg']:
    print(f"  {n:22s} FA {100*(vals(n)>=tc).mean():5.1f}%")
print('positives passing: ', {p: round(100*(vals(p)>=tc).mean(),1) for p in ['enroll-short1-pos','enroll-contam10-pos','dyn-burst-k2','dyn-burst-k4','dyn-short-k1','dyn-target-quiet','dom-phone-pos']})
