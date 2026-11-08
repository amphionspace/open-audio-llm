"""ECAPA verification vs the model's own decision margin, per pos/neg family (v12) + gate upper bound (v12, first-other v3).

  <python with sklearn> analyze.py --states-small <npz from states-sumfull-v12> --out ecapa-check.json
Inputs: scores-*.jsonl from score.py (same task dir), AmphionEval samples/predictions.
Score orientation: higher = "target present". model = -(logp None - max logp language) at the last layer.
"""
import argparse, json, collections, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, roc_curve

RUNS = "/workspace/data/ae-runs/whitebox-merge-sum-full/runs"
V12 = f"{RUNS}/run-20261107T054755Z-b3a6683a"
FO = f"{RUNS}/run-20261107T061511Z-6fd61563"        # ts-asr-first-other@v3-20261105
TASK = "/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/whitebox-20261107/tasks/10-ecapa-check"
HOT = ['short-hard-neg', 'enroll-short1-neg', 'realreg3-hard-neg', 'abs-hard', 'enroll-contam0-neg', 'enroll-contam10-neg',
       'short-rand-neg', 'abs-hard-k1', 'realreg3-mid2-neg', 'enroll-len2-neg']       # 09-fa-audit hotspots
HARD = ['short-hard|short-k1', 'abs-hard|dyn-k2', 'abs-hard-k1|dyn-k1', 'realreg3-hard|real-pos', 'enroll-short1',
        'enroll-len2', 'enroll-contam0', 'enroll-contam10']
BUDGETS = (0.005, 0.01)        # hits lost as a fraction of all positives
NEAR_TIE = 3.0                 # nat; band where a small bias flips the decision (09: 24% of FAs within ±2)
MARGIN_CLIP = 30.0             # nat; keeps the logistic fit from being driven by a few extreme margins


def load_run(run, scores):
    S = {}
    P = {json.loads(l)['id']: json.loads(l) for l in open(f'{run}/predictions.jsonl')}
    E = {json.loads(l)['id']: json.loads(l) for l in open(scores)}
    for l in open(f'{run}/samples.jsonl'):
        s = json.loads(l); sid = s['id']; sp = sid.split(':')[0]
        if sp.startswith('ts-2026'):      # customer TS: not part of this check (kept local, different source)
            continue
        S[sid] = dict(split=sp, present=bool(s['reference'].strip()), out=bool(P[sid]['prediction'].strip()), **E[sid])
    return S


def families(splits, kind):
    sp = set(splits); F = {}
    for x in sp:
        if x.endswith('-pos') and x[:-4] + '-neg' in sp:
            F[x[:-4]] = dict(pos=[x], neg=[x[:-4] + '-neg'])
    if kind == 'v12':
        k1 = [f'dyn-{s}-k1' for s in ('full', 'burst', 'overlap', 'short')]
        k2 = [f'dyn-{s}-k2' for s in ('alternate', 'burst', 'enter', 'leave')]
        realpos = sorted(x for x in sp if x.startswith('real-') and x.endswith('-pos'))
        F.update({'abs-hard-k1|dyn-k1': dict(neg=['abs-hard-k1'], pos=k1), 'abs-rand-k1|dyn-k1': dict(neg=['abs-rand-k1'], pos=k1),
                  'abs-hard|dyn-k2': dict(neg=['abs-hard'], pos=k2), 'abs-k2|dyn-k2': dict(neg=['abs-k2'], pos=k2),
                  'realreg3-hard|real-pos': dict(neg=['realreg3-hard-neg'], pos=realpos),
                  'realreg3-mid2|real-pos': dict(neg=['realreg3-mid2-neg'], pos=realpos),
                  'realreg3|real-pos': dict(neg=['realreg3-neg'], pos=realpos),
                  'short-hard|short-k1': dict(neg=['short-hard-neg'], pos=['short-k1-pos']),
                  'short-rand|short-k1': dict(neg=['short-rand-neg'], pos=['short-k1-pos'])})
    return dict(sorted(F.items()))


def auc(y, x):
    return round(float(roc_auc_score(y, x)), 4) if 0 < sum(y) < len(y) else None


def eer_threshold(y, x):
    fpr, tpr, thr = roc_curve(y, x); i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return float(thr[i]), float((fpr[i] + 1 - tpr[i]) / 2)


def gate(items, key, budget, n_pos, thr=None):
    """Suppress model output when score < t. t = largest value with suppressed hits <= budget * n_pos (unless given)."""
    hits = np.sort([d[key] for d in items if d['present'] and d['out']])
    if thr is None:
        k = int(np.floor(budget * n_pos)); thr = float(hits[k]) if k < len(hits) else np.inf  # strictly below hits[k]
    fa = [d for d in items if not d['present'] and d['out']]
    rm = [d for d in fa if d[key] < thr]
    return dict(threshold=round(thr, 4), hits_lost=int((hits < thr).sum()), fa_removed=len(rm), fa_total=len(fa),
                hot_fa_removed=sum(d['split'] in HOT for d in rm), hot_fa_total=sum(d['split'] in HOT for d in fa))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--states-small', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    S = load_run(V12, f'{TASK}/scores-v12.jsonl')
    z = np.load(a.states_small, allow_pickle=True)
    for sid, m, c, fam, emit in zip(z['id'], z['margin'], z['enc24_cos'], z['final_argmax'], z['emitted']):
        if sid in S:
            S[sid].update(model=-float(m), enc24=float(c), hf_agrees=bool(fam == emit))
    ids = sorted(S); items = [S[i] for i in ids]
    # out-of-fold logistic combination; folds grouped by split family (base name), so no family is scored by a model fit on it
    X = np.array([[d['ecapa_max'], np.clip(d['model'], -MARGIN_CLIP, MARGIN_CLIP)] for d in items])
    y = np.array([d['present'] for d in items], int)
    groups = [d['split'].removesuffix('-pos').removesuffix('-neg') for d in items]
    mu, sd = X.mean(0), X.std(0); Xs = (X - mu) / sd
    oof = np.zeros(len(items))
    for tr, te in GroupKFold(10).split(Xs, y, groups):
        oof[te] = LogisticRegression(C=1.0).fit(Xs[tr], y[tr]).decision_function(Xs[te])
    full = LogisticRegression(C=1.0).fit(Xs, y)
    for d, o in zip(items, oof):
        d['comb'] = float(o)
    R = dict(n_items=len(items), n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()),
             model_fa=sum(not d['present'] and d['out'] for d in items), model_miss=sum(d['present'] and not d['out'] for d in items),
             hf_vllm_first_token_disagree=sum(not d['hf_agrees'] for d in items),
             logistic_full_fit=dict(coef_std=full.coef_[0].round(3).tolist(), intercept=round(float(full.intercept_[0]), 3),
                                    feature_mean=mu.round(4).tolist(), feature_std=sd.round(4).tolist(),
                                    features=['ecapa_max', f'-margin clipped ±{MARGIN_CLIP}']))
    by = collections.defaultdict(list)
    for d in items:
        by[d['split']].append(d)
    # EER threshold for ECAPA (and comb) on clean enrollment pos/neg
    clean = by['enroll-clean-pos'] + by['enroll-clean-neg']
    yc = [d['present'] for d in clean]
    thr_e, eer_e = eer_threshold(yc, [d['ecapa_max'] for d in clean])
    thr_c, eer_c = eer_threshold(yc, [d['comb'] for d in clean])
    R['clean_eer'] = dict(ecapa=dict(threshold=round(thr_e, 4), eer=round(eer_e, 4)), comb=dict(threshold=round(thr_c, 4), eer=round(eer_c, 4)))

    def errs(group, comb=True):
        e = [d for d in group if d['present'] != d['out']]
        fa = [d for d in e if not d['present']]; mi = [d for d in e if d['present']]
        ok = lambda d, k, t: (d[k] >= t) == d['present']
        good = [d for d in group if d['present'] == d['out']]
        return dict(n_fa=len(fa), n_miss=len(mi), n_model_correct=len(good),
                    ecapa_right_on_model_correct=sum(ok(d, 'ecapa_max', thr_e) for d in good),
                    ecapa_right_fa=sum(ok(d, 'ecapa_max', thr_e) for d in fa), ecapa_right_miss=sum(ok(d, 'ecapa_max', thr_e) for d in mi),
                    **(dict(comb_right_fa=sum(ok(d, 'comb', thr_c) for d in fa), comb_right_miss=sum(ok(d, 'comb', thr_c) for d in mi))
                       if comb else {}))
    F = families(by, 'v12'); fam_rows = {}; hard_pool = []
    for name, f in F.items():
        g = [d for s in f['pos'] + f['neg'] for d in by[s]]
        yy = [d['present'] for d in g]
        fam_rows[name] = dict(pos=f['pos'], neg=f['neg'], n_pos=sum(yy), n_neg=len(yy) - sum(yy),
                              auc_ecapa=auc(yy, [d['ecapa_max'] for d in g]), auc_ecapa_whole=auc(yy, [d['ecapa_whole'] for d in g]),
                              auc_model=auc(yy, [d['model'] for d in g]), auc_comb=auc(yy, [d['comb'] for d in g]),
                              auc_enc24=auc(yy, [d['enc24'] for d in g]), **errs(g))
        if name in HARD:
            hard_pool += g
    R['families'] = fam_rows
    yh = [d['present'] for d in hard_pool]
    R['hard_pooled'] = dict(families=HARD, n=len(hard_pool), auc_ecapa=auc(yh, [d['ecapa_max'] for d in hard_pool]),
                            auc_model=auc(yh, [d['model'] for d in hard_pool]), auc_comb=auc(yh, [d['comb'] for d in hard_pool]),
                            auc_enc24=auc(yh, [d['enc24'] for d in hard_pool]))
    # does ECAPA separate present/absent among the model's near-ties? (|margin| <= NEAR_TIE nat)
    near = [d for d in items if abs(d['model']) <= NEAR_TIE]
    yn = [d['present'] for d in near]
    R['near_tie'] = dict(nat=NEAR_TIE, n=len(near), n_pos=sum(yn), auc_ecapa=auc(yn, [d['ecapa_max'] for d in near]),
                         auc_model=auc(yn, [d['model'] for d in near]), auc_comb=auc(yn, [d['comb'] for d in near]),
                         auc_enc24=auc(yn, [d['enc24'] for d in near]))
    R['errors'] = dict(all=errs(items), hotspot=errs([d for d in items if d['split'] in HOT]))
    R['gate_v12'] = {f'{b:.3f}': {k: gate(items, k, b, int(y.sum())) for k in ('model', 'ecapa_max', 'enc24', 'comb')} for b in BUDGETS}

    # first-other v3: ECAPA + the model's empty/non-empty output only (no states)
    FOs = load_run(FO, f'{TASK}/scores-firstother-v3.jsonl'); fo = list(FOs.values())
    yf = [d['present'] for d in fo]; npos = sum(yf)
    fby = collections.defaultdict(list)
    for d in fo:
        fby[d['split']].append(d)
    ff = {}
    for name, f in families(fby, 'fo').items():
        g = [d for s in f['pos'] + f['neg'] for d in fby[s]]; yy = [d['present'] for d in g]
        ff[name] = dict(n_pos=sum(yy), n_neg=len(yy) - sum(yy), auc_ecapa=auc(yy, [d['ecapa_max'] for d in g]),
                        auc_model_binary=auc(yy, [d['out'] for d in g]), **errs(g, comb=False),
                        gate_oracle_family={f'{b:.3f}': gate(g, 'ecapa_max', b, sum(yy)) for b in BUDGETS})  # threshold tuned on this family
    R['first_other_v3'] = dict(
        run=FO, n_items=len(fo), n_pos=npos, n_neg=len(fo) - npos, model_fa=sum(not d['present'] and d['out'] for d in fo),
        model_miss=sum(d['present'] and not d['out'] for d in fo), clean_eer_threshold_from_v12=round(thr_e, 4),
        auc_ecapa_all=auc(yf, [d['ecapa_max'] for d in fo]), auc_model_binary_all=auc(yf, [d['out'] for d in fo]),
        families=ff, errors=errs(fo, comb=False),
        gate_oracle={f'{b:.3f}': gate(fo, 'ecapa_max', b, npos) for b in BUDGETS},
        gate_v12_threshold={f'{b:.3f}': gate(fo, 'ecapa_max', b, npos, thr=R['gate_v12'][f'{b:.3f}']['ecapa_max']['threshold']) for b in BUDGETS})
    json.dump(R, open(a.out, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({k: R[k] for k in ('n_items', 'model_fa', 'model_miss', 'clean_eer', 'hard_pooled', 'near_tie', 'errors', 'gate_v12')}, indent=1))
    print(f"{'family':28s} npos nneg  ecapa  whole  model  comb   enc24  FA eR cR  miss eR cR")
    for n, r in fam_rows.items():
        print(f"{n:28s} {r['n_pos']:4d} {r['n_neg']:4d}  {r['auc_ecapa']}  {r['auc_ecapa_whole']}  {r['auc_model']}  {r['auc_comb']}  {r['auc_enc24']}  "
              f"{r['n_fa']} {r['ecapa_right_fa']} {r['comb_right_fa']}  {r['n_miss']} {r['ecapa_right_miss']} {r['comb_right_miss']}")
    print(json.dumps({k: v for k, v in R['first_other_v3'].items() if k != 'families'}, indent=1))
    for n, r in ff.items():
        print(n, r)


if __name__ == '__main__':
    main()
