"""For each v12 target-present sample, compare the TS-ASR errors with the same model's plain-ASR errors on the
same target utterances recorded alone (matched by corpus + utterance id from the plain-ASR benchmark run).
Usage: python clean_vs_ts.py TS_RUN_DIR ASR_RUN_DIR"""
import json, sys, collections
from amphion_eval.integrations.open_audio_llm import scoring
def ed(r, h):
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j - 1] + (r[i - 1] != h[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]
ts_run, asr_run = sys.argv[1], sys.argv[2]
clean = {}
for l in open(asr_run + '/predictions.jsonl'):
    p = json.loads(l); corpus = p['id'].split('@')[0]; utt = p['id'].rsplit(':', 1)[-1]
    clean[(corpus, utt)] = p
samples = {json.loads(l)['id']: json.loads(l) for l in open(ts_run + '/samples.jsonl')}
B = collections.defaultdict(lambda: collections.Counter())
for l in open(ts_run + '/predictions.jsonl'):
    p = json.loads(l)
    if p.get('task') != 'ts_asr' or p.get('language') != 'zh' or not p['reference']: continue
    s = samples[p['id']]; m = s['record'].get('metadata') or {}; lab = s['record'].get('labels') or {}
    tu = m.get('target_utts') or []
    if not tu: continue
    cps = [clean.get((u['id'].split(':')[0], u['id'].split(':', 1)[1])) for u in tu]
    if any(c is None for c in cps): continue
    r = scoring.tokens(p['reference'], 'zh'); h = scoring.tokens(p.get('prediction') or '', 'zh')
    cr = scoring.tokens(''.join(c['reference'] for c in cps), 'zh'); ch = scoring.tokens(''.join(c.get('prediction') or '' for c in cps), 'zh')
    if abs(len(cr) - len(r)) > 2: continue   # reference text differs (e.g., partial target) -> skip
    ov = lab.get('target_overlap_ratio'); ni = lab.get('n_interferers')
    keys = ['all', 'overlap=' + ('na' if ov is None else '0' if ov == 0 else '<0.3' if ov < .3 else '<0.6' if ov < .6 else '<0.9' if ov < .9 else '>=0.9'),
            'n_interf=' + str(ni), 'group=' + str(lab.get('group'))]
    e_ts = ed(r, h); e_cl = ed(cr, ch)
    for k in keys:
        c = B[k]; c['n'] += 1; c['ref'] += len(r); c['ts_err'] += e_ts; c['clean_err'] += e_cl; c['miss'] += (not h)
        c['ts_err_nomiss'] += e_ts if h else 0; c['ref_nomiss'] += len(r) if h else 0; c['clean_err_nomiss'] += e_cl if h else 0
print(f"{'bin':22s} {'n':>5s} {'ref':>6s} {'clean%':>7s} {'TS%':>6s} {'TS-clean':>8s} | excl. whole-miss: {'clean%':>6s} {'TS%':>6s}")
for k in sorted(B, key=lambda x: (x.split('=')[0], x)):
    c = B[k]
    print(f"{k:22s} {c['n']:5d} {c['ref']:6d} {100*c['clean_err']/c['ref']:6.2f}% {100*c['ts_err']/c['ref']:5.2f}% {100*(c['ts_err']-c['clean_err'])/c['ref']:+7.2f} | {100*c['clean_err_nomiss']/max(c['ref_nomiss'],1):6.2f}% {100*c['ts_err_nomiss']/max(c['ref_nomiss'],1):5.2f}%  miss {c['miss']}")
