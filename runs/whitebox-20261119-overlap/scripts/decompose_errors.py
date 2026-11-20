"""Decompose Chinese TS-ASR (v12) character errors into presence errors vs in-utterance errors,
and estimate how many wrong output characters are copied from the interferer's speech.
Usage: python decompose.py OUT.json RUN_DIR [RUN_DIR ...]   (RUN_DIR has samples.jsonl + predictions.jsonl)"""
import json, sys, collections, random, os
FALLBACK_SAMPLES = '/workspace/data/ae-runs/whitebox-merge-v18a-es116inc/runs/run-20261117T032925Z-961b82b3/samples.jsonl'  # same v12 benchmark ids; used when a run reused inference and has no samples.jsonl
from amphion_eval.integrations.open_audio_llm import scoring

def align(r, h):
    n, m = len(r), len(h)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        ri = r[i - 1]; Di = D[i]; Dp = D[i - 1]
        for j in range(1, m + 1):
            Di[j] = min(Dp[j - 1] + (ri != h[j - 1]), Dp[j] + 1, Di[j - 1] + 1)
    i, j = n, m; S = Dl = I = 0; wrong_h = []
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            if r[i - 1] != h[j - 1]: S += 1; wrong_h.append(j - 1)
            i -= 1; j -= 1
        elif i and D[i][j] == D[i - 1][j] + 1: Dl += 1; i -= 1
        else: I += 1; wrong_h.append(j - 1); j -= 1
    return S, Dl, I, wrong_h

def bigrams(t): return {(t[k], t[k + 1]) for k in range(len(t) - 1)}
def leaked(h, js, bg):
    return sum(1 for j in js if (j > 0 and (h[j - 1], h[j]) in bg) or (j + 1 < len(h) and (h[j], h[j + 1]) in bg))

def bins(lab, src):
    b = {}
    b['source'] = src.split('@')[0]
    b['group'] = str(lab.get('group'))
    b['n_interf'] = str(lab.get('n_interferers'))
    ov = lab.get('target_overlap_ratio')
    b['overlap'] = 'na' if ov is None else ('0' if ov == 0 else '<0.3' if ov < .3 else '<0.6' if ov < .6 else '<0.9' if ov < .9 else '>=0.9')
    g = lab.get('interferer_gain_db')
    b['interf_gain_db'] = 'na' if g is None else ('<=-6' if g <= -6 else '<0' if g < 0 else '0..3' if g <= 3 else '>3')
    b['enroll'] = str(lab.get('enroll_condition'))
    return b

out = {}
for run in sys.argv[2:]:
    samples = {}
    for l in open(run + '/samples.jsonl' if os.path.exists(run + '/samples.jsonl') else FALLBACK_SAMPLES):
        s = json.loads(l); samples[s['id']] = s
    preds = [json.loads(l) for l in open(run + '/predictions.jsonl')]
    interf_tok = {}
    for sid, s in samples.items():
        m = s['record'].get('metadata', {}) or {}
        txt = ''.join(u.get('text', '') for it in (m.get('interferers') or []) for u in it.get('utts', []))
        interf_tok[sid] = scoring.tokens(txt, 'zh') if txt else []
    ids_with_interf = [k for k, v in interf_tok.items() if v]
    rng = random.Random(0)
    T = collections.Counter(); B = collections.defaultdict(collections.Counter)
    for p in preds:
        if p.get('task') != 'ts_asr' or p.get('language') != 'zh': continue
        if p['id'] not in samples: continue
        s = samples[p['id']]; lab = s['record'].get('labels', {}) or {}
        r = scoring.tokens(p['reference'] or '', 'zh'); h = scoring.tokens(p.get('prediction') or '', 'zh')
        c = collections.Counter(); c['utt'] += 1; c['ref'] += len(r)
        if not r:
            c['absent_utt'] += 1
            if h: c['fa_utt'] += 1; c['fa_chars'] += len(h)
            bg = bigrams(interf_tok[p['id']]);
            if h and bg: c['fa_leak_chars'] += leaked(h, range(len(h)), bg)
        elif not h:
            c['miss_utt'] += 1; c['miss_chars'] += len(r)
        else:
            S, Dl, I, js = align(r, h)
            c['present_utt'] += 1; c['present_ref'] += len(r); c['sub'] += S; c['del'] += Dl; c['ins'] += I; c['wrong_h'] += len(js)
            if interf_tok[p['id']]:
                c['wrong_h_with_interf'] += len(js)
                c['leak'] += leaked(h, js, bigrams(interf_tok[p['id']]))
                other = interf_tok[rng.choice(ids_with_interf)]
                c['leak_chance'] += leaked(h, js, bigrams(other))
        T.update(c)
        for k, v in bins(lab, s['source']).items(): B[k + '=' + v].update(c)
    name = run.rstrip('/').split('/')[-3]
    missing = sum(1 for p in preds if p['id'] not in samples)
    if missing: print('WARN', name, 'predictions without sample', missing)
    out[name] = {'total': T, 'bins': B}
    err = T['fa_chars'] + T['miss_chars'] + T['sub'] + T['del'] + T['ins']
    print(f"== {name}  ref {T['ref']}  errors {err}  CER {100*err/T['ref']:.2f}")
    for k in ['fa_chars', 'miss_chars', 'sub', 'del', 'ins']:
        print(f"   {k:10s} {T[k]:6d}  {100*T[k]/err:5.1f}% of errors")
    print(f"   FA utt {T['fa_utt']}/{T['absent_utt']}  miss utt {T['miss_utt']}  in-utt wrong hyp chars {T['wrong_h']}, leak {T['leak']} ({100*T['leak']/max(T['wrong_h_with_interf'],1):.1f}% of wrong chars with interferer), chance {T['leak_chance']}; FA chars leak {T['fa_leak_chars']}/{T['fa_chars']}")
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
