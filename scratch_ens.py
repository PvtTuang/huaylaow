import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
import numpy as np
from lottery.models import LotteryResult

rs = list(LotteryResult.objects.exclude(first_prize='').order_by('draw_date'))
N = len(rs)
B = np.array([[int(c) for c in r.last_two] for r in rs])
T = np.array([[int(c) for c in r.last_two_top] for r in rs])
def oh(M, p):
    o = np.zeros((N, 10)); o[np.arange(N), M[:, p]] = 1; return o
C = {k: np.vstack([np.zeros((1, 10)), np.cumsum(oh(M, p), 0)]) for k, (M, p) in
     {'b1': (B, 0), 'b2': (B, 1), 't1': (T, 0), 't2': (T, 1)}.items()}

def cnt(t, W, wt=1.0):
    lo = max(0, t - W)
    c1 = (C['b1'][t] - C['b1'][lo]) + wt * (C['t1'][t] - C['t1'][lo])
    c2 = (C['b2'][t] - C['b2'][lo]) + wt * (C['t2'][t] - C['t2'][lo])
    return c1, c2

def score_ens(t, Ws, mix):
    s = np.zeros((10, 10))
    for W in Ws:
        c1, c2 = cnt(t, W)
        p1 = (c1 + 1) / (c1.sum() + 10); p2 = (c2 + 1) / (c2.sum() + 10)
        if mix == 'prod':
            s += np.log(np.outer(p1, p2))          # geometric ensemble
        else:
            s += np.outer(p1, p2)
    return s

def top3(s):
    idx = np.argsort(-s.ravel(), kind='stable')[:3]
    return set(int(i) for i in idx)

def run(Ws, mix, lo, hi):
    h = 0
    for t in range(lo, hi):
        p = top3(score_ens(t, Ws, mix))
        if B[t, 0] * 10 + B[t, 1] in p or T[t, 0] * 10 + T[t, 1] in p: h += 1
    return h

START, SPLIT = 100, N - 120
sets = {
    'W62 (current)': (62,),
    'ens 30+62+120': (30, 62, 120),
    'ens 45+62+80': (45, 62, 80),
    'ens 20..200': (20, 30, 62, 120, 200),
    'ens 50..75': (50, 56, 62, 68, 75),
    'ens 62+400': (62, 400),
}
print(f"train {START}..{SPLIT}, holdout last 120, chance 5.9%")
for name, Ws in sets.items():
    for mix in ('prod', 'sum'):
        a = run(Ws, mix, START, SPLIT); b = run(Ws, mix, SPLIT, N)
        print(f"{name:16s} {mix:4s} train {a/(SPLIT-START)*100:5.2f}%  holdout {b}/120={b/120*100:5.1f}%  all {(a+b)/(N-START)*100:5.2f}%")
