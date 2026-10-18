import os, itertools
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
import numpy as np
from lottery.models import LotteryResult

rs = list(LotteryResult.objects.exclude(first_prize='').order_by('draw_date'))
N = len(rs)
B = np.array([[int(c) for c in r.last_two] for r in rs])      # (N,2) bottom
T = np.array([[int(c) for c in r.last_two_top] for r in rs])  # (N,2) top
ALL = [(a, b) for a in range(10) for b in range(10)]

def onehot(M, pos):
    o = np.zeros((N, 10)); o[np.arange(N), M[:, pos]] = 1; return o
B1, B2, T1, T2 = onehot(B, 0), onehot(B, 1), onehot(T, 0), onehot(T, 1)
CB1, CB2, CT1, CT2 = [np.vstack([np.zeros((1, 10)), np.cumsum(x, 0)]) for x in (B1, B2, T1, T2)]

def window_counts(C, t, W):
    return C[t] - C[max(0, t - W)]

def picks(t, W1, W2, wt, mode):
    # digit counts in window
    c1 = window_counts(CB1, t, W1) + wt * window_counts(CT1, t, W1)
    c2 = window_counts(CB2, t, W2) + wt * window_counts(CT2, t, W2)
    if mode == 'hot':
        s = np.outer(c1, c2)
    elif mode == 'cold':
        s = -np.outer(c1, c2)
    elif mode == 'sum':
        s = c1[:, None] + c2[None, :]
    elif mode == 'hot_d1_cold_d2':
        s = np.outer(c1, -c2 + c2.max() + 1)
    else:  # cold_d1_hot_d2
        s = np.outer(-c1 + c1.max() + 1, c2)
    flat = [(-s[a, b], a * 10 + b) for a, b in ALL]
    flat.sort()
    return {x[1] for x in flat[:3]}

def run(W1, W2, wt, mode, lo, hi):
    h = 0
    for t in range(lo, hi):
        p = picks(t, W1, W2, wt, mode)
        b = B[t, 0] * 10 + B[t, 1]; tp = T[t, 0] * 10 + T[t, 1]
        if b in p or tp in p: h += 1
    return h

START = 100
SPLIT = N - 120
Ws = (10, 20, 30, 45, 62, 80, 120, 200)
res = []
for W1 in Ws:
    for W2 in Ws:
        for wt in (0, 0.5, 1.0):
            for mode in ('hot', 'cold', 'sum', 'hot_d1_cold_d2', 'cold_d1_hot_d2'):
                tr = run(W1, W2, wt, mode, START, SPLIT)
                res.append((tr / (SPLIT - START), W1, W2, wt, mode))
res.sort(reverse=True)
print(f"N={N} train {START}..{SPLIT} ({SPLIT-START} draws) holdout last 120; chance ~5.9%; configs={len(res)}")
for tr, W1, W2, wt, mode in res[:10]:
    ho = run(W1, W2, wt, mode, SPLIT, N)
    print(f"{mode:15s} W1={W1:3d} W2={W2:3d} wt={wt}  train={tr*100:5.2f}%  holdout={ho}/120={ho/120*100:5.1f}%")
