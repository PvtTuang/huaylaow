import os, sys, random
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
from collections import Counter
from lottery.models import LotteryResult

rs = list(LotteryResult.objects.exclude(first_prize='').order_by('draw_date'))
bot = [r.last_two for r in rs]
top = [r.last_two_top for r in rs]
N = len(rs)
ALL = [f"{a}{b}" for a in range(10) for b in range(10)]

def rank_digit_indep(W, use_top=True):
    def f(t):
        lo = max(0, t - W)
        c1, c2 = Counter(), Counter()
        for i in range(lo, t):
            c1[bot[i][0]] += 1; c2[bot[i][1]] += 1
            if use_top:
                c1[top[i][0]] += 1; c2[top[i][1]] += 1
        return sorted(ALL, key=lambda k: (-c1[k[0]] * c2[k[1]], k))
    return f

def rank_pair_freq(W, use_top=True, half=None):
    def f(t):
        lo = max(0, t - W)
        c = Counter()
        for i in range(lo, t):
            w = 0.5 ** ((t - 1 - i) / half) if half else 1.0
            c[bot[i]] += w
            if use_top: c[top[i]] += w
        return sorted(ALL, key=lambda k: (-c[k], k))
    return f

def rank_overdue(use_top=True):
    def f(t):
        last = {}
        for i in range(t):
            last[bot[i]] = i
            if use_top: last[top[i]] = i
        return sorted(ALL, key=lambda k: (last.get(k, -1), k))
    return f

def rank_cold_digit(W):
    def f(t):
        lo = max(0, t - W)
        c1, c2 = Counter(), Counter()
        for i in range(lo, t):
            for s in (bot[i], top[i]):
                c1[s[0]] += 1; c2[s[1]] += 1
        return sorted(ALL, key=lambda k: (c1[k[0]] * c2[k[1]], k))
    return f

def rank_markov(W):
    # next bottom conditioned on previous bottom's last digit -> digit freq
    def f(t):
        lo = max(1, t - W)
        prev = bot[t - 1][1]
        c1, c2 = Counter(), Counter()
        for i in range(lo, t):
            if bot[i - 1][1] == prev:
                c1[bot[i][0]] += 1; c2[bot[i][1]] += 1
        return sorted(ALL, key=lambda k: (-(c1[k[0]] + .1) * (c2[k[1]] + .1), k))
    return f

def rank_random(seed=1):
    rnd = random.Random(seed)
    def f(t):
        l = ALL[:]; rnd.shuffle(l); return l
    return f

algos = {'random': rank_random(), 'overdue': rank_overdue(), 'overdue_botonly': rank_overdue(False)}
for W in (20, 30, 62, 100, 150, 250, 400):
    algos[f'diind_W{W}'] = rank_digit_indep(W)
    algos[f'diind_bot_W{W}'] = rank_digit_indep(W, False)
    algos[f'pair_W{W}'] = rank_pair_freq(W)
for W in (62, 150, 400):
    algos[f'cold_W{W}'] = rank_cold_digit(W)
    algos[f'markov_W{W}'] = rank_markov(W)
for h in (10, 30, 80):
    algos[f'pair_half{h}'] = rank_pair_freq(400, True, h)

START = 100
def evaluate(f, k=3, lo=START, hi=N):
    hit = 0; n = 0
    for t in range(lo, hi):
        picks = set(f(t)[:k])
        if bot[t] in picks or top[t] in picks:
            hit += 1
        n += 1
    return hit, n

print(f"draws={N}, eval from {START}; random baseline for 3 picks ~{(1-(0.97)**2)*100:.1f}%")
rows = []
for name, f in algos.items():
    h, n = evaluate(f)
    h2, n2 = evaluate(f, lo=N - 120)
    rows.append((h / n, name, h, n, h2, n2))
rows.sort(reverse=True)
for r in rows:
    print(f"{r[1]:18s} all {r[2]:3d}/{r[3]} = {r[0]*100:5.2f}%   last120 {r[4]}/{r[5]} = {r[4]/r[5]*100:5.1f}%")
