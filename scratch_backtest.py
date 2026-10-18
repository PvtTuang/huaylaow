import os, django, random
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
from collections import Counter
from lottery.models import LotteryResult

rs = list(LotteryResult.objects.exclude(first_prize='').order_by('draw_date'))
last2 = [r.first_prize[-2:] for r in rs]          # 2 ตัวล่าง
top2 = [r.first_prize[2:4] for r in rs]           # 2 ตัวบน
N = len(rs)
PAIRS = [f"{a}{b}" for a in range(10) for b in range(10)]
K = 3


def top_k(scores):
    return sorted(PAIRS, key=lambda p: (-scores[p], p))[:K]


def diind(w, use_top=True):
    def f(t):
        lo = max(0, t - w)
        d1, d2 = Counter(), Counter()
        for i in range(lo, t):
            d1[last2[i][0]] += 1; d2[last2[i][1]] += 1
            if use_top:
                d1[top2[i][0]] += 1; d2[top2[i][1]] += 1
        return top_k({p: d1[p[0]] * d2[p[1]] for p in PAIRS})
    return f


def pair_freq(w):
    def f(t):
        c = Counter(last2[max(0, t - w):t])
        return top_k({p: c[p] for p in PAIRS})
    return f


def pair_freq_both(w):
    def f(t):
        c = Counter(last2[max(0, t - w):t]) + Counter(top2[max(0, t - w):t])
        return top_k({p: c[p] for p in PAIRS})
    return f


def overdue(t):
    sc = {}
    for p in PAIRS:
        g = next((t - 1 - i for i in range(t - 1, -1, -1) if last2[i] == p), t)
        sc[p] = g
    return top_k(sc)


def coldest_digits(w):
    def f(t):
        d1, d2 = Counter(), Counter()
        for i in range(max(0, t - w), t):
            d1[last2[i][0]] += 1; d2[last2[i][1]] += 1
        return top_k({p: -(d1[p[0]] * d2[p[1]]) for p in PAIRS})
    return f


def repeat_prev(t):
    return [last2[t - 1], top2[t - 1], last2[t - 1][::-1]]


def fixed(t):
    return ['00', '11', '22']


def hit(pred, t):  # ถูกถ้าตรง 2 ตัวล่าง หรือ 2 ตัวบน (ตรงกับที่เว็บนับ)
    return last2[t] in pred or top2[t] in pred


strategies = {'fixed 00/11/22': fixed, 'overdue': overdue, 'repeat_prev': repeat_prev}
for w in (10, 20, 30, 62, 100, 150, 250, 400):
    strategies[f'diind w={w}'] = diind(w)
    strategies[f'diind(bottom only) w={w}'] = diind(w, False)
    strategies[f'pair_freq w={w}'] = pair_freq(w)
    strategies[f'pair_freq both w={w}'] = pair_freq_both(w)
    strategies[f'coldest digits w={w}'] = coldest_digits(w)

START = 60
SPLIT = START + (N - START) // 2   # ครึ่งแรก = เลือกโมเดล, ครึ่งหลัง = ทดสอบจริง
rows = []
for name, f in strategies.items():
    h = [hit(f(t), t) for t in range(START, N)]
    a = h[:SPLIT - START]; b = h[SPLIT - START:]
    rows.append((name, sum(a) / len(a) * 100, sum(b) / len(b) * 100, sum(h) / len(h) * 100))

rnd = random.Random(1)
rand_h = []
for _ in range(200):
    rand_h.append(sum(hit(rnd.sample(PAIRS, K), t) for t in range(SPLIT, N)) / (N - SPLIT) * 100)

print(f"N={N} train(select)={SPLIT-START} test={N-SPLIT}  random baseline ~{sum(rand_h)/len(rand_h):.1f}%")
print(f"{'strategy':32s} {'train%':>7s} {'test%':>7s} {'all%':>7s}")
for r in sorted(rows, key=lambda x: -x[1]):
    print(f"{r[0]:32s} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.1f}")
best = max(rows, key=lambda x: x[1])
print("\nเลือกจาก train ดีสุด ->", best[0], f"train={best[1]:.1f}% test={best[2]:.1f}%")
print("ดีสุดบน test (เลือกโกง) ->", max(rows, key=lambda x: x[2])[0], f"{max(r[2] for r in rows):.1f}%")
